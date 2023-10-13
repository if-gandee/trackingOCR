import cv2
from PIL import Image, ImageDraw, ImageFont
import easyocr
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import numpy as np
from scipy.special import softmax
import time
import copy
from sklearn.cluster import KMeans

from find_obj import init_feature, filter_matches, explore_match, filter_matches_by_mean

from igpy.common.shortfunc import find_transform_similarity
import igpy.geometry.geometry as geom

class Defaults:
    NumPointsPerSampling = 6
    SampleTimes = 20
    MatchesFilterRatio = 0.5
    ParamChangeStep = 0.05
    OcrThreshold = 0.1
    TextFontSize = 30
    TextFontPath = "./font/simsun1.ttc"
    
    

def soften_matrix(h_matrix_candidates):
    '''
    This function takes a list of transformation matrices, h_matrix_candidates, and computes a softened average matrix.

    Args:
    h_matrix_candidates (list of 3x3 arrays): A list of transformation matrices to be averaged.

    Returns:
    h_matrix (3x3 array): The softened average transformation matrix.

    The function first checks if the input list is empty or None, and in such cases, it returns None as there is no data to process. For non-empty input lists, it performs the following steps:

    1. Decompose each transformation matrix into rotation (in radians), scale, and translation components.
    2. Compute the average scale, rotation, and translation for all matrices using a soft-max weighting scheme.
    3. Construct the softened average transformation matrix by combining the averaged scale, rotation, and translation components.

    The resulting average transformation matrix, h_matrix, represents the softened average of the input matrices.

    Note: This function assumes that the input matrices are 3x3 transformation matrices.
    '''
    if h_matrix_candidates is None or len(h_matrix_candidates) == 0:
        return None
    scale_list = []
    rot_list = []
    tran_list = []
    for h in h_matrix_candidates:
        rotmat, scale, tran = geom.decompose_transmat(h)
        rot = np.arctan2(rotmat[0, 1], rotmat[0, 0])
        scale_list.append(scale)
        rot_list.append(rot)
        tran_list.append(tran)
    scales = np.array(scale_list)
    rots = np.array(rot_list)
    trans = np.array(tran_list)
    avg_scale = np.sum(scales*softmax(-np.abs(scales-np.mean(scales, axis=0)), axis=0), axis=0)
    avg_rot = np.sum(rots*softmax(-np.abs(rots-np.mean(rots, axis=0)), axis=0), axis=0)
    avg_tran = np.sum(trans*softmax(-np.abs(trans-np.mean(trans, axis=0)), axis=0), axis=0)
    h_matrix = np.array([[avg_scale[0]*np.cos(avg_rot), -avg_scale[1]*np.sin(avg_rot), avg_tran[0]],
                        [avg_scale[0]*np.sin(avg_rot), avg_scale[1]*np.cos(avg_rot), avg_tran[1]],
                        [0, 0, 1]]).T
    return h_matrix
      
    
    

class OCRInfo:
    def __init__(self, text, points, confidence):
        self.text = text
        self.points = points.astype(np.int32)
        self.confidence = confidence
        self.h_matrix = np.eye(3)
        self.frame=None
        self.ref_kp = None
        self.ref_desc = None
    
    def init_local_info(self, img, detector, bbox_add=50):
        bbox = cv2.boundingRect(self.points)
        bbox = [bbox[0]-bbox_add, bbox[1]-bbox_add, bbox[2]+bbox_add*2, bbox[3]+bbox_add*2]
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.points], 255)
        img_masked = cv2.bitwise_and(img, img, mask=mask)
        kp, desc = detector.detectAndCompute(img_masked, None)
        self.frame = img_masked
        self.ref_kp = kp
        self.ref_desc = desc
        
        
    def kp_deepcopy(self):
        return [cv2.KeyPoint(x = k.pt[0], y = k.pt[1], 
                size = k.size, angle = k.angle, 
                response = k.response, octave = k.octave, 
                class_id = k.class_id) for k in self.ref_kp]

    def copy(self):
        self_copy = OCRInfo(self.text, self.points, self.confidence)
        self_copy.h_matrix = self.h_matrix.copy()
        if self.ref_kp is not None:
            self_copy.ref_kp = self.kp_deepcopy()
        if self.ref_desc is not None:
            self_copy.ref_desc = self.ref_desc.copy()
        return self_copy
    

class OcrReader:
    def __init__(self):
        self.reader = easyocr.Reader(['en', 'ch_sim'])
        self.m_ref_frame_info = None
        self.m_cur_frame_info = None
        self.detector, self.matcher = init_feature('brisk-flann')
        self.cur_frame_detector, _ = init_feature('brisk-flann')
        self.recognize_busy = False
        self.orb_process_busy = False
        self.ocr_executor = ThreadPoolExecutor(max_workers=2)
        self.orb_executor = ThreadPoolExecutor(max_workers=2)
        self.m_cur_lock = threading.Lock()
        self.m_ref_lock = threading.Lock()
        self.ocr_threshold = Defaults.OcrThreshold
        self.match_filter_ratio = Defaults.MatchesFilterRatio
        
    def read(self, img):
        
        if not self.recognize_busy:
            self.recognize_busy = True
            self.ocr_executor.submit(self.recognize, img)
            return

        if not self.orb_process_busy:
            self.orb_process_busy = True
            self.orb_executor.submit(self.orb_process, img)
    
    def get_ref_frame_info(self):
        if self.m_ref_frame_info is None:
            return None
        frame_info = {}
        with self.m_ref_lock:
            frame_info["frame"] = copy.deepcopy(self.m_ref_frame_info["frame"])
            frame_info["ocr"] = [_.copy() for _ in self.m_ref_frame_info["ocr"]]
        return frame_info
    
    def get_cur_frame_info(self):
        if self.m_cur_frame_info is None:
            return None
        frame_info = {}
        with self.m_cur_lock:
            frame_info["frame"] = copy.deepcopy(self.m_cur_frame_info["frame"])
            frame_info["kp_pairs"] = copy.deepcopy(self.m_cur_frame_info["kp_pairs"])
            frame_info["ocr"] = [_.copy() for _ in self.m_cur_frame_info["ocr"]]
        return frame_info
    
    def orb_process(self, img_ori):
        self.orb_process_busy = True
        img = img_ori.copy()
        if self.m_ref_frame_info is not None:
            all_kp_pairs=[]
            cur_ocr_list=[]
            with self.m_ref_lock:
                ref_ocr_list = [_.copy() for _ in self.m_ref_frame_info["ocr"]]
            cur_global_kp, cur_global_desc = self.cur_frame_detector.detectAndCompute(img, None)
            h_matrix = None
            for ref_ocr_info in ref_ocr_list:
                local_ref_kp = ref_ocr_info.ref_kp
                local_ref_desc = ref_ocr_info.ref_desc
                raw_matches = self.matcher.knnMatch(local_ref_desc, cur_global_desc, k=2)
                p1, p2, kp_pairs = filter_matches(local_ref_kp, cur_global_kp, raw_matches, self.match_filter_ratio)
                if len(p1) >= 6:
                    h_matrix_candidates = []
                    for _ in range(Defaults.SampleTimes):
                        try:
                            n=p1.shape[0]
                            rand_idx = np.random.choice(n, Defaults.NumPointsPerSampling, replace=False)
                            rp1 = p1[rand_idx]
                            rp2 = p2[rand_idx]
                            tmp_matrix = find_transform_similarity(rp1, rp2)
                            h_matrix_candidates.append(tmp_matrix)
                        except Exception as e:
                            tmp_matrix = None
                    h_matrix = soften_matrix(h_matrix_candidates)
                if h_matrix is not None:
                    all_kp_pairs += [_ for _ in zip(p1, p2)]
                    cur_points = cv2.perspectiveTransform(ref_ocr_info.points.reshape(-1, 1, 2).astype(np.float32), h_matrix.T).reshape(-1, 2)
                    cur_ocr_info = OCRInfo(ref_ocr_info.text, cur_points, ref_ocr_info.confidence)
                    cur_ocr_info.h_matrix = h_matrix
                    cur_ocr_list.append(cur_ocr_info)
                        
            with self.m_cur_lock:
                self.m_cur_frame_info = {
                    'frame': img,
                    'kp_pairs': all_kp_pairs,
                    'ocr': [_.copy() for _ in cur_ocr_list]
                }
        self.orb_process_busy = False

    def recognize(self, img):
        self.recognize_busy = True
        ocr_result = self.reader.readtext(img)
        ocr_list = []
        for res in ocr_result:
            ocr_info = OCRInfo(res[1], np.array(res[0]), res[2])
            ocr_info.init_local_info(img, self.detector)
            if ocr_info.confidence > self.ocr_threshold and ocr_info.ref_desc is not None and len(ocr_info.ref_desc) > 2:
                ocr_list.append(ocr_info)
        with self.m_ref_lock:
            self.m_ref_frame_info = {
                'frame': img,
                'ocr': [_.copy() for _ in ocr_list]
            }
        self.recognize_busy = False
            

if __name__ == '__main__':
    ocr_reader = OcrReader()
    capture = cv2.VideoCapture(0)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    debug_mode = False
    info_show = True
    while True:
        ret, frame = capture.read()
        if ret:
            ocr_reader.read(frame)
            cur_frame_info = ocr_reader.get_cur_frame_info()
            ref_frame_info = ocr_reader.get_ref_frame_info()
            show_frame = frame
            
            text_list = []
            if cur_frame_info is not None:
                show_frame = cur_frame_info['frame'].copy()
                kp_pairs = cur_frame_info['kp_pairs']
                ocr_list = cur_frame_info['ocr']
                for ocr_info in ocr_list:
                    points = ocr_info.points
                    text = ocr_info.text
                    confidence = ocr_info.confidence
                    if confidence > ocr_reader.ocr_threshold:
                        cv2.polylines(show_frame, [points.astype(np.int32)], True, (0, 255, 0), 2)
                        text_list.append([text, points])
                if debug_mode:
                    for p1, p2 in kp_pairs:
                        p1 = p1.astype(np.int32)
                        p2 = p2.astype(np.int32)
                        cv2.circle(show_frame, (p1[0], p1[1]), 2, (0, 255, 0), 2)
                        cv2.circle(show_frame, (p2[0], p2[1]), 2, (0, 0, 255), 2)
                        cv2.line(show_frame, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 2)
                        
            if debug_mode and ref_frame_info is not None:
                ocr_list = ref_frame_info['ocr']
                for ref_ocr_info in ocr_list:
                    points = ref_ocr_info.points
                    confidence = ref_ocr_info.confidence
                    if confidence > ocr_reader.ocr_threshold:
                        cv2.polylines(show_frame, [points.astype(np.int32)], True, (0, 0, 255), 2)

            pil_img = Image.fromarray(show_frame)
            draw = ImageDraw.Draw(pil_img)
            fontText = ImageFont.truetype(Defaults.TextFontPath, Defaults.TextFontSize, encoding="utf-8")
            if info_show:
                draw.text((20, 20), f"OCR检测阈值(快捷键“-、+”): {ocr_reader.ocr_threshold:.2f}", (255,0,0), font=fontText)
                draw.text((20, 50), f"匹配点筛选阈值(快捷键“1、2”): {1-ocr_reader.match_filter_ratio:.2f}", (255,0,0), font=fontText)
                draw.text((20, 80), f"Q键退出；D键开启/关闭debug模式；阈值越高越严格", (255,0,0), font=fontText)
            for text, points in text_list:
                draw.text((points[0][0], points[0][1]-30), text, (0,255,0), font=fontText)
            show_frame = np.array(pil_img)
            cv2.imshow("frame", show_frame)
            key_input =  cv2.waitKey(10)
            if key_input == ord('q'):
                break
            elif key_input == ord('-') or key_input == ord('_'):
                ocr_reader.ocr_threshold = max(0.0, ocr_reader.ocr_threshold - Defaults.ParamChangeStep)
            elif key_input == ord('=') or key_input == ord('+'):
                ocr_reader.ocr_threshold = min(1.0, ocr_reader.ocr_threshold + Defaults.ParamChangeStep)
            elif key_input == ord('0'):
                ocr_reader.match_filter_ratio = max(0.0, ocr_reader.match_filter_ratio - Defaults.ParamChangeStep)
            elif key_input == ord('9'):
                ocr_reader.match_filter_ratio = min(1.0, ocr_reader.match_filter_ratio + Defaults.ParamChangeStep)
            elif key_input == ord('d'):
                debug_mode = not debug_mode
            elif key_input == ord('s'):
                info_show = not info_show
    capture.release()
    cv2.destroyAllWindows()
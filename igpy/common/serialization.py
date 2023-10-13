import pickle
import base64

def encode_to_base64(x : object) -> str:
    ''' Encode an object to base64.
    '''
    return base64.b64encode(pickle.dumps(x)).decode('utf-8')

def decode_from_base64(x : str) -> object:
    ''' Decode an object from base64.
    '''
    return pickle.loads(base64.b64decode(x.encode('utf-8')))
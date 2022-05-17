import numpy as np
import cv2


def bts_to_img(bts):
    buff = np.fromstring(bts, np.uint8).reshape(1, -1)
    return cv2.imdecode(buff, cv2.IMREAD_COLOR)

def image_to_bts(frame):
    '''
    :param frame: WxHx3 ndarray
    '''
    _, bts = cv2.imencode('.webp', frame)
    return bts.tostring()



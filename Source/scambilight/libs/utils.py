import numpy as np
import cv2
import time
from contextlib import contextmanager
import random
import base64
# need these for other modules to load
from lumotag_utils import get_platform, _OS, TimeDiffObject

def convert_pts_to_convex_hull(points:list[list[int, int]]):
   return cv2.convexHull(np.array(points, dtype='int32'))

def ImageViewer_Quick_no_resize(inputimage,pausetime_Secs=0,presskey=False,destroyWindow=True):
    if inputimage is None:
        print("input image is empty")
        return
    ###handy quick function to view images with keypress escape andmore options
    cv2.imshow("img", inputimage.copy()); 


    if presskey==True:
        cv2.waitKey(0); #any key
   
    if presskey==False:
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                #for some reason
                pass
            
    if pausetime_Secs>0:
        time.sleep(pausetime_Secs)
    if destroyWindow==True: cv2.destroyAllWindows()


def str_to_bytes(string_: str):
    return str.encode(string_)

def bytes_to_str(bytes_: bytes):
    return bytes_.decode()

def encode_img_to_str(img: np.ndarray):
    """Encode single image for compatibility with json msg

    Args:
        thumb: image as numpy array

    Returns:
        str"""
    img_string = base64.b64encode(
            cv2.imencode(
                ext='.jpg',
                img=img)[1]).decode()
    return img_string

def img_width(img):
    if hasattr(img, 'shape'):
        return img.shape[1]
    else:
        return img[1]

def img_height(img):
    if hasattr(img, 'shape'):
        return img.shape[0]
    else:
        return img[0]

@contextmanager
def time_it_sparse(comment):
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        toc: float = time.perf_counter()
        if random.randint(1,1000) < 4:
            print(f"{comment}:proc time = {1000*(toc - tic):.3f}ms")


def decode_image_from_str(incoming_encoded_img: str):
    original = base64.b64decode(incoming_encoded_img)
    jpg_as_np = np.frombuffer(original, dtype=np.uint8)

    return cv2.imdecode(buf=jpg_as_np, flags=1)


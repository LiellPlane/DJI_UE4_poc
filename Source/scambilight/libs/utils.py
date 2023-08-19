import numpy as np
from abc import ABC, abstractmethod
import cv2
import time
import math
from dataclasses import dataclass, asdict
from time import perf_counter
from contextlib import contextmanager
import random
import enum
from typing import Optional
import requests
import base64
import json
from typing import Literal
import os

class _OS(str, enum.Enum):
    WINDOWS = "windows"
    RASPBERRY = "raspberry"

def get_platform():
    #  detect what OS we are on - test environment (Windows) or production (pi hardware)
    RASP_PI_4_OS = "armv7l"

    if hasattr(os, 'uname') is False:
        print("scambiloop raspberry presence failed, probably Windows system")
        return _OS.WINDOWS
    elif os.uname()[-1] == RASP_PI_4_OS:
        print("scambiloop raspberry presence detected, loading hardware libraries")
        return _OS.RASPBERRY
    else:
        raise Exception("Could not detect platform")

def convert_pts_to_convex_hull(points:list[list[int, int]]):
   return cv2.convexHull(np.array(points, dtype='int32'))

class TimeDiffObject:
    """stopwatch function"""

    def __init__(self) -> None:
        self._start_time = time.perf_counter()

    def get_dt(self) -> float:
        """gets time in seconds since last reset/init"""
        _stop_time = time.perf_counter()
        difference_ms = _stop_time-self._start_time
        return difference_ms

    def reset(self):
        self._start_time = time.perf_counter()


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
def time_it(comment):
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        toc: float = time.perf_counter()
        if random.randint(1,100) < 4:
            print(f"{comment}:proc time = {1000*(toc - tic):.3f}ms")
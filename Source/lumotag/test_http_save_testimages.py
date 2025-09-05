# functions to save incoming images from an http server and persist them in a specific manner
# each incoming image usually comes as a pair, but we will use the shape of the incoming image to determine
# what member of the pair it is, and save the image but generate a blank (black) image for the other member

# this way we can generate a suite of test data without polluting each cameras test data

# example close range filename: _closerange_cnt1256cnt1746219542_4913526.jpg
# example long range filename: _longrange_cnt3392cnt1746219601_8518264.jpg
import os
import numpy as np
import cv2
import time

TESTDATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "deadspot")

LONG_RANGE_DIMS = (1456, 1088)
CLOSE_RANGE_DIMS = (1920, 1080) 


class TestImageSaver:
    def __init__(self):
        self._blank_close = np.zeros((CLOSE_RANGE_DIMS[1], CLOSE_RANGE_DIMS[0]), dtype=np.uint8)
        self._blank_long = np.zeros((LONG_RANGE_DIMS[1], LONG_RANGE_DIMS[0]), dtype=np.uint8)
        self._counter = 1000
        
    def save_image_pair(self, image_array, image_id):
        h, w = image_array.shape[:2]
        close_filename = f"_closerange_cnt{self._counter}cnt{image_id}.jpg"
        long_filename = f"_longrange_cnt{self._counter}cnt{image_id}.jpg"
        if (w, h) == CLOSE_RANGE_DIMS:
            # Save close range image and blank long range
            cv2.imwrite(os.path.join(TESTDATA_PATH, close_filename), image_array)
            cv2.imwrite(os.path.join(TESTDATA_PATH, long_filename), self._blank_long)
        elif (w, h) == LONG_RANGE_DIMS:
            cv2.imwrite(os.path.join(TESTDATA_PATH, close_filename), self._blank_close)
            cv2.imwrite(os.path.join(TESTDATA_PATH, long_filename), image_array)
        
        self._counter += 1

# Auto-instantiate when imported
test_saver = TestImageSaver()




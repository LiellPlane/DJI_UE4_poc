import sys
import cv2
import shutil
from enum import Enum, auto
import os
import numpy as np
#sys.path.append(r"C:\Working\GIT\TestLab\TestLab")
#from matplotlib import pyplot as plt
import math
import math_utils
import random
import time
from contextlib import contextmanager
from typing import Iterator
import decode_clothID_v2
import random

class MouseHandler:
    
    def __init__(self, img):
        self.img = img.copy()
        self._prev_img = self.img
        self._all_points = {}
        self.points = []
        self._orig_img = img
        self.shape = "TRIANGLE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            self.img,
            self.shape,
            (50, 50),
            font,
            1.0,
            ( random.randint(0,200), 0, random.randint(0,200)),
            3,
            cv2.LINE_AA
            )
        
    def set_shape(self, shape):
        self.shape = shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            self.img,
            self.shape,
            (50, 50),
            font,
            1.0,
            ( random.randint(0,200), 0, random.randint(0,200)),
            3,
            cv2.LINE_AA
            )

    def __call__(self, event, x, y, *_):  # pylint: disable=invalid-name
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self._draw_point(len(self.points) - 1)

    def next_(self):
        if self.shape in self._all_points:
            self._all_points[self.shape].append(self.points.copy())
        else:
            self._all_points[self.shape] = [self.points.copy()]

        self.points = []
        print(self._all_points)
    def _draw_point(self, index):
        if self.shape == "TRIANGLE":
            colour = (0, 0, 255)
        if self.shape == "SQUARE": 
            colour = (255, 0, 0)
        #self.img = self._prev_img
        x, y = self.points[index]  # pylint: disable=invalid-name
        cv2.circle(self.img, (x, y), 4, colour, -1)
        if index:
            x_prev, y_prev = self.points[index - 1]
            cv2.line(self.img, (x_prev, y_prev), (x, y), (0, 255, 0), 2)
        self._prev_img = self.img.copy()
        # if len(self.points) >= 2:
        #     x0, y0 = self.points[0]  # pylint: disable=invalid-name
        #     cv2.line(self.img, (x, y), (x0, y0), (0, 255, 0), 2)


def ImageViewer_Quickv2_UserControl(inputimage,pausetime_Secs=0,presskey=False,destroyWindow=True):
    ###handy quick function to view images with keypress escape and more options
    if inputimage is None:
        return None
    cv2.imshow("img", inputimage); 
    UserRequest=""
    if presskey==True:
        cv2.imshow('img',inputimage)
        k = cv2.waitKey(33)
        
        #return character from keyboard input ascii code
        if k != -1:#no input detected
            try:
                UserRequest=(chr(k))
            except:
                UserRequest=None


    if presskey==False:
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                #for some reason
                pass
            
    if pausetime_Secs>0:
        time.sleep(pausetime_Secs)
    if destroyWindow==True: cv2.destroyAllWindows()

    return UserRequest

def FollowMouse(event, x, y, flags, param):
    '''Mouse interaction for OpenCV display'''
    pass
    # if event == cv2.EVENT_LBUTTONDOWN:
    #     #DataController.MouseTrack_MouseClick_LeftDown=True
    # if event == cv2.EVENT_RBUTTONDOWN:
    #     #DataController.MouseTrack_MouseClick_LeftDown=False


def main():
    input_imgs = decode_clothID_v2.GetAllFilesInFolder_Recursive(r"D:\lumotag_real_images")
   
    for img_filepath in input_imgs: 
        print(img_filepath)
        img = cv2.imread(img_filepath)
        cv2.namedWindow(wname := 'img')
        mouse_handler = MouseHandler(img)
        cv2.setMouseCallback(wname, mouse_handler)
        while True:
            res = ImageViewer_Quickv2_UserControl(mouse_handler.img, 0, True ,False)
            #print(res)
            if res == "s":
                mouse_handler.set_shape("SQUARE")
            if res == "t":
                mouse_handler.set_shape("TRIANGLE")
            if res == "n":
                mouse_handler.next_()
            if res == "r":
                break
if __name__ == '__main__':
    main()
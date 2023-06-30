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
from dataclasses import dataclass

def read_img(img_filepath):
    return cv2.imread(img_filepath)

def clahe_equalisation(img, claheprocessor):
    if claheprocessor is None:
        claheprocessor = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
    # colour
    if len(img.shape) >2:
        #luminosity
        lab_image=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab_image)
        #equ = cv2.equalizeHist(l)
        #updated_lab_img1=cv2.merge((equ,a,b))
        clahe_img= claheprocessor.apply(l)
        updated_lab_img1=cv2.merge((clahe_img,a,b))
        CLAHE_img = cv2.cvtColor(updated_lab_img1,cv2.COLOR_LAB2BGR)
    # grayscale
    else:
        CLAHE_img = claheprocessor.apply(img)
    return CLAHE_img

def _3_chan_equ(img):
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return equalized_img

def mono_img(img):
    if len(img.shape) < 3:
        return img
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def invert_img(img):
    return np.invert(img)

def equalise_img(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def blur_img(img, filtersize = 7):
    return cv2.GaussianBlur(img,(7,7),0)

def blur_average(img, filtersize = 7):
    kernel = np.ones((filtersize,filtersize),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    return dst

def normalise(img):
    image2_Norm = cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
    return image2_Norm

def threshold_img(img, low=0, high=255):
    #_ , th3 = cv2.threshold(img, low, 255,cv2.THRESH_BINARY)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,1)
    #_,th3 = cv2.threshold(img,low,high,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th3 = cv2.adaptiveThreshold(img,high,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return th3

def threshold_img_static(img, low=0, high=255):
    #_ , th3 = cv2.threshold(img, low, 255,cv2.THRESH_BINARY)
    #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,1)
    _,th3 = cv2.threshold(img,low,high,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th3 = cv2.adaptiveThreshold(img,high,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return th3

def edge_img(gray):
    #edges = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
    #edges = cv2.Canny(image=img, threshold1=100, threshold2=200)

    # Smoothing without removing edges.

    #     d- Diameter of each pixel neighborhood that is
    #     used during filtering. If it is non-positive, it is
    #     computed from sigmaSpace.
    # sigmaColor -Filter sigma in the color space. A larger value of the 
    # parameter means that farther colors within the pixel neighborhood
    # will be mixed together, resulting in larger areas of semi-equal color.

    # sigmaSpace - Filter sigma in the coordinate space. A larger value
    # of the parameter means that farther pixels will influence each other
    # as long as their colors are close enough. When d>0, it specifies the
    # neighborhood size regardless of sigmaSpace. Otherwise, d is proportional
    # to sigmaSpace.
    gray_filtered = cv2.bilateralFilter(gray, 5, 10, 4)

    # Applying the canny filter
    #edges = cv2.Canny(gray, 60, 120)
    edges_filtered = cv2.Canny(gray_filtered, 0, 60)

    # Stacking the images to print them together for comparison
    #images = np.hstack((gray, edges, edges_filtered))
    
    return edges_filtered

def simple_canny(blurred_img, lower, upper):
    # wide = cv2.Canny(blurred, 10, 200)
    # mid = cv2.Canny(blurred, 30, 150)
    # tight = cv2.Canny(blurred, 240, 250)
    return cv2.Canny(blurred_img, upper, lower)

def get_hist(img):
    #fig = plt.figure()
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #plt.hist(img.ravel(),256,[0,256]); plt.show()
    # plt.plot(hist)
    # plt.ylabel("histogram")
    # plt.ylim([0, max(hist)])
    # graph_image = np.array(fig.canvas.get_renderer()._renderer)
    # plt.cla()
    # plt.clf()
    # plt.close()
    
    # graph_image = np.array(fig.canvas.get_renderer()._renderer)
    return hist

def cut_square(img):
    length = 100 
    center_x = int(img.shape[0]/2)
    center_y = int(img.shape[1]/2)
    top = center_y - length
    lower = center_y + length
    left = center_x - length
    right = center_x + length
    cut = img[left:right,top:lower,:]
    return cut

def contours_img(img):
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    cv2.drawContours(out, contours, -1, 255,1)
    #cv2.drawContours(image=out, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    return out

def dilate(InputImage):
    kernel = np.ones((3, 3), np.uint8)
    img_blur = cv2.medianBlur(InputImage, 3)
    dilated_image = cv2.dilate(img_blur, kernel, iterations = 1)
    #eroded_image = cv2.erode(dilated_image, kernel, iterations = 5)
    return dilated_image

def median_blur(inputimage, kernalsize):
    return  cv2.medianBlur(inputimage, kernalsize)

def image_resize_ratio(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def resize_centre_img(image, screensize):

    # this is slow - might be faster passing in the image again?
    # TODO
    emptyscreen = np.zeros((screensize + (3,)), np.uint8)

    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if screensize[0] < screensize[1]:
        image = image_resize_ratio(
            image,
            height=screensize[0])
    else:
        image = image_resize_ratio(
            image,
            width=screensize[1])

    offset_x = (emptyscreen.shape[0] - image.shape[0]) // 2
    offset_y = (emptyscreen.shape[1] - image.shape[1]) // 2
    emptyscreen[
        offset_x:image.shape[0]+offset_x,
        offset_y:image.shape[1]+offset_y,
        :] = image

    return emptyscreen

def add_cross_hair(image, adapt):
    thick = 3
    midx = image.shape[0] // 2
    midy = image.shape[1] // 2
    # TODO another potential lag point
    if adapt is True:
        col = int(
            image[midx-50:midx+50, midy-50:midy+50, :].mean())
    else:
        col = 255
    image[midx-thick : midx+thick,:,:] = 0 
    image[midx-thick : midx+thick,:,1] = max(col, 50)
    image[:, midy-thick : midy+thick ,:] = 0 
    image[:, midy-thick : midy+thick ,1] = max(col, 50)

    return image

def get_internal_section(img, size: tuple[int, int]):
    midx = img.shape[0] // 2
    midy = img.shape[1] // 2
    regionx = size[0]//2
    regiony = size[1]//2
    return img[
        midx-regionx:midx+regionx,
        midy-regiony:midy+regiony]

def implant_internal_section(img, img_to_implant):

    if len(img.shape) < 3 and len(img_to_implant.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    midx = img.shape[0] // 2
    midy = img.shape[1] // 2
    regionx = img_to_implant.shape[0] // 2
    regiony = img_to_implant.shape[1] // 2
    img[midx-regionx:midx+regionx,
        midy-regiony:midy+regiony, :] = img_to_implant
    return img
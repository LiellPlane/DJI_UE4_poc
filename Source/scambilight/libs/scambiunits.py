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
import libs.fisheye_lib as fisheye_lib
from typing import Optional
import libs.async_cam_lib as async_cam_lib
import requests
import base64
import json
from typing import Literal
from libs.utils import get_platform, _OS, convert_pts_to_convex_hull
from libs.collections import Edges

@dataclass
class ScambiInit():
    led_positionxy: tuple
    sample_area_left: int
    sample_area_right: int
    sample_area_top: int
    sample_area_lower: int
    inverse_warp_m: any
    img_shape: Optional[any]
    img_circle: Optional[int]
    edge: Edges
    position_normed: float
    position_norm_start: float
    position_norm_end: float
    id: int

@dataclass
class ScambiWarp():
    roi = []
    warped_led_pos: any = None
    warped_bounding_rect: any = None
    bb_left: int = None
    bb_right: int = None
    bb_top: int = None
    bb_lower: int = None
    sample_area_lerp_contour: any = None
    convex_hulls_lerp_contour: any = None
    @property
    def bb_height(self):
        return abs(self.bb_lower-self.bb_top)
    @property
    def bb_width(self):
        return abs(self.bb_right-self.bb_left)
    

class Scambi_unit():
    def __init__(self,
                 init_object: ScambiInit):

        """supply led and roi positions on a flat representation of
        the screen
        
        inverse_warp will map these positions with the inverse homography
        if the screen isn't presented flat
        
        img_shape and img_circle are optional parameters that will 
        additionally warp the already perspective warped rois
        to a fisheye of img_circle
        
        needs to be manually initialised by calling .initialise"""
        self.initobj = init_object
        self.perpwarp = ScambiWarp()
        self.fishwarp = ScambiWarp()
        #self.roi = []
        #self.warped_led_pos = None
        #self.warped_bounding_rect = None
        #self.bb_left = None
        #self.bb_right = None
        #self.bb_top = None
        #self.bb_lower = None
        self.last_dom_col_dif = None
        self.colour = (
            random.randint(1,255),
            random.randint(1,255),
            random.randint(1,255))
        #self.sample_area_lerp_contour = None
        #self.convex_hulls_lerp_contour = None
        self.physical_led_pos = None
        
    def initialise(self):
        self.warp_rois_homograpy()
        self.fisheye_rois()
        print(f"scambiunit {self.initobj.id} initialised")

    def assign_physical_LED_pos(self, pos: int):
        self.physical_led_pos = pos

    @property
    def edge(self):
        return self.initobj.edge

    @property
    def position_normed(self):
        return self.initobj.position_normed
    
    @property
    def position_norm_start(self):
        return self.initobj.position_norm_start

    @property
    def position_norm_end(self):
        return self.initobj.position_norm_end

    def lerp_sample_area(self):
        """Create interpolation between corners so we can
        warp in a non-linear fashion such as fish eye"""
        contours = []
        # along top left to right
        lin = np.linspace(self.initobj.sample_area_left, self.initobj.sample_area_right, 10)
        for x in lin:
            contours.append((x, self.initobj.sample_area_top))
        # down right side
        lin = np.linspace(self.initobj.sample_area_top, self.initobj.sample_area_lower, 10)
        for y in lin:
            contours.append((self.initobj.sample_area_right, y))
        # along bottom right to left
        lin = np.linspace(self.initobj.sample_area_right, self.initobj.sample_area_left, 10)
        for x in lin:
            contours.append((x, self.initobj.sample_area_lower))
        # up left side
        lin = np.linspace(self.initobj.sample_area_lower, self.initobj.sample_area_top, 10)
        for y in lin:
            contours.append((self.initobj.sample_area_left, y))
        cont_ints = [(int(i[0]), int(i[1])) for i in contours]
        return cont_ints

    def warp_rois_homograpy(self):
        """warp rois if we just have a perspective warp"""
        # warp sample colour region
        temp_roi = []
        temp_roi.append(np.asarray([self.initobj.sample_area_left, self.initobj.sample_area_top, 1]))
        temp_roi.append(np.asarray([self.initobj.sample_area_left, self.initobj.sample_area_lower, 1]))
        temp_roi.append(np.asarray([self.initobj.sample_area_right, self.initobj.sample_area_lower, 1]))
        temp_roi.append(np.asarray([self.initobj.sample_area_right, self.initobj.sample_area_top, 1]))
        self.perpwarp.roi = []
        for pt in temp_roi:
            homog_coords = np.matmul(self.initobj.inverse_warp_m, pt)
            new_pt = list((np.floor(homog_coords[0:2]/homog_coords[-1])).astype(int))
            self.perpwarp.roi.append(new_pt)
        self.perpwarp.roi = np.asarray(self.perpwarp.roi).astype(np.int32)
        self.perpwarp.roi = self.perpwarp.roi.reshape((-1, 1, 2))
        # if self.perpwarp.roi.shape[0] != 4:
        #     print("something funky happening", self.perpwarp.roi)
        #     raise Exception("perpwarp.roi bad shape!! should only be 4 elements")
        # warp expected LED region
        temp_led_pos = np.asarray(self.initobj.led_positionxy + (1,))
        homog_coords = np.matmul(self.initobj.inverse_warp_m, temp_led_pos)
        self.perpwarp.warped_led_pos = tuple((np.floor(homog_coords[0:2]/homog_coords[-1])).astype(int))

        #get bounding box for ROI
        self.perpwarp.warped_bounding_rect = cv2.boundingRect(self.perpwarp.roi)
        left, top, w, h = self.perpwarp.warped_bounding_rect
        right = left + w
        lower = top + h
        self.perpwarp.bb_left = left
        self.perpwarp.bb_right = right
        self.perpwarp.bb_top = top
        self.perpwarp.bb_lower = lower

        #warp the lerped rois (rois with points between corners for non-linear warping)
        warped_lerp_roi = []
        lerp_area = self.lerp_sample_area()
        for lerp_pt in lerp_area:
            temp_lerp_pos = np.asarray(lerp_pt + (1,))
            homog_coords = np.matmul(self.initobj.inverse_warp_m, temp_lerp_pos)
            warped_lerp_pt = tuple((np.floor(homog_coords[0:2]/homog_coords[-1])).astype(int))
            warped_lerp_roi.append(warped_lerp_pt)
        self.perpwarp.sample_area_lerp_contour = warped_lerp_roi
        
    def fisheye_rois(self):
        """warp the rois to fisheye - expected to have
        warped perspective first"""
        
        fisheyeser = fisheye_lib.fisheye_tool(
            img_width_height=tuple(reversed(self.initobj.img_shape[0:2])),
            image_circle_size=self.initobj.img_circle)
        
        temp_pts = []

        for pt in self.perpwarp.sample_area_lerp_contour:
            temp_pts.append(
                fisheyeser.brute_force_find_fisheye_pt(pt))

        self.fishwarp.sample_area_lerp_contour = temp_pts
        self.fishwarp.convex_hulls_lerp_contour = convert_pts_to_convex_hull(
            self.fishwarp.sample_area_lerp_contour)

        self.fishwarp.warped_led_pos = fisheyeser.brute_force_find_fisheye_pt(self.perpwarp.warped_led_pos)
        
        #recalculate
        self.fishwarp.roi = np.asarray([np.asarray(fisheyeser.brute_force_find_fisheye_pt(list(pt[0]))) for pt in self.perpwarp.roi]).reshape(-1, 1, 2)
        self.fishwarp.warped_bounding_rect = cv2.boundingRect(self.fishwarp.roi)
        left, top, w, h = self.fishwarp.warped_bounding_rect
        right = left + w
        lower = top + h
        self.fishwarp.bb_left = left
        self.fishwarp.bb_right = right
        self.fishwarp.bb_top = top
        self.fishwarp.bb_lower = lower
    
    def draw_lerp_contour(self, img):

        cv2.drawContours(
            image=img,
            contours=[self.fishwarp.convex_hulls_lerp_contour],
            contourIdx=-1,
            color=(100,200,250),
            thickness=1,
            lineType=cv2.LINE_AA)
        
        return img

    def draw_warped_roi(self, img):
        #for pt in self.roi:
        #    img[pt[1], pt[0], :] = (255, 255, 255)
        img = cv2.polylines(img,
                            [self.fishwarp.roi],
                      isClosed=True, color=(random.randint(30,255),random.randint(30,255),random.randint(30,255)),
                      thickness=1)
        
        #cv2.drawContours(img, [br], -1, (0,0,255), 1)
        #cv2.circle(img,(br[2], br[3]),2,(255,255,255),-1)
        #cv2.rectangle(img,(br[0], br[1]),(br[2], br[3]),(0,255,0),3)
        return img
    
    def draw_warped_boundingbox(self,img):
        x, y, w, h = self.fishwarp.warped_bounding_rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 1)

        return img

    def draw_led_pos(self, img, colour=None, offset=None, size=16):

        pos = self.initobj.led_positionxy
        if offset is not None:
            pos = (pos[0] + offset[0], pos[1] + offset[1])
        if colour is None:
            cv2.circle(img,pos,size,(255,0,100),-1)
        else:
            cv2.circle(img,pos,size,colour,-1)

    def draw_warped_led_pos(self, img, colour=None, offset=None, size=16):
    
        pos = self.fishwarp.warped_led_pos
        if offset is not None:
            pos = (pos[0] + offset[0], pos[1] + offset[1])
        if colour is None:
            cv2.circle(img,pos,size,(255,0,100),-1)
        else:
            cv2.circle(img,pos,size,colour,-1)

        return img

    def draw_rectangle(self, img):
        draw_rectangle(
            self.initobj.sample_area_left,
            self.initobj.sample_area_right,
            self.initobj.sample_area_top,
            self.initobj.sample_area_lower, img)

    def get_dominant_colour_perspective(self, img):
        pass

    def get_mean_colour(self, img, subsample):
        sample_area = img[
            self.fishwarp.bb_top:self.fishwarp.bb_lower,
            self.fishwarp.bb_left:self.fishwarp.bb_right,
            :]
        
        self.colour = tuple(
            [int(i) for i in sample_area.mean(axis=0).mean(axis=0)])
        return self.colour

    def get_dom_colour_with_auto_subsample(self, img, cut_off):
        min_edge = min([self.fishwarp.bb_height, self.fishwarp.bb_width])
        subsampling = math.ceil(min_edge/cut_off)
        if subsampling < 1:
            raise Exception("problem with subsampling", self.fishwarp.bb_height, self.fishwarp.bb_width)
        return self.get_dominant_colour_flat(img, subsampling)

    def get_dominant_colour_flat(self, img, subsample):
        """keep subsample between 1 (unity) and 4 usually"""
        sample_area = img[
            self.fishwarp.bb_top:self.fishwarp.bb_lower:subsample,
            self.fishwarp.bb_left:self.fishwarp.bb_right:subsample,
            :]
        if random.randint(0,5000) < 2:
            print(f"{self.fishwarp.bb_lower-self.fishwarp.bb_top} * {self.fishwarp.bb_right-self.fishwarp.bb_left}")
        data = np.reshape(sample_area, (-1,3))
        data = np.float32(data)
        epsilon = 1.0
        max_iter = 10
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,max_iter,epsilon)
        flags = cv2.KMEANS_PP_CENTERS
        _, _, centers = cv2.kmeans(
            data=data,
            K=1,
            bestLabels=None,
            criteria=criteria,
            attempts=5,
            flags=flags)
        dom_col = centers[0].astype(np.int32)
        dom_col = [int(i) for i in dom_col]
        self.colour = tuple(dom_col)
        return self.colour

    def is_sample_area_smaller(self, cut_off: int):
        tp = self.fishwarp
        if (abs(tp.bb_right - tp.bb_left)) < cut_off or (abs(tp.bb_lower-tp.bb_top)) < cut_off:
            return True
        return False

def get_dominant_colour_flat_vectorize(img, list_of_scambiunits):
    return 
    for unit in list_of_scambiunits:

        sample_area = img[
            unit.bb_top:unit.bb_lower,
            unit.bb_left:unit.bb_right,
            :]
        plop=1
    if random.randint(0,500) < 2:
        print(f"{self.bb_lower-self.bb_top} * {self.bb_right-self.bb_left}")
    data = np.reshape(sample_area, (-1,3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(
        data=data,
        K=1,
        bestLabels=None,
        criteria=criteria,
        attempts=10,
        flags=flags)
    dom_col = centers[0].astype(np.int32)
    dom_col = [int(i) for i in dom_col]
    return tuple([int(i) for i in centers[0]])
    return (0, 0, 0)
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
from utils import time_it
from dataclasses import dataclass
from my_collections import (
    ShapeItem,
    Shapes,
    ShapeInfo_BulkProcess)
import img_processing as img_pro



def GetAllFilesInFolder_Recursive(root):
    ListOfFiles=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            FullpathOfFile=(os.path.join(path, name))
            ListOfFiles.append(FullpathOfFile)
    return ListOfFiles

def DeleteFiles_RecreateFolder(FolderPath):
    Deltree(FolderPath)
    os.mkdir(FolderPath)

def Deltree(Folderpath):
      # check if folder exists
    if len(Folderpath)<6:
        raise("Input:" + str(Folderpath),"too short - danger")
        raise ValueError("Deltree error - path too short warning might be root!")
        return
    if os.path.exists(Folderpath):
         # remove if exists
         shutil.rmtree(Folderpath)
    else:
         # throw your exception to handle this special scenario
         #raise Exception("Unknown Error trying to Deltree: " + Folderpath)
         pass
    return

class AutoStrEnum(str, Enum):
    """
    StrEnum where auto() returns the field name.
    See https://docs.python.org/3.9/library/enum.html#using-automatic-values
    """
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name

class Debug_Images(AutoStrEnum):
    ERROR_no_contours = auto()
    GOOD_CANDIDATE_ContourCount = auto()
    POSITIVE_ID_ContourCount = auto()
    unfiltered_contours = auto()
    Filtered_area_contours = auto()
    filtered_circularity_contours = auto()
    original_input = auto()
    clahe_equalisation = auto()
    initial_thresh = auto()
    input_to_contours = auto()
    macro_candidates = auto()
    all_contours = auto()
    fitered_contours = auto()
    ID_BADGE = auto()
    find_shape = auto()


class WorkingData():
    def __init__(self, debug=False, debugimgs=r"D:/lumodebug/") -> None:
        self.debugimgs = debugimgs
        self.debug = debug
        self.debug_img_cnt = 0
        self.debug_subfldr = None
        if self.debug is True:
            DeleteFiles_RecreateFolder(self.debugimgs)
        self.claheprocessor = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
        self.approx_epsilon = 0.02
    @staticmethod
    def get_blob_params():
        DefaultBlobParams= cv2.SimpleBlobDetector_Params()
        DefaultBlobParams.filterByArea = True
        DefaultBlobParams.minArea = 40
        DefaultBlobParams.maxArea = 30000
        DefaultBlobParams.minDistBetweenBlobs = 2
        DefaultBlobParams.filterByColor = False
        DefaultBlobParams.filterByConvexity = False
        DefaultBlobParams.minCircularity = 0.5
        DefaultBlobParams.filterByInertia = False
        DefaultBlobParams.minInertiaRatio = 0.4
        return DefaultBlobParams

    def img_view_or_save_if_debug(self, img, description, resize = True):
        
        if self.debug is True:
            out_img = img.copy()
            if resize is True:
                resize_x =  int(img.shape[1]*(1000/img.shape[1]))
                resize_y =  int(img.shape[0]*(1000/img.shape[1]))
                out_img = cv2.resize(out_img, (resize_x,resize_y), interpolation = cv2.INTER_AREA)
            if self.debug_subfldr is None:
                filename = f"{self.debugimgs}\\0{self.debug_img_cnt}_{description}.jpg"
            else:
                if not os.path.exists(f"{self.debugimgs}\\{self.debug_subfldr}"):
                    os.mkdir(f"{self.debugimgs}\\{self.debug_subfldr}")
                filename = f"{self.debugimgs}\\{self.debug_subfldr}\\0{self.debug_img_cnt}_{description}.jpg"
            cv2.imwrite(filename,out_img)
            print(f"DEBUG = TRUE: saving debug file to {filename}")
            self.debug_img_cnt += 1



def draw_pattern_output(image, patterndetails: ShapeItem):
    """draw graphics for user if a pattern is found
    TODO: maybe want floating numbers etc above this which
    will eventually need a user registry"""
    min_bbox = patterndetails.boundingbox_min
    cX, cY = patterndetails.centre_x_y

    # corners of square
    cv2.circle(image, tuple(min_bbox[0]), 3, img_pro.RED, 1)
    cv2.circle(image, tuple(min_bbox[2]), 3, img_pro.RED, 1)
    cv2.circle(image, tuple(min_bbox[1]), 3, img_pro.RED, 1)
    cv2.circle(image, tuple(min_bbox[3]), 3, img_pro.RED, 1)

    # centre of pattern
    cv2.circle(image, (cX, cY), 5, img_pro.RED, 1)
   
    # bounding box of contour - this does not handle perspective
    cv2.drawContours(image, [min_bbox], 0, img_pro.RED)

    #draw barcode sampling lines - for illustration only
    # may not match exactly with generated sampled lines
    cv2.line(image, tuple(min_bbox[0]), tuple(min_bbox[2]), img_pro.RED, 1) 
    cv2.line(image, tuple(min_bbox[1]), tuple(min_bbox[3]), img_pro.RED, 1) 


def get_approx_shape_and_bbox_bulk(
        contours,
        dataobject : WorkingData) -> ShapeInfo_BulkProcess:

    bulkprocess = ShapeInfo_BulkProcess()

    pts_0 = np.zeros((len(contours), 2), dtype="int32")
    pts_1 = np.zeros((len(contours), 2), dtype="int32")
    pts_2 = np.zeros((len(contours), 2), dtype="int32")

    for index, c in enumerate(contours):
        bulkprocess.contour[index] = c
        bulkprocess.convex_hull_contour[index] =cv2.convexHull(c)
        bulkprocess.minRect[index] = cv2.minAreaRect(c)
        bulkprocess.min_bbox[index] = cv2.boxPoints(bulkprocess.minRect[index])
        bulkprocess.min_bbox[index] = np.intp(bulkprocess.min_bbox[index]).astype(int)
        bulkprocess.contour_pxl_cnt[index] = cv2.contourArea(c)
        bulkprocess.min_bbox_pxl_cnt[index] = cv2.contourArea(bulkprocess.min_bbox[index])
        # cv2.arcLength() is used to calculate the perimeter of the contour.
        # If the second argument is True then it considers the contour to be closed.
        # Then this perimeter is used to calculate the epsilon value for cv2.approxPolyDP() 
        # function with a precision factor for approximating a shape
        bulkprocess.approx_contour[index] = cv2.approxPolyDP(
            c,
            dataobject.approx_epsilon*cv2.arcLength(c, True),
            True)

        pts_0[index] = bulkprocess.min_bbox[index][0]
        pts_1[index] = bulkprocess.min_bbox[index][1]
        pts_2[index] = bulkprocess.min_bbox[index][2]

    bulkprocess.dists_0_to_1 = np.sqrt(np.sum((pts_0 - pts_1)**2, axis=1))
    bulkprocess.dists_1_to_2 = np.sqrt(np.sum((pts_1 - pts_2)**2, axis=1))

    return bulkprocess
            
def get_approx_shape_and_bbox(
        contour,
        img,
        dataobject : WorkingData,
        index = 0) -> ShapeItem:
    # cv2.arcLength() is used to calculate the perimeter of the contour.
    # If the second argument is True then it considers the contour to be closed.
    # Then this perimeter is used to calculate the epsilon value for cv2.approxPolyDP() 
    # function with a precision factor for approximating a shape
    
    contour = cv2.convexHull(contour)
    
    # filter first by minimum bounding box of raw contours:
    minRect = cv2.minAreaRect(contour)
    min_bbox = cv2.boxPoints(minRect)
    min_bbox = np.intp(min_bbox).astype(int)
    contour_pxl_cnt = cv2.contourArea(contour)
    
    min_bbox_pxl_cnt = cv2.contourArea(min_bbox)
    # filter by how much of ideal square is taken up by contour area
    # with extreme perspective this will not be sufficient
    
    if contour_pxl_cnt < (min_bbox_pxl_cnt * 0.50):
        # try and repair it
        
        if dataobject.debug is True:
            #img_debug = img.copy()
            #img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2BGR)
            #cv2.drawContours(img_debug, [contour, min_bbox], 0, (0,0,255))
            #dataobject.img_view_or_save_if_debug(img_debug, "not_enough_pixels_for_sqr")
            
            return ShapeItem(
                    id=index,
                    approx_contour=None,
                    default_contour=contour,
                    filtered_contour=None,
                    boundingbox=None,
                    boundingbox_min=min_bbox,
                    boundingbox_ellipse=None,
                    img_cut=None,
                    sum_int_angles=None,
                    size=contour_pxl_cnt,
                    min_bbx_size=cv2.contourArea(min_bbox),
                    shape=Shapes.BAD_PIXELS,
                    centre_x_y=None,
                    _2d_samples=None,
                    notes_for_debug_file="not_enough_pixels_for_sqr")
        return None

    approx = cv2.approxPolyDP(
        contour,
        dataobject.approx_epsilon*cv2.arcLength(contour, True),
        True)
    

    if len(approx) not in [4, 5, 6, 7, 8]:
        return ShapeItem(
            id=index,
            approx_contour=approx,
            default_contour=contour,
            filtered_contour=None,
            boundingbox=None,
            boundingbox_min=min_bbox,
            boundingbox_ellipse=None,
            img_cut=None,
            sum_int_angles=None,
            size=contour_pxl_cnt,
            min_bbx_size=cv2.contourArea(min_bbox),
            shape=Shapes.BAD_APPROX_LEN,
            centre_x_y=None,
            _2d_samples=None,
            notes_for_debug_file=f"bad_approxlen")
    
    
    #taking a wild guess for rotated rectangle - can;t be far off
    w = np.linalg.norm(min_bbox[0]-min_bbox[1])
    h = np.linalg.norm(min_bbox[1]-min_bbox[2])
    #w = np.sqrt(np.sum((min_bbox[0]-min_bbox[1])**2))
    #h = np.sqrt(np.sum((min_bbox[1]-min_bbox[2])**2))
    #x, y, w, h = cv2.boundingRect(contour)
    
    # filter by ratio
    if w < 10 or h < 10:
        return None
    

    if w/h < 0.1 or w/h > 9:
        if dataobject.debug is True:
            #img_debug = img.copy()
            #img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2BGR)
            #cv2.drawContours(img_debug, [contour, min_bbox], 0, (0,0,255))
            ratio = str(round(w/h, 3))
            ratio = ratio.replace(".", "p")
            #dataobject.img_view_or_save_if_debug(img_debug, f"bad_ratio{ratio}")
            
            return ShapeItem(
                    id=index,
                    approx_contour=None,
                    default_contour=contour,
                    filtered_contour=None,
                    boundingbox=None,
                    boundingbox_min=min_bbox,
                    boundingbox_ellipse=None,
                    img_cut=None,
                    sum_int_angles=None,
                    size=contour_pxl_cnt,
                    min_bbx_size=cv2.contourArea(min_bbox),
                    shape=Shapes.BAD_RATIO,
                    centre_x_y=None,
                    _2d_samples=None,
                    notes_for_debug_file=f"bad_ratio{ratio}")
        return None
    


    # filtered_cont = None
    # # filter close together points, sometimes outlier doesnt tend to work?
    # res, val = math_utils.filter_close_points(approx)
    # if res is True:
    #     filtered_cont = val



    # occasionally we get a triangle or square with a blunt edge,
    # so remove this extra point by filtering outlier distances

    # if filtered_cont is None:
    #     res, val = math_utils.filter_outlier_edges(approx)
    # else:
    #     res, val = math_utils.filter_outlier_edges(filtered_cont)
    # if res is True:
    #     filtered_cont = val
    #int_angle = int(math_utils.get_internal_angles_of_shape(approx))
    
    # minRect = cv2.minAreaRect(approx)
    # min_bbox = cv2.boxPoints(minRect)
    # min_bbox = np.intp(min_bbox)
    # min_bbox_pxl_cnt = cv2.contourArea(min_bbox)

    # contour_pxl_cnt = cv2.contourArea(contour)
    # can't make ellipse with <5 points
    # if len(contour) > 4:
    #     ellipse = cv2.fitEllipse(contour)
    # else:
    #     ellipse = None
    #unaligned_bbx = cv2.boundingRect(contour)

    # get centre

    output = None
    shape_ = Shapes.UNKNOWN
    #test for square
    # TODO rough at moment
    # this is a pattern which is square with an inner circle




    if len(approx) in [4, 5, 6, 7, 8]:

        minRect = cv2.minAreaRect(approx)
        min_bbox = cv2.boxPoints(minRect)
        min_bbox = np.intp(min_bbox)
        min_bbox_pxl_cnt = cv2.contourArea(min_bbox)
        contour_pxl_cnt = cv2.contourArea(contour)


        if contour_pxl_cnt <= (min_bbox_pxl_cnt * 0.80):
            return ShapeItem(
                id=index,
                approx_contour=approx,
                default_contour=contour,
                filtered_contour=None,
                boundingbox=None,
                boundingbox_min=min_bbox,
                boundingbox_ellipse=None,
                img_cut=None,
                sum_int_angles=None,
                size=contour_pxl_cnt,
                min_bbx_size=cv2.contourArea(min_bbox),
                shape=Shapes.BAD_APPROX_PXL,
                centre_x_y=None,
                _2d_samples=None,
                notes_for_debug_file=f"BAD_APPROX_PXL")

        if contour_pxl_cnt > (min_bbox_pxl_cnt * 0.80):
            
            # we know we have a square - lets see if it 
            # has the internal inverse colour circle pattern
            
            moments = cv2.moments(contour)
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            # perimeter_10pc = cv2.arcLength(contour, True) * 0.1
            
            
                 # arbitrary edge ratio range

                    #dataobject.img_view_or_save_if_debug(sqr_sample_area, "SQuare_centre")

            
            sample_line1 = img_pro.bresenham_line_ski(
                x1=min_bbox[0][0],
                y1=min_bbox[0][1],
                x2 = min_bbox[2][0],
                y2 = min_bbox[2][1])

            sample_line2 = img_pro.bresenham_line_ski(
                x1=min_bbox[1][0],
                y1=min_bbox[1][1],
                x2 = min_bbox[3][0],
                y2 = min_bbox[3][1])

            
            averages = []
            averages2 = []
            pixel_div_count = 90
            _step = max(int((math.floor(len(sample_line1)) / pixel_div_count)), 1)
            sample_size = 1
            for i in range (sample_size, len(sample_line1)-sample_size, _step):
                sample_area = img[sample_line1[i][1]-sample_size:sample_line1[i][1]+sample_size, sample_line1[i][0]-sample_size: sample_line1[i][0]+sample_size]
                averages.append(sample_area.mean())
            for i in range (sample_size, len(sample_line2)-sample_size, _step):
                sample_area = img[sample_line2[i][1]-sample_size:sample_line2[i][1]+sample_size, sample_line2[i][0]-sample_size: sample_line2[i][0]+sample_size]
                averages2.append(sample_area.mean())

            # if dataobject.debug is True:
            #     img_debug = img.copy()
            #     cv2.circle(img_debug, (cX, cY), 5, 255, 1)
            #     crop_img = img_debug[y:y+h, x:x+w]
            #     dataobject.img_view_or_save_if_debug(crop_img, "SquareFound")
            #     cv2.circle(img_debug, tuple(min_bbox[0]), 3, 255, 1)
            #     cv2.circle(img_debug, tuple(min_bbox[2]), 3, 255, 1)
            #     cv2.circle(img_debug, tuple(min_bbox[1]), 3, 0, 1)
            #     cv2.circle(img_debug, tuple(min_bbox[3]), 3, 0, 1)
            #     try:
            #         for xy, ave_col in zip(sample_line1, averages):
            #             img_debug[xy[1]-50,xy[0]-50] = ave_col
            #         for xy, ave_col in zip(sample_line2, averages):
            #             img_debug[xy[1],xy[0]+50] = ave_col
            #         for xy in sample_line1:
            #             img_debug[xy[1],xy[0]] = 255
            #         for xy in sample_line2:
            #             img_debug[xy[1],xy[0]] = 255
            #     except Exception:
            #         pass



            #     cv2.drawContours(img_debug, [min_bbox], 0, 255)
            #     dataobject.img_view_or_save_if_debug(img_debug, "testline")
            #     crop_img = img_debug[y:y+h, x:x+w]
            #     dataobject.img_view_or_save_if_debug(crop_img, "corners of square")
            shape_ = Shapes.SQUARE
    
# if len(approx) in [3, 4, 5, 6]:
#     if contour_pxl_cnt > (min_bbox_pxl_cnt * 0.40):
#         if contour_pxl_cnt < (min_bbox_pxl_cnt * 0.60):
#             shape_ = Shapes.TRIANGLE

            output = ShapeItem(
                id=index,
                approx_contour=approx,
                default_contour=None,
                filtered_contour=None,
                boundingbox=None,
                boundingbox_min=min_bbox,
                boundingbox_ellipse=None,
                img_cut=None,
                sum_int_angles=None,
                size=contour_pxl_cnt,
                min_bbx_size = cv2.contourArea(min_bbox),
                shape=shape_,
                centre_x_y=[cX, cY],
                _2d_samples=[averages, averages2],
                notes_for_debug_file=None)
    
    return output


def get_approx_shape_and_bbox2(
        img,
        img_blurred,
        dataobject : WorkingData,
        index: int,
        bulk_process: ShapeInfo_BulkProcess) -> ShapeItem:

    contour_pxl_cnt = bulk_process.contour_pxl_cnt[index]
    min_bbox_pxl_cnt = bulk_process.min_bbox_pxl_cnt[index]
    contour = bulk_process.contour[index]
    min_bbox = bulk_process.min_bbox[index]
    approx = bulk_process.approx_contour[index]
    if contour_pxl_cnt < (min_bbox_pxl_cnt * 0.50):
        # try and repair it
        
        if dataobject.debug is True:
            #img_debug = img.copy()
            #img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2BGR)
            #cv2.drawContours(img_debug, [contour, min_bbox], 0, (0,0,255))
            #dataobject.img_view_or_save_if_debug(img_debug, "not_enough_pixels_for_sqr")
            
            return ShapeItem(
                    id=index,
                    approx_contour=None,
                    default_contour=contour,
                    filtered_contour=None,
                    boundingbox=None,
                    boundingbox_min=min_bbox,
                    boundingbox_ellipse=None,
                    img_cut=None,
                    sum_int_angles=None,
                    size=contour_pxl_cnt,
                    min_bbx_size=cv2.contourArea(min_bbox),
                    shape=Shapes.BAD_PIXELS,
                    centre_x_y=None,
                    _2d_samples=None,
                    notes_for_debug_file="not_enough_pixels_for_sqr")
        return None

    
    if len(approx) not in [4, 5, 6, 7, 8]:
        return ShapeItem(
            id=index,
            approx_contour=approx,
            default_contour=contour,
            filtered_contour=None,
            boundingbox=None,
            boundingbox_min=min_bbox,
            boundingbox_ellipse=None,
            img_cut=None,
            sum_int_angles=None,
            size=contour_pxl_cnt,
            min_bbx_size=cv2.contourArea(min_bbox),
            shape=Shapes.BAD_APPROX_LEN,
            centre_x_y=None,
            _2d_samples=None,
            notes_for_debug_file=f"bad_approxlen")
    
    
    #taking a wild guess for rotated rectangle - can;t be far off
    w = bulk_process.dists_0_to_1[index]
    h = bulk_process.dists_1_to_2[index]
    #w = np.sqrt(np.sum((min_bbox[0]-min_bbox[1])**2))
    #h = np.sqrt(np.sum((min_bbox[1]-min_bbox[2])**2))
    #x, y, w, h = cv2.boundingRect(contour)
    
    # filter by ratio
    if w < 10 or h < 10:
        return None
    

    if w/h < 0.1 or w/h > 9:
        if dataobject.debug is True:
            #img_debug = img.copy()
            #img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2BGR)
            #cv2.drawContours(img_debug, [contour, min_bbox], 0, (0,0,255))
            ratio = str(round(w/h, 3))
            ratio = ratio.replace(".", "p")
            #dataobject.img_view_or_save_if_debug(img_debug, f"bad_ratio{ratio}")
            
            return ShapeItem(
                    id=index,
                    approx_contour=None,
                    default_contour=contour,
                    filtered_contour=None,
                    boundingbox=None,
                    boundingbox_min=min_bbox,
                    boundingbox_ellipse=None,
                    img_cut=None,
                    sum_int_angles=None,
                    size=contour_pxl_cnt,
                    min_bbx_size=cv2.contourArea(min_bbox),
                    shape=Shapes.BAD_RATIO,
                    centre_x_y=None,
                    _2d_samples=None,
                    notes_for_debug_file=f"bad_ratio{ratio}")
        return None
    


    # filtered_cont = None
    # # filter close together points, sometimes outlier doesnt tend to work?
    # res, val = math_utils.filter_close_points(approx)
    # if res is True:
    #     filtered_cont = val



    # occasionally we get a triangle or square with a blunt edge,
    # so remove this extra point by filtering outlier distances

    # if filtered_cont is None:
    #     res, val = math_utils.filter_outlier_edges(approx)
    # else:
    #     res, val = math_utils.filter_outlier_edges(filtered_cont)
    # if res is True:
    #     filtered_cont = val
    #int_angle = int(math_utils.get_internal_angles_of_shape(approx))
    
    # minRect = cv2.minAreaRect(approx)
    # min_bbox = cv2.boxPoints(minRect)
    # min_bbox = np.intp(min_bbox)
    # min_bbox_pxl_cnt = cv2.contourArea(min_bbox)

    # contour_pxl_cnt = cv2.contourArea(contour)
    # can't make ellipse with <5 points
    # if len(contour) > 4:
    #     ellipse = cv2.fitEllipse(contour)
    # else:
    #     ellipse = None
    #unaligned_bbx = cv2.boundingRect(contour)

    # get centre

    output = None
    shape_ = Shapes.UNKNOWN
    #test for square
    # TODO rough at moment
    # this is a pattern which is square with an inner circle




    if len(approx) in [4, 5, 6, 7, 8]:

        minRect = cv2.minAreaRect(approx)
        min_bbox = cv2.boxPoints(minRect)
        min_bbox = np.intp(min_bbox)
        min_bbox_pxl_cnt = cv2.contourArea(min_bbox)
        contour_pxl_cnt = cv2.contourArea(contour)


        if contour_pxl_cnt <= (min_bbox_pxl_cnt * 0.80):
            return ShapeItem(
                id=index,
                approx_contour=approx,
                default_contour=contour,
                filtered_contour=None,
                boundingbox=None,
                boundingbox_min=min_bbox,
                boundingbox_ellipse=None,
                img_cut=None,
                sum_int_angles=None,
                size=contour_pxl_cnt,
                min_bbx_size=cv2.contourArea(min_bbox),
                shape=Shapes.BAD_APPROX_PXL,
                centre_x_y=None,
                _2d_samples=None,
                notes_for_debug_file=f"BAD_APPROX_PXL")

        if contour_pxl_cnt > (min_bbox_pxl_cnt * 0.80):
            
            # we know we have a square - lets see if it 
            # has the internal inverse colour circle pattern
            
            moments = cv2.moments(contour)
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            # perimeter_10pc = cv2.arcLength(contour, True) * 0.1
            
            
                 # arbitrary edge ratio range

                    #dataobject.img_view_or_save_if_debug(sqr_sample_area, "SQuare_centre")

            
            sample_line1 = img_pro.bresenham_line_ski(
                x1=min_bbox[0][0],
                y1=min_bbox[0][1],
                x2 = min_bbox[2][0],
                y2 = min_bbox[2][1])

            sample_line2 = img_pro.bresenham_line_ski(
                x1=min_bbox[1][0],
                y1=min_bbox[1][1],
                x2 = min_bbox[3][0],
                y2 = min_bbox[3][1])

       
            averages = []
            averages2 = []
            pixel_div_count = 90
            _step = max(int((math.floor(len(sample_line1)) / pixel_div_count)), 1)
            sample_size = 1
            if contour_pxl_cnt > 1600:
                img2use = img_blurred
            else:
                img2use = img

            samples_per_line = 20
            
            for i in range (sample_size, len(sample_line1)-sample_size, _step):
                averages.append(img2use[np.clip(sample_line1[i][1], 1,img2use.shape[0]-1), np.clip(sample_line1[i][0], 1,img2use.shape[1]-1)])
                #averages.append(img2use[sample_line1[i][0], sample_line1[i][0]])
            for i in range (sample_size, len(sample_line2)-sample_size, _step):
                averages2.append(img2use[np.clip(sample_line2[i][1], 1,img2use.shape[0]-1), np.clip(sample_line2[i][0], 1,img2use.shape[1]-1)])

            # for i in range (sample_size, len(sample_line1)-sample_size, _step):
            #     averages.append(img2use[sample_line1[i][1], sample_line1[i][0]])
            # for i in range (sample_size, len(sample_line2)-sample_size, _step):
            #     averages2.append(img2use[sample_line2[i][1], sample_line2[i][0]])
            
            # if dataobject.debug is True:
            #     img_debug = img.copy()
            #     cv2.circle(img_debug, (cX, cY), 5, 255, 1)
            #     crop_img = img_debug[y:y+h, x:x+w]
            #     dataobject.img_view_or_save_if_debug(crop_img, "SquareFound")
            #     cv2.circle(img_debug, tuple(min_bbox[0]), 3, 255, 1)
            #     cv2.circle(img_debug, tuple(min_bbox[2]), 3, 255, 1)
            #     cv2.circle(img_debug, tuple(min_bbox[1]), 3, 0, 1)
            #     cv2.circle(img_debug, tuple(min_bbox[3]), 3, 0, 1)
            #     try:
            #         for xy, ave_col in zip(sample_line1, averages):
            #             img_debug[xy[1]-50,xy[0]-50] = ave_col
            #         for xy, ave_col in zip(sample_line2, averages):
            #             img_debug[xy[1],xy[0]+50] = ave_col
            #         for xy in sample_line1:
            #             img_debug[xy[1],xy[0]] = 255
            #         for xy in sample_line2:
            #             img_debug[xy[1],xy[0]] = 255
            #     except Exception:
            #         pass



            #     cv2.drawContours(img_debug, [min_bbox], 0, 255)
            #     dataobject.img_view_or_save_if_debug(img_debug, "testline")
            #     crop_img = img_debug[y:y+h, x:x+w]
            #     dataobject.img_view_or_save_if_debug(crop_img, "corners of square")
            shape_ = Shapes.SQUARE
    
# if len(approx) in [3, 4, 5, 6]:
#     if contour_pxl_cnt > (min_bbox_pxl_cnt * 0.40):
#         if contour_pxl_cnt < (min_bbox_pxl_cnt * 0.60):
#             shape_ = Shapes.TRIANGLE

            output = ShapeItem(
                id=index,
                approx_contour=approx,
                default_contour=None,
                filtered_contour=None,
                boundingbox=None,
                boundingbox_min=min_bbox,
                boundingbox_ellipse=None,
                img_cut=None,
                sum_int_angles=None,
                size=contour_pxl_cnt,
                min_bbx_size = cv2.contourArea(min_bbox),
                shape=shape_,
                centre_x_y=[cX, cY],
                _2d_samples=[averages, averages2],
                notes_for_debug_file=None)
    
    return output


def debug_save_images(img, contours, text : str, dataobject: WorkingData):
    if dataobject.debug is True:
        img_check_contours = img.copy()
        img_check_contours = np.zeros_like(cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB))
        for i, cnt in enumerate([i for i in contours]):
            cv2.drawContours(
                image=img_check_contours,
                contours=[cnt],
                contourIdx=-1,
                color=(random.randint(20,255),random.randint(20,255),random.randint(20,255)),
                thickness=1,
                lineType=cv2.LINE_AA)
        dataobject.img_view_or_save_if_debug(img_check_contours, text)


def get_possible_candidates(img, dataobject : WorkingData):
    """get all contours of image, and filter to remove noise
    
    provide thresholded image (might have to inverted to avoid segments
    on edge of image being classed as external), will filter contours for circularity"""
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html

    smallest_area = (img.shape[0]*0.01) *  (img.shape[1]*0.01)
    largest_area = (img.shape[0]*0.9) *  (img.shape[1]*0.9)

    dataobject.img_view_or_save_if_debug(img, Debug_Images.input_to_contours.value)
    #  get all contours, 
    with time_it("get possible candidates: find contours"):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"get possible candidates: {len(contours)} contours found")
    debug_save_images(img, contours, Debug_Images.unfiltered_contours.value, dataobject)


    with time_it("get possible candidates: filter"):
        # filter by area
        contours_area = []
        hierarchy_area = []
        if hierarchy is None or len(contours) == 0:
            return [], []
        for con, hier in zip(contours, hierarchy[0]):
            area = cv2.contourArea(con)
            #print(area)
            if (area > smallest_area) and (area < largest_area):
                contours_area.append(con)
                hierarchy_area.append(hier)
            else:
                pass
                #print("rejeted area", area)
                #debug_save_images(img, [con], "rejected_area", dataobject)
        if len(contours_area) != len(hierarchy_area):
            raise Exception("bad unzip - use python 3.10 for strict=true")

        debug_save_images(img, contours_area, Debug_Images.Filtered_area_contours.value, dataobject)
        #ff
        # filter by circularity - * warning might filter out very fuzzy images
        contours_cirles = []
        hierarchy_cirles = []
        for con, hier in zip(contours_area, hierarchy_area):
            perimeter = cv2.arcLength(con, True)
            area = cv2.contourArea(con)
            if perimeter == 0:
                break
            circularity = 4*math.pi*(area/(perimeter*perimeter))
            if circularity > 0.2:
                contours_cirles.append(con)
                hierarchy_cirles.append(hier)
        if len(contours_cirles) != len(hierarchy_cirles):
            raise Exception("bad unzip - use python 3.10 for strict=true")
    debug_save_images(img, contours_cirles, Debug_Images.filtered_circularity_contours.value, dataobject)
    print(f"get possible candidates: {len(contours_cirles)} contours postfilter")

    if dataobject.debug is True:
        out = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        cv2.drawContours(image=out, contours=contours_cirles, contourIdx=-1, color=(0, 255, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        dataobject.img_view_or_save_if_debug(out, Debug_Images.macro_candidates.value)

    return contours_cirles, hierarchy_cirles


def get_dist(sample: list[int]):
    std_d=np.std(sample)
    mean=np.mean(sample)
    return std_d, mean

def has_child_contour(hierarchy: np.array):
    """pass in hierarchy for contour
    array([63, 24, 26, -1], dtype=int32)
    
    position 0 1 [2] 3 is the child
    
    in the example above this contour has a 
    child contour #26. -1 is code for no
    child"""
    if hierarchy[2] < 1:
        return False
    return True

def analyse_candidates_shapematch(
        original_img,
        original_blurred_image,
        contours : tuple [np.ndarray],
        dataobject : WorkingData,
        contour_hierarchy : tuple [np.ndarray]):
    """ For each input contour, try and match to a primary shape"""
    debug_save_images(original_img, contours, "input_to_shape_matcher", dataobject)
    # FILTER OUT SHAPES WITH CHILDREN
    # MIND THAT SMALL NOISE MIGHT BE PRESENT AS CHILDREN
    # IN REMANING HIERARCHY! Need to improve filtering complexity
    # contours_nochild = []
    # hierarchy_nochild = []
    # for cnt, hier in zip(contours, contour_hierarchy):
    #     if has_child_contour(hier) is False:
    #         contours_nochild.append(cnt)
    #         hierarchy_nochild.append(hier)

    # debug_save_images(original_img, contours_nochild, "no_childs", dataobject)

    contour_stats = []
    # with time_it("AC: get approx shape"):
    #     for index, c in enumerate(contours):
    #         contour_stats.append(get_approx_shape_and_bbox(
    #             c,
    #             original_img,
    #             dataobject,
    #             index))

    with time_it("AC: get approx shape 2"):
        bulk_process = get_approx_shape_and_bbox_bulk(
                    contours,
                    dataobject)

        for index, c in enumerate(contours):
            contour_stats.append(get_approx_shape_and_bbox2(
                original_img,
                original_blurred_image,
                dataobject,
                index,
                bulk_process))
    plop=1
    # if dataobject.debug == True:
    #     debug_img = original_img.copy()
    #     debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
    #     for c in contour_stats:
    #         cv2.drawContours(debug_img, [c.approx_contour], -1, (0,0,255), 1)
    #         x,y,w,h = c.boundingbox
    #         ROI = debug_img[y:y+h, x:x+w]
    #         dataobject.img_view_or_save_if_debug(
    #             ROI,
    #             f"check_shape_extract_{c.sum_int_angles}d_{c.id}")


    tote_samples = []
    squrs_found = [cont for cont in contour_stats if cont is not None and cont.shape == Shapes.SQUARE]

    def eb34(list1):
        flat_list = []
        for i in list1:
            if isinstance(i, list):
                for j in eb34(i):
                    flat_list.append(j)
            else:
                flat_list.append(i)
        return flat_list
    samples_all = eb34([x._2d_samples for x in squrs_found])
    print(f"Sample points: {len(samples_all)/2}")
    #tote_samples [x in i._2d_samples for i in squrs_found]

    print("total samples: ", len(tote_samples))
    if dataobject.debug == True:
        #img_bbxoes = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
        #img_bbxoes_2 = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
        #img_bbxoes_3 = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
        # for c in contour_stats:
        #     if c is None: continue
        #     cv2.drawContours(img_bbxoes, [c.boundingbox_min], 0, (0,0,255))
        #     #if c.boundingbox_ellipse is not None:
        #     #    cv2.ellipse(img_bbxoes_2, c.boundingbox_ellipse,(0,255,0))
        #     cv2.drawContours(img_bbxoes_3, [c.approx_contour], 0, (255,0,255))
        #dataobject.img_view_or_save_if_debug(img_bbxoes, "bounding_boxes")
        #dataobject.img_view_or_save_if_debug(img_bbxoes_2, "fit_ellipse")
        #dataobject.img_view_or_save_if_debug(img_bbxoes_3, "approx_shape")

        # break out triangles and squares
        
        BAD_PIXELS = [cont for cont in contour_stats if cont is not None and cont.shape == Shapes.BAD_PIXELS]
        BAD_RATIO = [cont for cont in contour_stats if cont is not None and cont.shape == Shapes.BAD_RATIO]
        BAD_APPROX_LEN = [cont for cont in contour_stats if cont is not None and cont.shape == Shapes.BAD_APPROX_LEN]
        BAD_APPROX_PXL =  [cont for cont in contour_stats if cont is not None and cont.shape == Shapes.BAD_APPROX_PXL]
        debug_img = original_img.copy()
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        for c in squrs_found:
            cv2.drawContours(debug_img, [c.approx_contour], -1, (0,255,0), 2)
        dataobject.img_view_or_save_if_debug(
            debug_img,
            f"squares_found")

        debug_img = original_img.copy()
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        for c in BAD_PIXELS:
            cv2.drawContours(debug_img, [c.default_contour], -1, (0,0,255), 2)
            cv2.drawContours(debug_img, [c.boundingbox_min.reshape(4,1,2).astype(int)], -1, (255,0,0), 1)
        dataobject.img_view_or_save_if_debug(
            debug_img,
            f"BAD_PIXELS")

        debug_img = original_img.copy()
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        for c in BAD_RATIO:
            cv2.drawContours(debug_img, [c.default_contour], -1, (255,0,0), 2)
            cv2.drawContours(debug_img, [c.boundingbox_min.reshape(4,1,2).astype(int)], -1, (255,0,0), 1)
        dataobject.img_view_or_save_if_debug(
            debug_img,
            f"BAD_RATIO")

        debug_img = original_img.copy()
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        for c in BAD_APPROX_LEN:
            cv2.drawContours(debug_img, [c.approx_contour], -1, (0,255,0), 2)
        dataobject.img_view_or_save_if_debug(
            debug_img,
            f"BAD_APPROX_LEN")
        
        debug_img = original_img.copy()
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        for c in BAD_APPROX_PXL:
            cv2.drawContours(debug_img, [c.approx_contour], -1, (0,255,0), 2)
        dataobject.img_view_or_save_if_debug(
            debug_img,
            f"BAD_APPROX_PXL")
        

        # break out individual squares found:

        for c in squrs_found:
            #try:
            if c.size > 1600:
                img2use = original_blurred_image
            else:
                img2use = original_img
            debug_img = img2use.copy()
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
            w = int(np.linalg.norm(c.boundingbox_min[0]-c.boundingbox_min[1]))
            h = int(np.linalg.norm(c.boundingbox_min[1]-c.boundingbox_min[2]))
            x = c.centre_x_y[0]
            y = c.centre_x_y[1]
            draw_pattern_output(debug_img, c)
            cv2.drawContours(debug_img, [c.approx_contour], -1, (0,255,0), 1)
            crop_img =  debug_img[max(0,y-h):y+h, max(0,x-w):x+w]
            if len([True for i in crop_img.shape if i == 0]) > 0:
                plop=1
                pass
            dataobject.img_view_or_save_if_debug(crop_img, "SquareFound")
            out_img = cv2.resize(np.asarray(c._2d_samples[0]), (200,500))
            dataobject.img_view_or_save_if_debug(out_img, "squarecode")
            out_img = cv2.resize(np.asarray(c._2d_samples[1]), (200,500))
            dataobject.img_view_or_save_if_debug(out_img, "squarecode")
            #except Exception:
             #   print("error with debug contour outputs")
        # if  len(squrs_found) > 0:
        #     debug_img = original_img.copy()
        #     debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        #     for c in squrs_found:
        #         cv2.drawContours(debug_img, [c.approx_contour], -1, (0,255,0), 2)
        #     dataobject.img_view_or_save_if_debug(
        #         debug_img,
        #         f"shapes_found_sqr")

        # if  len(tris_found) > 0:
        #     debug_img = original_img.copy()
        #     debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        #     for c in tris_found:
        #         cv2.drawContours(debug_img, [c.approx_contour], -1, (0,0,255), 2)
        #     dataobject.img_view_or_save_if_debug(
        #         debug_img,
        #         f"shapes_found_tri")

        # if len(unknown_found) > 0:
        #     debug_img = original_img.copy()
        #     debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        #     for c in unknown_found:
        #         cv2.drawContours(debug_img, [c.approx_contour], -1, (255,0,0), 2)
        #     dataobject.img_view_or_save_if_debug(
        #         debug_img,
        #         f"shapes_found_tri")
        # if dataobject.debug == True:
        #     debug_img = original_img.copy()
        #     debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        #     for c in filtered_objs:
        #             cv2.drawContours(debug_img, [c.filtered_contour], -1, (255,0,0), 2)
        #     dataobject.img_view_or_save_if_debug(
        #         debug_img,
        #         f"filtered_shapes")
            
        #for i, c in enumerate(contours):
        #    _, img_bbxoes = check_shape(c, dataobject, img_bbxoes, 0)
            
        #dataobject.img_view_or_save_if_debug(img_bbxoes, f"checkshape")
    #output_colour = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    # output_contour_data = []
    # for c in contour_stats:
    #     if c is None: continue
    #     draw_pattern_output(image=output_colour, patterndetails=c)
    #     output_contour_data.append(c)
    #output_contour_data = [c for c in contour_stats if c is not None]
    # for c in unknown_found:
    #     cv2.drawContours(output_colour, [c.approx_contour], -1, (30,0,90), 3)
    # for c in squrs_found:
    #     cv2.drawContours(output_colour, [c.approx_contour], -1, (0,255,0), 3)
    # for c in tris_found:
    #     cv2.drawContours(output_colour, [c.approx_contour], -1, (0,0,255), 3)
    return squrs_found


def block_filter_highfreq_areas(cannyied_img, block_pc, max_white_per_block, original_image):
    """expects a canny image or whatever results in edges (high frequencies)"""
    imgx = cannyied_img.shape[0]
    imgy = cannyied_img.shape[1]
    xrange = [i for i in range(0, imgx, int(imgx*(block_pc/100)//1))]
    xrange += [imgx for i in [xrange[-1]] if i != imgx]
    yrange = [i for i in range(0, imgy, int(imgy*(block_pc/100)//1))]
    yrange += [imgy for i in [xrange[-1]] if i != imgy]
    max = cannyied_img.max()
    for xdex in range (0, len(xrange)-1):
        for ydex in range (0, len(yrange)-1):
            testarea = cannyied_img[
                xrange[xdex]: xrange[xdex+1],
                yrange[ydex]: yrange[ydex+1]]
            #b = np.array(random.choices(testarea, k=2))
            if testarea.mean() > max_white_per_block:
                cannyied_img[
                xrange[xdex]: xrange[xdex+1],
                yrange[ydex]: yrange[ydex+1]] = 0
                original_image[
                xrange[xdex]: xrange[xdex+1],
                yrange[ydex]: yrange[ydex+1]] = 0

    return cannyied_img, original_image

def find_lumotag(inputimg, dataobject : WorkingData):

    """analyse input image for specific lumotag pattern"""
    #~2ms
    with time_it("pre-processing: total"):
        with time_it("grayscale"):
            if len(inputimg.shape)>2:
                img_grayscale = cv2.cvtColor(inputimg,cv2.COLOR_BGR2GRAY)
            else:
                img_grayscale = inputimg
        dataobject.img_view_or_save_if_debug(inputimg, Debug_Images.original_input.value, resize=False)
        #copy original image into folder
        #orig_img = img.copy()
        
        #~3ms for grayscale
    
        #print("equalisation")
        with time_it("pre-processing: blur"):
            #img_op = cv2.blur(img_grayscale,(3,3)) # fastest filter
            img_op = cv2.medianBlur(img_grayscale, 7)
            dataobject.img_view_or_save_if_debug(img_op, "blur_7_7", resize=False)

        # with time_it("pre-processing/filtering: clahe_equalisation"):
        #     img_op=img_pro.clahe_equalisation(img_op, dataobject.claheprocessor)
        #     dataobject.img_view_or_save_if_debug(img_op, Debug_Images.clahe_equalisation.value, resize=False)
        #     ''''test area'''
   
   #this section about 25ms
    #with time_it():
        
        #gray_orig = img_pro.mono_img(orig_img)
    #with time_it():
        #print("median_blur")
        ##blurred = median_blur(gray_orig,7)

        #dataobject.img_view_or_save_if_debug(squr_img, "median_blur", resize=False)
        #edge_im = edge_img(blurred)edge_img
    #with time_it():
        #print("canny_filter")
        #blurred = median_blur(gray_orig,7)
        #plop = edge_img(squr_img)
        #dataobject.img_view_or_save_if_debug(plop, "canny_filter")
        #edge_im = edge_img(blurred)edge_img
    #with time_it():
        #print("threshold_img")


        #squr_img=edge_img(gray_orig)
        with time_it("pre-processing: threshold_img"):
            #img_op=img_pro.threshold_img_static(img_op,low=40,high=255)
            img_op=img_pro.threshold_img(img_op,low=40,high=255)
            # squr_img=img_pro.simple_canny(
            #     blurred_img=squr_img,
            #     lower=0,
        #     upper=255)

        dataobject.img_view_or_save_if_debug(img_op, "thresholdimg")
    #with time_it():
        #print("invert_img")
        #squr_img=invert_img(squr_img)
        #dataobject.img_view_or_save_if_debug(squr_img, "invert_img")
        with time_it("pre-processing: blur again"):
            #img_op = cv2.blur(img_grayscale,(3,3)) # fastest filter
            img_op = cv2.medianBlur(img_op, 3)
            dataobject.img_view_or_save_if_debug(img_op,"blur_3_3_again", resize=False)

        with time_it("pre-processing: blur orig for sampler"):
            #img_op = cv2.blur(img_grayscale,(3,3)) # fastest filter
            org_img_grayscale_blur = cv2.medianBlur(img_grayscale, 5)
            dataobject.img_view_or_save_if_debug(org_img_grayscale_blur, "blur_for_sampling", resize=False)

    with time_it("get_possible_candidates total"):
        contours, hierarchy=get_possible_candidates(img_op, dataobject)

    # if len(contours) == 0:
    #     print("no results found for image")
    #     return []

    with time_it("analyse_candidates TOTAL"):
        output_contour_data = analyse_candidates_shapematch(
                                                original_img=inputimg,
                                                original_blurred_image=org_img_grayscale_blur,
                                                contours = contours,
                                                contour_hierarchy = hierarchy,
                                                dataobject = dataobject)
    # if analyse_IDs is not None:
    #     dataobject.img_view_or_save_if_debug(analyse_IDs, Debug_Images.ID_BADGE.value)
    #     return analyse_IDs, playerfound
    
    return output_contour_data



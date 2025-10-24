import check_barcode
import cv2
from scipy.signal import find_peaks
from typing import Tuple, List, Optional
from enum import Enum, auto
import os
import numpy as np
from numpy.typing import NDArray
#sys.path.append(r"C:\Working\GIT\TestLab\TestLab")
#from matplotlib import pyplot as plt
import math
from itertools import chain
import random
import functools 
from utils import (
    time_it,
    custom_print,
    DeleteFiles_RecreateFolder
)
from dataclasses import dataclass
from my_collections import (
    ShapeItem,
    Shapes,
    ShapeInfo_BulkProcess)
import img_processing as img_pro
from configs import base_find_lumotag_config
#from sklearn.neighbors import KDTree

MIN_TAG_VARIANCE = 25 # OBSOLETE? max-min for grayscale values of lumotag - peaks so might be
MAX_PATTERN_SYMMETRY_ERROR = 5 # OBSOLETE?
SAMPLES_PER_LINE = 20 # this is for how many samples we do along each diagonal of the barcode

ContourType = NDArray[np.int32]  # Shape: (N, 1, 2) where N is number of points

# For multiple contours (what findContours returns)
ContoursType = List[ContourType]

def GetAllFilesInFolder_Recursive(root):
    ListOfFiles=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            FullpathOfFile=(os.path.join(path, name))
            ListOfFiles.append(FullpathOfFile)
    return ListOfFiles


class AutoStrEnum(str, Enum):
    """
    StrEnum where auto() returns the field name.
    See https://docs.python.org/3.9/library/enum.html#using-automatic-values
    """
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name

@dataclass
class Spokes:
    """for square barcode, each spoke emanates from the centre"""
    line_pts: list[np.ndarray, np.ndarray]
    barcode_segment: check_barcode.CodeSegment


@dataclass
class SpokesData(Spokes):
    line_sample_pts: list[np.ndarray]
    
    @classmethod
    def from_base(cls, base: Spokes, line_sample_pts: list[np.ndarray]):
        return cls(**base.__dict__, line_sample_pts=line_sample_pts)
# @dataclass
# class SpokesData(Spokes):
#     line_sample_pts: list[np.ndarray]


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
    def __init__(
            self,
            OS_friendly_name: str,
            debugdetails: base_find_lumotag_config) -> None:
        self.debug_img_cnt = 0
        self.debug_subfldr = None
        self.debug_details = debugdetails
        # TODO potential error assuming that file system can handle forward slashes
        # as the config file is meant for different OS's
        self.debug_details.SAVE_IMAGES_PATH += f"{OS_friendly_name}/"
        if self.debug_details.SAVE_IMAGES_DEBUG is True:
            DeleteFiles_RecreateFolder(self.debug_details.SAVE_IMAGES_PATH)
        self.claheprocessor = cv2.createCLAHE(
            clipLimit=1.0, tileGridSize=(32, 32)
            )
        self.approx_epsilon = 0.02

    @staticmethod
    def get_blob_params():
        DefaultBlobParams = cv2.SimpleBlobDetector_Params()
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
        
        if self.debug_details.SAVE_IMAGES_DEBUG is True:
            out_img = img.copy()
            if resize is True:
                resize_x = int(img.shape[1]*(1000/img.shape[1]))
                resize_y = int(img.shape[0]*(1000/img.shape[1]))
                out_img = cv2.resize(out_img, (resize_x,resize_y), interpolation = cv2.INTER_AREA)
            if self.debug_subfldr is None:
                filename = f"{self.debug_details.SAVE_IMAGES_PATH}\\0{self.debug_img_cnt}_{description}.jpg"
            else:
                if not os.path.exists(f"{self.debug_details.SAVE_IMAGES_PATH}\\{self.debug_subfldr}"):
                    os.mkdir(f"{self.debug_details.SAVE_IMAGES_PATH}\\{self.debug_subfldr}")
                filename = f"{self.debug_details.SAVE_IMAGES_PATH}\\{self.debug_subfldr}\\0{self.debug_img_cnt}_{description}.jpg"
            cv2.imwrite(filename, out_img)
            print(f"DEBUG = TRUE: saving debug file to {filename}")
            self.debug_img_cnt += 1



# def draw_pattern_output(image, patterndetails: ShapeItem):
#     """draw graphics for user if a pattern is found
#     TODO: maybe want floating numbers etc above this which
#     will eventually need a user registry"""
#     min_bbox = patterndetails.boundingbox_min
#     cX, cY = patterndetails.centre_x_y
#     closest_corners = patterndetails.closest_corners
#     # corners of square
#     cv2.circle(image, tuple(min_bbox[0]), 3, img_pro.RED, 1)
#     cv2.circle(image, tuple(min_bbox[2]), 3, img_pro.RED, 1)
#     cv2.circle(image, tuple(min_bbox[1]), 3, img_pro.RED, 1)
#     cv2.circle(image, tuple(min_bbox[3]), 3, img_pro.RED, 1)


#     # centre of pattern
#     cv2.circle(image, (cX, cY), 5, img_pro.RED, 1)
   
#     # bounding box of contour - this does not handle perspective
#     cv2.drawContours(image, [min_bbox], 0, img_pro.RED)

#     #draw barcode sampling lines - for illustration only
#     # may not match exactly with generated sampled lines
#     cv2.line(image, tuple(closest_corners[0]), tuple(closest_corners[2]), img_pro.RED, 1) 
#     cv2.line(image, tuple(closest_corners[1]), tuple(closest_corners[3]), img_pro.RED, 1) 


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
            
# def get_approx_shape_and_bbox(
#         contour,
#         img,
#         dataobject : WorkingData,
#         index = 0) -> ShapeItem:
#     # cv2.arcLength() is used to calculate the perimeter of the contour.
#     # If the second argument is True then it considers the contour to be closed.
#     # Then this perimeter is used to calculate the epsilon value for cv2.approxPolyDP() 
#     # function with a precision factor for approximating a shape
    
#     contour = cv2.convexHull(contour)
    
#     # filter first by minimum bounding box of raw contours:
#     minRect = cv2.minAreaRect(contour)
#     min_bbox = cv2.boxPoints(minRect)
#     min_bbox = np.intp(min_bbox).astype(int)
#     contour_pxl_cnt = cv2.contourArea(contour)
    
#     min_bbox_pxl_cnt = cv2.contourArea(min_bbox)
#     # filter by how much of ideal square is taken up by contour area
#     # with extreme perspective this will not be sufficient
    
#     if contour_pxl_cnt < (min_bbox_pxl_cnt * 0.50):
#         # try and repair it
        
#         if dataobject.debug is True:
#             #img_debug = img.copy()
#             #img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2BGR)
#             #cv2.drawContours(img_debug, [contour, min_bbox], 0, (0,0,255))
#             #dataobject.img_view_or_save_if_debug(img_debug, "not_enough_pixels_for_sqr")
            
#             return ShapeItem(
#                     id=index,
#                     approx_contour=None,
#                     default_contour=contour,
#                     filtered_contour=None,
#                     boundingbox=None,
#                     boundingbox_min=min_bbox,
#                     sample_positions=None,
#                     closest_corners=None,
#                     sum_int_angles=None,
#                     size=contour_pxl_cnt,
#                     min_bbx_size=cv2.contourArea(min_bbox),
#                     shape=Shapes.BAD_PIXELS,
#                     centre_x_y=None,
#                     _2d_samples=None,
#                     notes_for_debug_file="not_enough_pixels_for_sqr")
#         return None

#     approx = cv2.approxPolyDP(
#         contour,
#         dataobject.approx_epsilon*cv2.arcLength(contour, True),
#         True)
    

#     if len(approx) not in [4, 5, 6, 7, 8]:
#         return ShapeItem(
#             id=index,
#             approx_contour=approx,
#             default_contour=contour,
#             filtered_contour=None,
#             boundingbox=None,
#             boundingbox_min=min_bbox,
#             sample_positions=None,
#             closest_corners=None,
#             sum_int_angles=None,
#             size=contour_pxl_cnt,
#             min_bbx_size=cv2.contourArea(min_bbox),
#             shape=Shapes.BAD_APPROX_LEN,
#             centre_x_y=None,
#             _2d_samples=None,
#             notes_for_debug_file=f"bad_approxlen")
    
    
#     #taking a wild guess for rotated rectangle - can;t be far off
#     w = np.linalg.norm(min_bbox[0]-min_bbox[1])
#     h = np.linalg.norm(min_bbox[1]-min_bbox[2])
#     #w = np.sqrt(np.sum((min_bbox[0]-min_bbox[1])**2))
#     #h = np.sqrt(np.sum((min_bbox[1]-min_bbox[2])**2))
#     #x, y, w, h = cv2.boundingRect(contour)
    
#     # filter by ratio
#     if w < 10 or h < 10:
#         return None
    

#     if w/h < 0.1 or w/h > 9:
#         if dataobject.debug_details.PRINT_DEBUG is True:
#             #img_debug = img.copy()
#             #img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2BGR)
#             #cv2.drawContours(img_debug, [contour, min_bbox], 0, (0,0,255))
#             ratio = str(round(w/h, 3))
#             ratio = ratio.replace(".", "p")
#             #dataobject.img_view_or_save_if_debug(img_debug, f"bad_ratio{ratio}")
            
#             return ShapeItem(
#                     id=index,
#                     approx_contour=None,
#                     default_contour=contour,
#                     filtered_contour=None,
#                     boundingbox=None,
#                     boundingbox_min=min_bbox,
#                     sample_positions=None,
#                     closest_corners=None,
#                     sum_int_angles=None,
#                     size=contour_pxl_cnt,
#                     min_bbx_size=cv2.contourArea(min_bbox),
#                     shape=Shapes.BAD_RATIO,
#                     centre_x_y=None,
#                     _2d_samples=None,
#                     notes_for_debug_file=f"bad_ratio{ratio}")
#         return None
    


#     # filtered_cont = None
#     # # filter close together points, sometimes outlier doesnt tend to work?
#     # res, val = math_utils.filter_close_points(approx)
#     # if res is True:
#     #     filtered_cont = val



#     # occasionally we get a triangle or square with a blunt edge,
#     # so remove this extra point by filtering outlier distances

#     # if filtered_cont is None:
#     #     res, val = math_utils.filter_outlier_edges(approx)
#     # else:
#     #     res, val = math_utils.filter_outlier_edges(filtered_cont)
#     # if res is True:
#     #     filtered_cont = val
#     #int_angle = int(math_utils.get_internal_angles_of_shape(approx))
    
#     # minRect = cv2.minAreaRect(approx)
#     # min_bbox = cv2.boxPoints(minRect)
#     # min_bbox = np.intp(min_bbox)
#     # min_bbox_pxl_cnt = cv2.contourArea(min_bbox)

#     # contour_pxl_cnt = cv2.contourArea(contour)
#     # can't make ellipse with <5 points
#     # if len(contour) > 4:
#     #     ellipse = cv2.fitEllipse(contour)
#     # else:
#     #     ellipse = None
#     #unaligned_bbx = cv2.boundingRect(contour)

#     # get centre

#     output = None
#     shape_ = Shapes.UNKNOWN
#     #test for square
#     # TODO rough at moment
#     # this is a pattern which is square with an inner circle




#     if len(approx) in [4, 5, 6, 7, 8]:

#         minRect = cv2.minAreaRect(approx)
#         min_bbox = cv2.boxPoints(minRect)
#         min_bbox = np.intp(min_bbox)
#         min_bbox_pxl_cnt = cv2.contourArea(min_bbox)
#         contour_pxl_cnt = cv2.contourArea(contour)


#         if contour_pxl_cnt <= (min_bbox_pxl_cnt * 0.80):
#             return ShapeItem(
#                 id=index,
#                 approx_contour=approx,
#                 default_contour=contour,
#                 filtered_contour=None,
#                 boundingbox=None,
#                 boundingbox_min=min_bbox,
#                 sample_positions=None,
#                 closest_corners=None,
#                 sum_int_angles=None,
#                 size=contour_pxl_cnt,
#                 min_bbx_size=cv2.contourArea(min_bbox),
#                 shape=Shapes.BAD_APPROX_PXL,
#                 centre_x_y=None,
#                 _2d_samples=None,
#                 notes_for_debug_file=f"BAD_APPROX_PXL")

#         if contour_pxl_cnt > (min_bbox_pxl_cnt * 0.80):
            
#             # we know we have a square - lets see if it 
#             # has the internal inverse colour circle pattern
            
#             moments = cv2.moments(contour)
#             cX = int(moments["m10"] / moments["m00"])
#             cY = int(moments["m01"] / moments["m00"])
#             # perimeter_10pc = cv2.arcLength(contour, True) * 0.1
            
            
#                  # arbitrary edge ratio range

#                     #dataobject.img_view_or_save_if_debug(sqr_sample_area, "SQuare_centre")

            
#             sample_line1_diag = img_pro.efficient_line_sampler(
#                 x1=min_bbox[0][0],
#                 y1=min_bbox[0][1],
#                 x2 = min_bbox[2][0],
#                 y2 = min_bbox[2][1],
#                 25)

#             sample_line2_diag = img_pro.efficient_line_sampler(
#                 x1=min_bbox[1][0],
#                 y1=min_bbox[1][1],
#                 x2 = min_bbox[3][0],
#                 y2 = min_bbox[3][1],
#                 25)

            
#             averages = []
#             averages2 = []
#             pixel_div_count = 90
#             _step = max(int((math.floor(len(sample_line1_diag)) / pixel_div_count)), 1)
#             sample_size = 1
#             for i in range (sample_size, len(sample_line1_diag)-sample_size, _step):
#                 sample_area = img[sample_line1_diag[i][1]-sample_size:sample_line1_diag[i][1]+sample_size, sample_line1_diag[i][0]-sample_size: sample_line1_diag[i][0]+sample_size]
#                 averages.append(sample_area.mean())
#             for i in range (sample_size, len(sample_line2_diag)-sample_size, _step):
#                 sample_area = img[sample_line2_diag[i][1]-sample_size:sample_line2_diag[i][1]+sample_size, sample_line2_diag[i][0]-sample_size: sample_line2_diag[i][0]+sample_size]
#                 averages2.append(sample_area.mean())

#             shape_ = Shapes.SQUARE
    
# # if len(approx) in [3, 4, 5, 6]:
# #     if contour_pxl_cnt > (min_bbox_pxl_cnt * 0.40):
# #         if contour_pxl_cnt < (min_bbox_pxl_cnt * 0.60):
# #             shape_ = Shapes.TRIANGLE

#             output = ShapeItem(
#                 id=index,
#                 approx_contour=approx,
#                 default_contour=None,
#                 filtered_contour=None,
#                 boundingbox=None,
#                 boundingbox_min=min_bbox,
#                 sample_positions=None,
#                 closest_corners=None,
#                 sum_int_angles=None,
#                 size=contour_pxl_cnt,
#                 min_bbx_size = cv2.contourArea(min_bbox),
#                 shape=shape_,
#                 centre_x_y=[cX, cY],
#                 _2d_samples=[averages, averages2],
#                 notes_for_debug_file=None)
    
#     return output


def resize_array(arr, new_length):
    old_indices = np.arange(len(arr))
    new_indices = np.linspace(0, len(arr) - 1, new_length)
    return np.interp(new_indices, old_indices, arr).astype(np.uint8)

def get_approx_shape_and_bbox2(
        img,
        img_blurred,
        dataobject: WorkingData,
        index: int,
        bulk_process: ShapeInfo_BulkProcess,
        min_distance: int = 5,
        last_centres: Optional[list[int, int]] = []) -> ShapeItem:


    # get cx and cy early so can drop out
    contour = bulk_process.contour[index]
    moments = cv2.moments(contour)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])



    contour_pxl_cnt = bulk_process.contour_pxl_cnt[index]
    min_bbox_pxl_cnt = bulk_process.min_bbox_pxl_cnt[index]
    
    min_bbox = bulk_process.min_bbox[index]
    approx = bulk_process.approx_contour[index]

    for previous_pt in last_centres:
        if isinstance(previous_pt, list):
            dist = np.linalg.norm(np.array(previous_pt) - np.array([cX, cY]))
            if dist < min_distance:
                if dataobject.debug_details.SAVE_IMAGES_DEBUG is True:
                    return ShapeItem(
                        id=index,
                        approx_contour=approx,
                        default_contour=contour,
                        filtered_contour=None,
                        boundingbox=None,
                        boundingbox_min=min_bbox,
                        sample_positions=None,
                        closest_corners=None,
                        size=contour_pxl_cnt,
                        min_bbx_size=cv2.contourArea(min_bbox),
                        shape=Shapes.TOO_CLOSE,
                        centre_x_y=None,
                        _2d_samples=None,
                        notes_for_debug_file=f"bad_distance")
                else:
                    return None


    if contour_pxl_cnt < (min_bbox_pxl_cnt * 0.50):
        # try and repair it
        
        # if dataobject.debug_details.SAVE_IMAGES_DEBUG is True:
        #     #img_debug = img.copy()
        #     #img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2BGR)
        #     #cv2.drawContours(img_debug, [contour, min_bbox], 0, (0,0,255))
        #     #dataobject.img_view_or_save_if_debug(img_debug, "not_enough_pixels_for_sqr")
        if dataobject.debug_details.SAVE_IMAGES_DEBUG is True:
            return ShapeItem(
                    id=index,
                    approx_contour=None,
                    default_contour=contour,
                    filtered_contour=None,
                    boundingbox=None,
                    boundingbox_min=min_bbox,
                    sample_positions=None,
                    closest_corners=None,
                    size=contour_pxl_cnt,
                    min_bbx_size=cv2.contourArea(min_bbox),
                    shape=Shapes.BAD_PIXELS,
                    centre_x_y=None,
                    _2d_samples=None,
                    notes_for_debug_file="not_enough_pixels_for_sqr")
        return None
        #return None

    
    if len(approx) not in [4, 5, 6, 7, 8]:
        if dataobject.debug_details.SAVE_IMAGES_DEBUG is True:
            return ShapeItem(
                id=index,
                approx_contour=approx,
                default_contour=contour,
                filtered_contour=None,
                boundingbox=None,
                boundingbox_min=min_bbox,
                sample_positions=None,
                closest_corners=None,
                size=contour_pxl_cnt,
                min_bbx_size=cv2.contourArea(min_bbox),
                shape=Shapes.BAD_APPROX_LEN,
                centre_x_y=None,
                _2d_samples=None,
                notes_for_debug_file=f"bad_approxlen")
        else:
            return None
        


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
        if dataobject.debug_details.SAVE_IMAGES_DEBUG is True:
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
                    sample_positions=None,
                    closest_corners=None,
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

    minRect = cv2.minAreaRect(approx)
    min_bbox = cv2.boxPoints(minRect)
    min_bbox = np.intp(min_bbox)
    min_bbox_pxl_cnt = cv2.contourArea(min_bbox)
    contour_pxl_cnt = cv2.contourArea(contour)

    pxl_err = 0.70
    if contour_pxl_cnt <= (min_bbox_pxl_cnt * pxl_err):
        return ShapeItem(
            id=index,
            approx_contour=approx,
            default_contour=contour,
            filtered_contour=None,
            boundingbox=None,
            boundingbox_min=min_bbox,
            sample_positions=None,
            closest_corners=None,
            size=contour_pxl_cnt,
            min_bbx_size=cv2.contourArea(min_bbox),
            shape=Shapes.BAD_APPROX_PXL,
            centre_x_y=None,
            _2d_samples=None,
            notes_for_debug_file=f"BAD_APPROX_PXL")

    if contour_pxl_cnt > (min_bbox_pxl_cnt * pxl_err):
        
        # we know we have a square - lets see if it 
        # has the internal inverse colour circle pattern
        

        # perimeter_10pc = cv2.arcLength(contour, True) * 0.1
        
        
                # arbitrary edge ratio range

                #dataobject.img_view_or_save_if_debug(sqr_sample_area, "SQuare_centre")

        # TODO put here the code to 
        # change the bresenham lines from the bounding box corners
        # to the corners of the approximated shape
        # 
        #  Get corners of 
        nearest_points = [] # corners only going clockwise (probably - doesnt matter so much)
        for pt in min_bbox:
            nearest_points.append(closest_point(pt, approx.reshape(-1, 2)))


        if use_blurred_image(contour_pxl_cnt):
            img2use = img_blurred
        else:
            img2use = img


        spoke_samples_corners, spoke_samples_middle_edges = get_spokecode_samples(
            img2use,
            [cX, cY],
            nearest_points,
            samples_per_line=SAMPLES_PER_LINE
            )
        res = check_barcode.get_ID(spoke_samples_corners, spoke_samples_middle_edges)


        # retry if the barcode is out of focus or motion blurred with constricted sample reach
        if res.res is False:
            if res.retry_reduce_blur is True:
                # Fix the last sample of each quarter by copying from second-to-last
                # this is to improve sampling targetting if the barcode is out of focus or motion blurred
                quarter_length = len(spoke_samples_corners) // 4
                for i in range(4):
                    end_idx = (i + 1) * quarter_length
                    # Copy second-to-last value to last position in each quarter
                    spoke_samples_corners[end_idx - 1] = spoke_samples_corners[end_idx - 2]
                    spoke_samples_middle_edges[end_idx - 1] = spoke_samples_middle_edges[end_idx - 2]
                res = check_barcode.get_ID(spoke_samples_corners, spoke_samples_middle_edges)





        if res.res is True:
            shape_ = Shapes.SQUARE
        else:
            shape_ = Shapes.ALMOST_ID
 

        output = ShapeItem(
            id=index,
            approx_contour=approx,
            default_contour=None,
            filtered_contour=None,
            boundingbox=None,
            boundingbox_min=min_bbox,
            sample_positions=None,#samplepos,
            closest_corners=nearest_points,
            size=contour_pxl_cnt,
            min_bbx_size = cv2.contourArea(min_bbox),
            shape=shape_,
            centre_x_y=[cX, cY],
            _2d_samples=None,
            notes_for_debug_file=None,
            decoded_id=res.decoded_id)

    return output


def debug_save_images(img, contours, text : str, dataobject: WorkingData):
    if dataobject.debug_details.SAVE_IMAGES_DEBUG is True:
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
        dataobject.img_view_or_save_if_debug(img_check_contours, f"{text}_l{len(contours)}")


def get_possible_candidates(img, contours: ContoursType, hierarchy: any, dataobject : WorkingData):
    """get all contours of image, and filter to remove noise
    
    provide thresholded image (might have to inverted to avoid segments
    on edge of image being classed as external), will filter contours for circularity"""
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html

    smallest_area = max((img.shape[0]*0.01) *  (img.shape[1]*0.01), 100)
    largest_area = (img.shape[0]*0.9) *  (img.shape[1]*0.9)

    #d#ataobject.img_view_or_save_if_debug(img, Debug_Images.input_to_contours.value)
    #  get all contours, 

    custom_print(f"get possible candidates: {len(contours)} contours found", dataobject.debug_details.PRINT_DEBUG)
    debug_save_images(img, contours, Debug_Images.unfiltered_contours.value, dataobject)


    with time_it("get possible candidates: filter", dataobject.debug_details.PRINT_DEBUG):
        # filter by area
        contours_area = []
        hierarchy_area = []
        if hierarchy is None or len(contours) == 0 or contours is None:
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
            if circularity > 0.5:
                contours_cirles.append(con)
                hierarchy_cirles.append(hier)
        if len(contours_cirles) != len(hierarchy_cirles):
            raise Exception("bad unzip - use python 3.10 for strict=true")
    debug_save_images(img, contours_cirles, Debug_Images.filtered_circularity_contours.value, dataobject)
    custom_print(f"get possible candidates: {len(contours_cirles)} contours postfilter", dataobject.debug_details.PRINT_DEBUG)

    if dataobject.debug_details.SAVE_IMAGES_DEBUG is True:
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

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def closest_point(target_point, points):
    min_distance = float('inf')
    closest = None
    for point in points:
        distance = euclidean_distance(target_point, point)
        if distance < min_distance:
            min_distance = distance
            closest = point
    return closest


def get_barcode_spokes(closest_corners: list[int,int],centre_x_y: list[int,int] ) -> list[Spokes]:
    """get matching points of line start and line end for the 8 spokes moving around the barcode
    this should start with a top corner then more around 45 degrees
    
    we will also label the points as corner or midline as that will help with the ID"""
    all_points = []
    # start in middle each time
    for index, pt in enumerate(closest_corners):
        next_index = (index+1) % 4 # index wrap around
        all_points.append(Spokes(line_pts=[centre_x_y, pt], barcode_segment=check_barcode.CodeSegment.CORNER))
        midpoint = (closest_corners[index] + closest_corners[next_index]) / 2
        all_points.append(Spokes(line_pts=[centre_x_y, midpoint], barcode_segment=check_barcode.CodeSegment.MIDLINE))

    return all_points

def get_sample_tracks(spokes: list[Spokes], samples_per_line):
    """for the spoke, create samples from centre to outside and attach together"""
    #samples_per_line = (SAMPLES_PER_LINE//2)
    #num_spokes = len(spokes)
    #total_samples = num_spokes * samples_per_line
    #barcode = np.zeros((total_samples, 2), dtype="uint8")
    all_samples = []
    for spoke in spokes:
        #start_idx = i * samples_per_line
        #end_idx = start_idx + samples_per_line
        sampled_points = img_pro.efficient_line_sampler(
            x1=spoke.line_pts[0][0],
            y1=spoke.line_pts[0][1],
            x2=spoke.line_pts[1][0],
            y2=spoke.line_pts[1][1],
            num_samples=samples_per_line
        )
        all_samples.append(SpokesData.from_base(spoke, line_sample_pts=sampled_points))
        #barcode[start_idx:end_idx] = sampled_points

    return all_samples



def get_spokecode_samples(img, midpoint, nearest_points, samples_per_line)->Tuple[list, list]:
    """
    img: np image array, mono
    midpoint: centre of object list[int, int] | ndarray
    nearest_points: 4 sequential coordinates around object centre that defines corners
    samples_per_line: line is defined as start = centrepoint and end is either a corner
    or a midpoint (if corners define a square - the midpoint is on edge)
    &%%#%%@&&&&&%%%%###%#%###&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&%%##%%%%%###%%&&&&&&&&&&&&&&&&&&&&&&&&&&&&%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%, .. #%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%* . ../,. .%%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%#.... ,%&&&&%...*%%&&&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&%%%%&&%/. ..#&&&&&&&&&&(...%%&&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&%%%%%%#......(&&&&&&&&&&&&&&&%....%%&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&%&&&%%%%%. ..../%&&&&&&&&&&&&&&&&&&&&%...(%%&%%%%&&&&&&&&&&&&&&&&&
    &&&&&&&&&%&&%%%%%%%.... .,%&&&&&&&&&&&&&&&&&&&&&&&&&&*.. %%%%%%&%&&&&&&&&&&&&&&&
    &&&&&&&&&%%%%%%......,%&&&&&&&&#&&&&&&&&&&&&&&&&&&&&&&%....%%%%%&&&&&&&&&&&&&&&&
    &&&&&&&&%%%, ... .#%&&&&&&&&&&&&&&&&&&&&&&&&&&&&#.,%&&&%#...(%%%%&&&&&&&&&&&&&&&
    &&&&&&#...   .(%&&&&&&&%%&&&&&&&&&&&&&&&&&&&&&%......%&&&%,.. %%%%%&%%%&&&&&&&&&
    &&&&&&&,,   .%&&%&&&&#.....&&&&&&&&&#&&%&&&&&&&%. ...,&&&&&%...*%%%%%%%%%&&&&&&&
    &&&&&&&&%#    (&&&&&%,.   .&&&&&&&&%......./(&&&&%%%%&&&&&&&&/.. #%%%%%%&&&&&&&&
    &&&&&&&&%&%*.   #&&&&%/...&&&&&&&&%/..,..  .*&&&&&&&&&%%%&&&&&%.  .%%%%%&&%%&&&&
    &&&&&&&&&%%&%.   .%&&&&&&&&&&&&&&&&%,.. ... %&&&&&&&%,....(&&&&&#. .,%%%%%%&&&&&
    &&&%%&&&%%%%%%%.  ./&&&&&&#..  #%&&%%%#/,/%&&&&&&&&&% ..  *&&&&&&%, ..,%%%&&&&&&
    &&&&&%&&&&&&%%&%#.   %&&&%(. ...*%&&&&&&&&&&&&&&&&&&&%*..#&&&&&&&&#... (%%%&&&&&
    %&&&&&&&&&&&&&%%&%,    %%#%%..  #%&&&&&&&&&&&&&&&&&&&&&&&&&&&%#. ... .#%%%%%%%&&
    &&%&&&&&&&&&&&&%%&%# . .*%&&&&&&&&&%%%&&&&&&&&&&&&&&&&&&&%%...    .%%%%%%%%%%%%%
    &&%%&&&&&&&&%%%%%%%%%(  . #%&&&&&#  .*  (&&&&&&&&&&&&&%/.   . .%%%%%%%%%%%%%%%%%
    &&&&&&&&&&&&%%%%%%%%%%%.  ..%%%%%#      .%&&&&&&&&%/ .     #%%%%%%%%%%%%%%%%%%%%
    &&&&&&&%%&&&&&&%%%%%%%%%(    *%%%%%%%#%%%&&&&&%#.   .  *#%%%%%%%%%%%%%%%%%%%%%%%
    &&&&&&&&%%%%%%%&&%%%%%%%%%.  . #%%%%&&&&&%%#, .    ,#%%%%%%%%%%%%%%%%%%%%%%%%&&&
    &&&&&&&&%%%%%%%%%%%%%%%%%%%#  . .%%%&%%%*      .#%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&&
    &&&&&&&&&&%%%%%%%%%%%%%%%%%%%/    *%(...   .#%%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&&&&&
    &&&&&&&&&&%%%%%%%%%%%%%%%%%%%%%.        (%%%%%&%%&&&&&%%%%%%%%%%%%%&&&&&&&&&&&&&
    %&&&&&&&&&&%%%%%%%%%%%%%%%%%%%%%(   ,%%%%%&%&%%%%%%%&&&&&&%%%%%%%&&&&&&&&&&&&&&&
    %%&&&&&&&&&&%%%%%%%%%%%%%%%%%%%%%%%%%%%%%&&&&%%%%%%%%&&&&&&&&&&&&&&&&&&&&&&&&&&&
    """
    # Get all samples, then break up for corners/midpoints at end
    # more performant to do in a couple vectorised operations
    spoke_details = get_barcode_spokes(nearest_points, midpoint)
    sample_lines = get_sample_tracks(
        spoke_details,
        samples_per_line=samples_per_line
        )
    cornersegments = [
        i.line_sample_pts
        for i in sample_lines
        ]
    stacked = np.stack(cornersegments).reshape(-1, 2)

    samples = img_pro.fast_sample(image=img,coordinates=stacked)
    # now we have a mix of both classes. as we know we only have two 
    # classes, we know what class the samples start with and we know
    # the pitch, we can seperate them easily from the main array
    reshaped = samples.reshape(-1, samples_per_line)
    corner_class_samples = reshaped[::2].flatten()
    midedge_class_samples = reshaped[1::2].flatten()

    return corner_class_samples, midedge_class_samples


def draw_barcode_spokes(img, shape_data: ShapeItem):
    """draw lines emanating from centre of shape"""

    #spokes = img.copy()
    # get the sample coordinates to plot
    spoke_point_pairs = get_barcode_spokes(shape_data.closest_corners, shape_data.centre_x_y)
    barcode_array = get_sample_tracks(spoke_point_pairs, samples_per_line=SAMPLES_PER_LINE)

    # get the entire process to get image samples
    spoke_samples_corners, spoke_samples_middle_edges = get_spokecode_samples(
        img,
        shape_data.centre_x_y,
        shape_data.closest_corners,
        samples_per_line=SAMPLES_PER_LINE
        )
    #spoke_samples_corners = img_pro.normalise_np_array(spoke_samples_corners)
    #spoke_samples_middle_edges = img_pro.normalise_np_array(spoke_samples_middle_edges)

    res = check_barcode.get_ID(spoke_samples_corners, spoke_samples_middle_edges)
    # retry if the barcode is out of focus or motion blurred with constricted sample reach
    # TODO: This is duplicating code, try and make it nicer when doing the redo so we don't have to maintain in two places
    if res.res is False:
        if res.retry_reduce_blur is True:
            # Fix the last sample of each quarter by copying from second-to-last
            # this is to improve sampling targetting if the barcode is out of focus or motion blurred
            quarter_length = len(spoke_samples_corners) // 4
            for i in range(4):
                end_idx = (i + 1) * quarter_length
                # Copy second-to-last value to last position in each quarter
                spoke_samples_corners[end_idx - 1] = spoke_samples_corners[end_idx - 2]
                spoke_samples_middle_edges[end_idx - 1] = spoke_samples_middle_edges[end_idx - 2]
            res = check_barcode.get_ID(spoke_samples_corners, spoke_samples_middle_edges)


    colour_gradient = [(255, i, 255-i) for i in range(0, 255, int(255/8))]

    debug_img = img.copy()
    debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
    # print out the sample coordinates so we can visualise them,
    # spitting corner points and mid-edge points
    colarray_corners = []
    colarray_midline = []
    for index, spoke in enumerate(barcode_array):
        col = list(colour_gradient[index])

        if spoke.barcode_segment == check_barcode.CodeSegment.CORNER:
            col[0] = 255
            col[1] = 0
            colarray_corners.append(col)
        else:
            col[0] = 0
            col[1] = 255
            colarray_midline.append(col)
        for pt in spoke.line_sample_pts:
            debug_img[int(pt[1]), int(pt[0])] = col

    # visualise the barcodes
    # get the whitebars for real
    white_bars, _ = check_barcode.decode_white_bars(spoke_samples_corners)
    visualise_corners = np.zeros_like(spoke_samples_corners)
    for bar_pos in white_bars.white_bar_positions:
        visualise_corners[bar_pos[0]: bar_pos[1]] = 1

    spoke_samples_corners_normed = img_pro.normalise_np_array(spoke_samples_corners)
    spoke_samples_corners_normed = img_pro.binarize_barcode(spoke_samples_corners_normed)

    corner_samples_colours= check_barcode.visualise_color_barcode(
        colarray_corners, height=img.shape[0],
        segmentise=4
        )
    middle_samples_colours= check_barcode.visualise_color_barcode(
        colarray_midline, height=img.shape[0],
        segmentise=4
        )
    corner_samples_decoded = check_barcode.visualise_1d_barcode(
        (visualise_corners * 255).astype("uint8"), height=img.shape[0],
        segmentise=4
        )
    
    corner_samples = check_barcode.visualise_1d_barcode(
        (spoke_samples_corners * 1).astype("uint8"), height=img.shape[0],
        segmentise=4
        )
    corner_samples_norm = check_barcode.visualise_1d_barcode(
        (spoke_samples_corners_normed * 255).astype("uint8"), height=img.shape[0],
        segmentise=4
        )

    viewing_buffer = check_barcode.visualise_1d_barcode(
        (spoke_samples_middle_edges * 0).astype("uint8"), height=img.shape[0]
        )
    
    
    if res.res is True:
        viewing_buffer[:] = (0,255,0)
    else:
        viewing_buffer[:] = (0,0,255)
    debug_img = img_pro.add_text_to_image(debug_img,f" id{res.decoded_id} {res.sqr_err}:{res.status}")
    spoke_samples_middle_edges_norm = img_pro.normalise_np_array(spoke_samples_middle_edges)
    spoke_samples_middle_edges_norm = img_pro.binarize_barcode(spoke_samples_middle_edges_norm)
    midedge_samples = check_barcode.visualise_1d_barcode(
        (spoke_samples_middle_edges * 1).astype("uint8"), height=img.shape[0],
        segmentise=4
        )
    midedge_samples_norm = check_barcode.visualise_1d_barcode(
        (spoke_samples_middle_edges_norm * 255).astype("uint8"), height=img.shape[0],
        segmentise=4
        )

    debug_img = np.hstack(
        [   
            debug_img,
            corner_samples_colours,
            corner_samples_decoded,
            corner_samples,
            corner_samples_norm,
            viewing_buffer,
            
            midedge_samples,
            middle_samples_colours,
            midedge_samples_norm
            ])
    return debug_img


def use_blurred_image(_size: int)->bool:
    """large captures should use blur (otherwise could sample noise)
    smaller sizes use unblurred (as could loose all detail)"""
    if _size > 150000:
        return True
    return False


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

    with time_it("AC: check barcode bulk", dataobject.debug_details.PRINT_DEBUG):
        bulk_process = get_approx_shape_and_bbox_bulk(
                    contours,
                    dataobject)

    with time_it("AC: check barcode final", dataobject.debug_details.PRINT_DEBUG):
        for index, c in enumerate(contours):
            contour_stats.append(get_approx_shape_and_bbox2(
                original_img,
                original_blurred_image,
                dataobject,
                index,
                bulk_process,
                last_centres=[i.centre_x_y for i in contour_stats if i and i.shape== Shapes.SQUARE]))

    # if "cam1inner" in dataobject.debug_details.SAVE_IMAGES_PATH:
    #     plop=1
    tote_samples = []
    squrs_found = [cont for cont in contour_stats if cont is not None and cont.shape == Shapes.SQUARE]

    if dataobject.debug_details.SAVE_IMAGES_DEBUG is True or dataobject.debug_details.PRINT_DEBUG is True:
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
        
        custom_print(f"Sample points: {len(samples_all)/2}", dataobject.debug_details.PRINT_DEBUG)
        #tote_samples [x in i._2d_samples for i in squrs_found]


        custom_print(f"total samples: {len(tote_samples)}", dataobject.debug_details.PRINT_DEBUG)


        #img_bbxoes = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
        #img_bbxoes_2 = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
        #img_bbxoes_3 = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
        # for c in contour_stats:
        #     if c is None: continue
        #     cv2.drawContours(img_bbxoes, [c.boundingbox_min], 0, (0,0,255))
        #     #if c.sample_positions is not None:
        #     #    cv2.ellipse(img_bbxoes_2, c.sample_positions,(0,255,0))
        #     cv2.drawContours(img_bbxoes_3, [c.approx_contour], 0, (255,0,255))
        #dataobject.img_view_or_save_if_debug(img_bbxoes, "bounding_boxes")
        #dataobject.img_view_or_save_if_debug(img_bbxoes_2, "fit_ellipse")
        #dataobject.img_view_or_save_if_debug(img_bbxoes_3, "approx_shape")

        # break out triangles and squares
        
        BAD_PIXELS = [cont for cont in contour_stats if cont is not None and cont.shape == Shapes.BAD_PIXELS]
        BAD_RATIO = [cont for cont in contour_stats if cont is not None and cont.shape == Shapes.BAD_RATIO]
        BAD_APPROX_LEN = [cont for cont in contour_stats if cont is not None and cont.shape == Shapes.BAD_APPROX_LEN]
        BAD_APPROX_PXL =  [cont for cont in contour_stats if cont is not None and cont.shape == Shapes.BAD_APPROX_PXL]
        TOO_CLOSE =  [cont for cont in contour_stats if cont is not None and cont.shape == Shapes.TOO_CLOSE]
        ALMOST_ID = [cont for cont in contour_stats if cont is not None and cont.shape == Shapes.ALMOST_ID]
        
        # BAD_PIXELS = []
        # BAD_RATIO = []
        # BAD_APPROX_LEN = []
        # BAD_APPROX_PXL =  []
        # TOO_CLOSE =  []
        # ALMOST_ID = []
        
        debug_img = original_img.copy()
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)


        if squrs_found:
            for c in squrs_found:
                cv2.drawContours(debug_img, [c.approx_contour], -1, (0, 255, 0), 2)
            dataobject.img_view_or_save_if_debug(
                debug_img,
                f"squares_found")
            for c in squrs_found:
                debugname = random.randint(111111, 999999) # sorry, lazy tagging to check 
                if use_blurred_image(c.size):
                    img2use = original_blurred_image
                else:
                    img2use = original_img
                #cv2.drawContours(debug_img, [c.approx_contour], -1, (255,0,0), 2)
                debug_imgx = img2use.copy()
                debug_imgx = draw_barcode_spokes(debug_imgx, c)
                dataobject.img_view_or_save_if_debug(
                    debug_imgx,
                    f"VALID_SPOKIES{debugname}")


            #     dataobject.img_view_or_save_if_debug(
            #         draw_barcode_spokes(c.notes_for_debug_file[0], c),
            #         f"VALID_SPOKIESK{debugname}_CHECK_OG")   

            #     dataobject.img_view_or_save_if_debug(c.notes_for_debug_file[1],
            #         f"VALID_SPOKIES{debugname}_CHECK_REGEN")
            # # trying to find out why we are getting some failed images in our results
            # # at least point every analysis should be green for valid barcode
            # # but we are getting red analysis which doesnt make any sense
            # spoke_samples_corners, _ = get_spokecode_samples(
            #     img2use.copy(),
            #     c.centre_x_y,
            #     c.closest_corners,
            #     samples_per_line=SAMPLES_PER_LINE
            #     )
            # res = check_barcode.get_ID(spoke_samples_corners)
            # if not res:
            #     plop=1





        debug_img = original_img.copy()
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)

        if ALMOST_ID:
            for c in ALMOST_ID:
                if use_blurred_image(c.size):
                    img2use = original_blurred_image
                else:
                    img2use = original_img
                cv2.drawContours(debug_img, [c.approx_contour], -1, (255,0,0), 2)
                debug_imgx = img2use.copy()
                debug_imgx = draw_barcode_spokes(debug_imgx, c)
                dataobject.img_view_or_save_if_debug(
                    debug_imgx,
                    f"ALMOST_ID_SPOKES")
            # dataobject.img_view_or_save_if_debug(
            #     debug_img,
            #     f"ALMOST_ID")
            
        
        debug_img = original_img.copy()
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)

        if TOO_CLOSE:
            for c in TOO_CLOSE:
                cv2.drawContours(debug_img, [c.default_contour], -1, (255,0,0), 2)
                cv2.drawContours(debug_img, [c.boundingbox_min.reshape(4,1,2).astype(int)], -1, (255,0,0), 1)

            dataobject.img_view_or_save_if_debug(
                debug_img,
                f"TOO_CLOSE")
            
        debug_img = original_img.copy()
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        if BAD_PIXELS:
            for c in BAD_PIXELS:
                cv2.drawContours(debug_img, [c.default_contour], -1, (0,0,255), 2)
                cv2.drawContours(debug_img, [c.boundingbox_min.reshape(4,1,2).astype(int)], -1, (255,0,0), 1)
            dataobject.img_view_or_save_if_debug(
                debug_img,
                f"BAD_PIXELS")

        debug_img = original_img.copy()
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)

        if BAD_RATIO:
            for c in BAD_RATIO:
                cv2.drawContours(debug_img, [c.default_contour], -1, (255,0,0), 2)
                cv2.drawContours(debug_img, [c.boundingbox_min.reshape(4,1,2).astype(int)], -1, (255,0,0), 1)
            dataobject.img_view_or_save_if_debug(
                debug_img,
                f"BAD_RATIO")

        debug_img = original_img.copy()
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)

        if BAD_APPROX_LEN:
            for c in BAD_APPROX_LEN:
                cv2.drawContours(debug_img, [c.approx_contour], -1, (0,255,0), 2)
            dataobject.img_view_or_save_if_debug(
                debug_img,
                f"BAD_APPROX_LEN")
            
        debug_img = original_img.copy()
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        if BAD_APPROX_PXL:
            for c in BAD_APPROX_PXL:
                cv2.drawContours(debug_img, [c.approx_contour], -1, (0,255,0), 2)
            dataobject.img_view_or_save_if_debug(
                debug_img,
                f"BAD_APPROX_PXL")
            

        # break out individual squares found:


        for c in squrs_found:
            #try:
            if use_blurred_image(c.size):
                img2use = original_blurred_image
            else:
                img2use = original_img
            debug_img = img2use.copy()
            #debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
            debug_img = draw_barcode_spokes(debug_img, c)
            dataobject.img_view_or_save_if_debug(debug_img, "test_spokes")
        # for c in squrs_found:
        #     #try:
        #     if use_blurred_image(c.size):
        #         img2use = original_blurred_image
        #     else:
        #         img2use = original_img
        #     debug_img = img2use.copy()
        #     debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        #     w = int(np.linalg.norm(c.boundingbox_min[0]-c.boundingbox_min[1]))
        #     h = int(np.linalg.norm(c.boundingbox_min[1]-c.boundingbox_min[2]))
        #     x = c.centre_x_y[0]
        #     y = c.centre_x_y[1]
        #     img_pro.draw_pattern_output(
        #         debug_img,
        #         c,
        #         debug=dataobject.debug_details.SAVE_IMAGES_DEBUG)
            
        #     # closest corners
        #     cv2.circle(debug_img, tuple(c.closest_corners[0]), 3, img_pro.BLUE, 1)
        #     cv2.circle(debug_img, tuple(c.closest_corners[2]), 3, img_pro.BLUE, 1)
        #     cv2.circle(debug_img, tuple(c.closest_corners[1]), 3, img_pro.BLUE, 1)
        #     cv2.circle(debug_img, tuple(c.closest_corners[3]), 3, img_pro.BLUE, 1)
        
        #     cv2.drawContours(debug_img, [c.approx_contour], -1, (0, 255, 0), 1)
        #     crop_img = debug_img[max(0,y-h):y+h, max(0,x-w):x+w]
        #     # if len([True for i in crop_img.shape if i == 0]) > 0:
        #     #     plop=1
        #     #     pass
        #     dataobject.img_view_or_save_if_debug(crop_img, "SquareFound")
            
        #     height = 500
        #     #raise Exception("Please update with new barcode analyser")
        #     ratio1 = height/len(c._2d_samples[0])
        #     ratio2 = height/len(c._2d_samples[1])
        #     peaks1, _ = get_peaks(c._2d_samples[0])
        #     peaks2, _ = get_peaks(c._2d_samples[1])

        #     out_img1 = cv2.resize(np.asarray(c._2d_samples[0]), (200, height), interpolation=cv2.INTER_NEAREST)

        #     out_img1 = cv2.cvtColor(out_img1, cv2.COLOR_GRAY2BGR)

        #     for peak in peaks1:
        #         cv2.circle(out_img1, (100, int(peak*ratio1)), 5, (0,0,255), -1)

        #         #out_img1[int(peak*ratio), 100] = (0,0,255)
        #     #dataobject.img_view_or_save_if_debug(out_img1, "squarecode")
        #     out_img2 = cv2.resize(np.asarray(c._2d_samples[1]), (200, height), interpolation=cv2.INTER_NEAREST)
        #     out_img2 = cv2.cvtColor(out_img2, cv2.COLOR_GRAY2BGR)
        #     for peak in peaks2:
        #         cv2.circle(out_img2, (100, int(peak*ratio2)), 5, (0,0,255), -1)
        #     #dataobject.img_view_or_save_if_debug(out_img2, "squarecode")

        #     stacked_img = np.hstack((
        #         out_img1,
        #         np.zeros(out_img1.shape, np.uint8),
        #         out_img2))
        
        #     dataobject.img_view_or_save_if_debug(stacked_img, "stacked_img")

    return squrs_found


def decode_barcode(data, threshold=0.5):
    """
    Decodes a barcode from a 1D uint8 array by detecting transitions.
    
    Parameters:
    - data: np.ndarray, 1D array of uint8 values representing the scanned barcode.
    - threshold: Float between 0 and 1; intensity value used to binarize the normalized data.
    
    Returns:
    - transitions: List of indices where transitions occur.
    - widths: List of widths of bars and spaces.
    - binary_data: Binarized version of the input data after normalization.
    """
    # Step 1: Normalize the data to range [0, 1]
    data_min = data.min()
    data_max = data.max()
    if data_max > data_min:
        normalized_data = (data - data_min) / (data_max - data_min)
    else:
        normalized_data = np.zeros_like(data, dtype=float)
    
    # Step 2: Thresholding using a fixed threshold (e.g., 0.5)
    binary_data = (normalized_data > threshold).astype(int)
    
    # Step 3: Finding Transitions
    diff_data = np.diff(binary_data)
    transition_indices = np.flatnonzero(diff_data) + 1  # Add 1 due to diff shift
    transitions = transition_indices.tolist()
    
    # Include start and end positions
    positions = [0] + transitions + [len(data)]
    
    # Step 4: Extracting Bar Widths
    widths = np.diff(positions)
    
    return transitions, widths.tolist(), binary_data

def check_for_patternv2(samples) -> Tuple[bool, List[check_barcode.FilteredWhiteBars]]:
    whitebars = []
    for sample in samples:
        whitebars.append(check_barcode.filter_white_bars(
            check_barcode.decode_white_bars(np.array(sample))[0],
            length_array=len(sample)
            ))
    return check_barcode.check_pattern_valid(whitebars, len(sample)), whitebars


# def check_for_pattern(samples):
#     peaks = []
#     peaks_dic={}
#     normed_samples = []
#     min_val = None
#     #raise Exception("fix me")
#     sample_with_peaks = None
#     sample_sans_peaks = None
#     symmetric_err = float("inf")
#     for index, sample in enumerate(samples):
#         peaks_, normed_sample = get_peaks(sample)
#         normed_samples.append(normed_sample)
#         peaks.append(peaks_)
#         if len(peaks[-1]) > 0:
#             symmetric_err = abs(functools.reduce(lambda a, b: a + b, [(len(sample)/2)-x for x in peaks[-1]]))
#             peaks_dic["withpeaks_normed"] = normed_sample
#             peaks_dic["withpeaks"] = sample
#             min_val = min(normed_sample)
#             sample_with_peaks = index
#         else:
#             #peaks_dic["sanspeaks"] = normed_sample
#             peaks_dic["sanspeaks"] = sample
#             sample_sans_peaks = index
#             # we expect black here - so have to normalise this sample with the
#             # limits of the peaked sample
#             #normalized_data_should_be_black = (sample - np.min(normed_samples[0])) / (np.max(normed_samples[0]) - np.min(normed_samples[0]))
#     # for now check that we have one sample line with no peaks and one with 2
#     # later we can make sure peaks are in the positions we expect

#     # we expect one set of samples with 2 peaks and one set with none
#     if set([len(x) for x in peaks]) != set([2, 0]):
#         return False
#     # we expect the peaks to be equally distributed across midline
#     if symmetric_err > MAX_PATTERN_SYMMETRY_ERROR:
#         return False
    
#     # # this should be impossible but for some reason its happening
#     # if not "withpeaks" in peaks_dic:
#     #     return False
#     # we expected the middle of the peaks to be black or close to
#     # this is normalised values
#     if peaks_dic["withpeaks_normed"][int(len(peaks_dic["withpeaks_normed"])/2)] > 0.3:
#         return False
#     # we expected the seamples with peaks to be 
#     #Y_true = peaks_dic["sanspeaks_normed"][2:-2]
#     #Y_predicated = [min_val] * len(samples[sample_sans_peaks])
#     #MSE = np.square(np.subtract(Y_true, Y_predicated)).mean()
#     # if MSE > 505555:
#     #     return False

#     # we expect the sample without peaks to have a low grayscale
#     # similar to the middle of the one with peaks
#     dark_of_peak = peaks_dic["withpeaks"][int(len(peaks_dic["withpeaks"])/2)]
#     # this is just to get an idea of its performance
#     # bear in mind if very dark image compressed range this might die
#     dark_of_peak_max = dark_of_peak * 1.1
#     midpt = int(len(peaks_dic["sanspeaks"])/2)
#     if any(peak > dark_of_peak_max for peak in (peaks_dic["sanspeaks"][midpt],)):
#         return False

#     return True


def get_peaks(sample):
    #  std_dev = np.std(sample)
    _range = max(sample)-min(sample)
    if _range < MIN_TAG_VARIANCE: # arbitrary threshold
        return [], []
    normalized_data = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
    #prominence = int(_range / 3)  # arbitrary way to filter out low prominence peaks
    peaks, _ = find_peaks(normalized_data, height=0.4, prominence=0.2, width=2, distance=4)

    return peaks, normalized_data


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
def quantize_box(box, precision=5):
    # Rounds each coordinate to the nearest multiple defined by precision
    return tuple((box / precision).round().astype(int))




def find_lumotag_mser(inputimg, dataobject : WorkingData):
    """analyse input image for specific lumotag pattern"""
    with time_it("pre-processing msers: total", dataobject.debug_details.PRINT_DEBUG):
        with time_it("grayscale",dataobject.debug_details.PRINT_DEBUG):
            if len(inputimg.shape)>2:
                img_grayscale = cv2.cvtColor(inputimg,cv2.COLOR_BGR2GRAY)
            else:
                img_grayscale = inputimg # get the image source changing here sometimes
        dataobject.img_view_or_save_if_debug(img_grayscale, Debug_Images.original_input.value, resize=False)

        with time_it("pre-processing: blur" ,dataobject.debug_details.PRINT_DEBUG):
            img_op = cv2.blur(img_grayscale,(5,5)) # fastest filter
            # img_op = cv2.medianBlur(img_grayscale, 5)
            dataobject.img_view_or_save_if_debug(img_op, "blur_5_5", resize=False)
        with time_it("pre-processing: get mser regions",dataobject.debug_details.PRINT_DEBUG):
            # bounding box in form: x, y, w, h = box
            msers, bboxes = img_pro.get_mser_regions(img_op)

            if dataobject.debug_details.SAVE_IMAGES_DEBUG:
                debug_save_images(img_op, msers, "UnfliteredMsers", dataobject)
            unique_boxes = {}
            indexes_to_keep = []
            filtered_msers = []
            # for idx, box in enumerate(bboxes):
            #     key = quantize_box(box, precision=5)
            #     if key not in unique_boxes:
            #         unique_boxes[key] = idx
            #         indexes_to_keep.append(idx)
            for idx, (box, mser) in enumerate(zip(bboxes, msers)):
                key = quantize_box(box, precision=5)
                if key not in unique_boxes:
                    unique_boxes[key] = idx
                    filtered_msers.append(mser)


            filtered_msers = [cv2.convexHull(mser.reshape(-1, 1, 2)) for mser in filtered_msers]


    with time_it("get_possible_candidates total",dataobject.debug_details.PRINT_DEBUG):
        contours, hierarchy=get_possible_candidates(img_op,filtered_msers, [[None for _ in filtered_msers]], dataobject)


    with time_it("analyse_candidates TOTAL",dataobject.debug_details.PRINT_DEBUG):
        output_contour_data = analyse_candidates_shapematch(
                                                original_img=img_grayscale,
                                                original_blurred_image=img_op,
                                                contours = contours,
                                                contour_hierarchy = hierarchy,
                                                dataobject = dataobject)

    return output_contour_data


def find_lumotag_special_case(inputimg, dataobject : WorkingData):

    """analyse input image for specific lumotag pattern
    
    specialcaseNoBlurs is for the close-range camera, which at the time of writing
    had a small hotspot and the rest of the image black due to the distribution of the IR torch
    here we can experiment with turning off the blurs as we will have less candidates to work from"""
    with time_it("pre-processing: total", dataobject.debug_details.PRINT_DEBUG):
        with time_it("grayscale",dataobject.debug_details.PRINT_DEBUG):
            if len(inputimg.shape)>2:
                img_grayscale = cv2.cvtColor(inputimg,cv2.COLOR_BGR2GRAY)
            else:
                img_grayscale = inputimg
        dataobject.img_view_or_save_if_debug(inputimg, Debug_Images.original_input.value, resize=False)

        # careful here - can have loads of candidates if not using any blur
        # but the image we get at thise range is usually blown out by the IR torch
        with time_it("pre-processing: blur" ,dataobject.debug_details.PRINT_DEBUG):
            img_op = cv2.blur(img_grayscale,(3,3)) # fastest filter
            # img_op = cv2.medianBlur(img_grayscale, 5)
            dataobject.img_view_or_save_if_debug(img_op, "blur_5_5", resize=False)

        with time_it("pre-processing: threshold_img",dataobject.debug_details.PRINT_DEBUG):
            #img_op=img_pro.threshold_img_static(img_op,low=40,high=255)
            img_op=img_pro.threshold_img(img_op,high=255)

            dataobject.img_view_or_save_if_debug(img_op, "thresholdimg")


        # with time_it("pre-processing: blur again",dataobject.debug_details.PRINT_DEBUG):
        #     img_op = cv2.blur(img_op,(3,3)) # fastest filter
        #     # img_op = cv2.medianBlur(img_op, 3)
        #     dataobject.img_view_or_save_if_debug(img_op,"blur_3_3_again", resize=False)

        # org_img_grayscale_blur = img_grayscale
        # with time_it("pre-processing: blur orig for sampler",dataobject.debug_details.PRINT_DEBUG):
        #     org_img_grayscale_blur = cv2.blur(img_grayscale,(5,5)) # fastest filter
        #     # org_img_grayscale_blur = cv2.medianBlur(img_grayscale, 5)
        #     dataobject.img_view_or_save_if_debug(org_img_grayscale_blur, "blur_for_sampling", resize=False)

    with time_it("get_possible_candidates total",dataobject.debug_details.PRINT_DEBUG):
        with time_it("get possible candidates: find contours", dataobject.debug_details.PRINT_DEBUG):
            contours, hierarchy = cv2.findContours(img_op, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours, hierarchy = get_possible_candidates(img_op,contours, hierarchy, dataobject)

    with time_it("analyse_candidates TOTAL",dataobject.debug_details.PRINT_DEBUG):
        output_contour_data = analyse_candidates_shapematch(
                                                original_img=inputimg,
                                                original_blurred_image=img_grayscale,
                                                contours = contours,
                                                contour_hierarchy = hierarchy,
                                                dataobject = dataobject)

    return output_contour_data


def find_lumotag(inputimg, dataobject : WorkingData):
    with time_it("pre-processing: total", dataobject.debug_details.PRINT_DEBUG):
        with time_it("grayscale",dataobject.debug_details.PRINT_DEBUG):
            if len(inputimg.shape)>2:
                img_grayscale = cv2.cvtColor(inputimg,cv2.COLOR_BGR2GRAY)
            else:
                img_grayscale = inputimg
        dataobject.img_view_or_save_if_debug(inputimg, Debug_Images.original_input.value, resize=False)

        with time_it("pre-processing: blur" ,dataobject.debug_details.PRINT_DEBUG):
            img_op = cv2.blur(img_grayscale,(5,5)) # fastest filter
            # img_op = cv2.medianBlur(img_grayscale, 5)
            dataobject.img_view_or_save_if_debug(img_op, "blur_5_5", resize=False)


        with time_it("pre-processing: threshold_img",dataobject.debug_details.PRINT_DEBUG):
            #img_op=img_pro.threshold_img_static(img_op,low=40,high=255)
            img_op=img_pro.threshold_img(img_op,high=255)

            dataobject.img_view_or_save_if_debug(img_op, "thresholdimg")


        with time_it("pre-processing: blur again",dataobject.debug_details.PRINT_DEBUG):
            img_op = cv2.blur(img_op,(3,3)) # fastest filter
            # img_op = cv2.medianBlur(img_op, 3)
            dataobject.img_view_or_save_if_debug(img_op,"blur_3_3_again", resize=False)

        with time_it("pre-processing: blur orig for sampler",dataobject.debug_details.PRINT_DEBUG):
            org_img_grayscale_blur = cv2.blur(img_grayscale,(5,5)) # fastest filter
            # org_img_grayscale_blur = cv2.medianBlur(img_grayscale, 5)
            dataobject.img_view_or_save_if_debug(org_img_grayscale_blur, "blur_for_sampling", resize=False)

    with time_it("get_possible_candidates total",dataobject.debug_details.PRINT_DEBUG):
        with time_it("get possible candidates: find contours", dataobject.debug_details.PRINT_DEBUG):
            contours, hierarchy = cv2.findContours(img_op, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours, hierarchy = get_possible_candidates(img_op,contours, hierarchy, dataobject)

    with time_it("analyse_candidates TOTAL",dataobject.debug_details.PRINT_DEBUG):
        output_contour_data = analyse_candidates_shapematch(
                                                original_img=inputimg,
                                                original_blurred_image=org_img_grayscale_blur,
                                                contours = contours,
                                                contour_hierarchy = hierarchy,
                                                dataobject = dataobject)

    return output_contour_data



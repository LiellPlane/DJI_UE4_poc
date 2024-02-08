
from enum import Enum, auto
from dataclasses import dataclass
import numpy as np
from typing import ClassVar


class AutoStrEnum(str, Enum):
    """
    StrEnum where auto() returns the field name.
    See https://docs.python.org/3.9/library/enum.html#using-automatic-values
    """
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name


class _OS(str, Enum):
    WINDOWS = "windows"
    RASPBERRY = "raspberry"
    LINUX = "digusting linux"
    MAC_OS = "disgusting mac os"


class Shapes(AutoStrEnum):
    SQUARE = auto()
    TRIANGLE = auto()
    CIRCLE = auto()
    UNKNOWN = auto()
    BAD_RATIO = auto()
    BAD_PIXELS = auto()
    BAD_APPROX_LEN = auto()
    BAD_APPROX_PXL = auto()

# @dataclass
# class Analysis_Results:
#     """Package of image analysis results"""
#     contourdata: list = None
#     crop_details: list = None
#     shrink_img_percent: float = None


@dataclass
class ScreenNormalisedPositions:
    top: float
    lower: float
    left: float
    right: float


@dataclass
class UI_playerInfo:
    photo: ScreenNormalisedPositions
    user_tagname: ScreenNormalisedPositions
    user_info: ScreenNormalisedPositions


@dataclass
class ShapeInfo_BulkProcess:
    """If processing shape candidates in a bulk
    process to vectorise operations
    
    static check can fuck off, sorry
    """
    contour: ClassVar[dict] = {}
    dists_0_to_1: np.array = None
    dists_1_to_2: np.array = None
    approx_contour: ClassVar[dict] = {}
    minRect: ClassVar[dict] = {}
    min_bbox: ClassVar[dict] = {}
    convex_hull_contour: ClassVar[dict] = {}
    contour_pxl_cnt: ClassVar[dict] = {}
    min_bbox_pxl_cnt: ClassVar[dict] = {}


@dataclass
class ShapeItem:
    id: str
    approx_contour: np.array
    default_contour: np.array
    filtered_contour: np.array
    boundingbox: np.array
    boundingbox_min: np.array
    sample_positions: np.array
    closest_corners: np.array
    sum_int_angles: float
    size: int
    min_bbx_size: int
    shape: Shapes
    centre_x_y: list[int]
    _2d_samples: list[list]
    notes_for_debug_file: str

    def add_offset_for_graphics(self, offset: list[int, int]):
        """mutating function is used when a cropped part of the image
        has been analysed, and we wish to print the graphics on uncropped"""
        if self.approx_contour is not None:
            self.approx_contour += np.asarray(offset)
        if self.boundingbox_min is not None:
            self.boundingbox_min += np.asarray(offset)
        if self.centre_x_y is not None:
            self.centre_x_y[0] += offset[0]
            self.centre_x_y[1] += offset[1]

    def add_resize_offset(self, offset: int):
        """mutating function is used when a subsampled image has
        been used, and we want to print graphics on original size"""
        if self.approx_contour is not None:
            self.approx_contour *= offset
        if self.boundingbox_min is not None:
            self.boundingbox_min *= offset
        if self.centre_x_y is not None:
            self.centre_x_y[0] *= offset
            self.centre_x_y[1] *= offset

    def transform_points(self, affine_transform):
        """Transform points to fit transformed video feedback
        https://stackoverflow.com/questions/53569897/affine-transformation-in-image-processing

        expects a 2* 3 affine matrix
        affine_transform
        
        [[  0.27389706   0.         251.        ]
        [  0.           0.27472527   0.        ]]
        """

        concat_affine = np.eye(3)
        concat_affine[0:2, :] = affine_transform

        if self.approx_contour is not None:
            extra_element = np.ones((self.approx_contour.shape[0], 1, 1), dtype=int)
            concat_toaffine = (
                np.concatenate(
                (self.approx_contour, extra_element), axis=-1)).transpose().reshape(3, self.approx_contour.shape[0])
            res = np.matmul(concat_affine, concat_toaffine)
            self.approx_contour = res.transpose()[:,0:2].reshape(self.approx_contour.shape[0],1,2).astype(int)

        if self.boundingbox_min is not None:
            extra_element = np.ones((self.boundingbox_min.shape[0], 1), dtype=int)
            concat_toaffine = (
                np.concatenate(
                (self.boundingbox_min, extra_element), axis=-1)).transpose().reshape(3, self.boundingbox_min.shape[0])
            res = np.matmul(concat_affine, concat_toaffine)
            self.boundingbox_min = res.transpose()[:,0:2].reshape(self.boundingbox_min.shape[0],2).astype(np.int64)

        if self.centre_x_y is not None:
            extra_element = np.array(self.centre_x_y + [1])
            np.matmul(concat_affine, extra_element)
            self.centre_x_y = list(np.matmul(concat_affine, extra_element)[0:2].astype(np.int64))

        if self.closest_corners is not None:
            output = np.array(self.closest_corners)
            # add column of 1s
            output = np.hstack((output, np.ones((4,1))))
            output = np.matmul(concat_affine, output.transpose())[0:2]
            self.closest_corners = list(output[0:2].transpose().astype(np.int64))

@dataclass
class ImagingMode():
    camera_model: str
    res_width_height: tuple[int, int]
    doc_description: str
    shared_mem_reversed: bool
    special_notes: str

@dataclass
class SharedMem_ImgTicket:
    index: int
    res: dict
    buf_size: any
    id: int

@dataclass
class CropSlicing:
    left: int
    right: int
    top: int
    lower: int

@dataclass
class AffinePoints:
    top_left_w_h: tuple[int, int]
    top_right_w_h: tuple[int, int]
    lower_right_w_h: tuple[int, int]

    def as_array(self):
        return [
            self.top_left_w_h,
            self.top_right_w_h,
            self.lower_right_w_h]

    def add_offset_w(self, offset):
        self.top_left_w_h = (self.top_left_w_h[0] + offset, self.top_left_w_h[1])
        self.top_right_w_h = (self.top_right_w_h[0] + offset, self.top_right_w_h[1])
        self.lower_right_w_h = (self.lower_right_w_h[0] + offset, self.lower_right_w_h[1])

    def add_offset_h(self, offset):
        self.top_left_w_h = (self.top_left_w_h[0], self.top_left_w_h[1] + offset)
        self.top_right_w_h = (self.top_right_w_h[0], self.top_right_w_h[1] + offset)
        self.lower_right_w_h = (self.lower_right_w_h[0], self.lower_right_w_h[1] + offset)

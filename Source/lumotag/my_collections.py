
from enum import Enum, auto
from dataclasses import dataclass, field
import numpy as np
from typing import ClassVar, Union, Callable
from functools import lru_cache


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


class CHANNEL_COLOUR(Enum):
    BLUE_CHANNEL = 0
    RED_CHANNEL = 2
    GREEN_CHANNEL = 1

@dataclass
class HeightWidth:
    height: int
    width: int

class Shapes(AutoStrEnum):
    ALMOST_ID = auto()
    TOO_CLOSE = auto()
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
class ScreenPixelPositions:
    top: int
    lower: int
    left: int
    right: int
    top_frame: int = field(init=False)
    lower_frame: int = field(init=False)
    left_frame: int = field(init=False)
    right_frame: int = field(init=False)

    def __post_init__(self):
        self.calculate_padding()

    def calculate_padding(self):
        padding = 6
        self.top_frame = max(self.top - padding, 0)
        self.lower_frame = self.lower + padding
        self.left_frame = max(self.left - padding, 0)
        self.right_frame = self.right + padding

    # #@lru_cache(maxsize=None)
    # def get_rotated_points(self, angle_degs):
    #     theta = np.radians(angle_degs)
    #     rotation_matrix = np.array([
    #         [np.cos(theta), -np.sin(theta)],
    #         [np.sin(theta), np.cos(theta)]
    #     ])
    #     points = np.array([
    #         [self.left, self.top],
    #         [self.right, self.top],
    #         [self.right, self.lower],
    #         [self.left, self.lower]
    #     ])
    #     # get mapping of the rotated element from the static image, so we can copy it in at display time
    #     rotated_points = np.dot(points, rotation_matrix.T)
    #     return rotated_points


@dataclass(frozen=True) # need this so we can hash the dataclass for caching 
class ScreenNormalisedPositions:
    top: float
    lower: float
    left: float
    right: float

    @lru_cache
    def get_pixel_positions_with_ratio(self, img_shape, element_shape):

        element_height = element_shape[0]
        element_width = element_shape[1]

        img_height = img_shape[0]
        img_width = img_shape[1]

        pxl_width_pc = self.right - self.left
        pxl_height_pc = self.lower - self.top

        ui_boxsize_width_pxls = pxl_width_pc * img_width
        ui_boxsize_height_pxls = pxl_height_pc * img_height

        # try one and if it fails try the other
        resize_ratios = [
            ui_boxsize_width_pxls / element_width,
            ui_boxsize_height_pxls / element_height
        ]

        for resize_ratio in resize_ratios:
            new_width = element_width * resize_ratio
            new_height = element_height * resize_ratio
            if all([
                new_width <= ui_boxsize_width_pxls,
                new_height <= ui_boxsize_height_pxls
            ]):
                break

        if abs((element_height / new_height) - (element_width / new_width)) > 0.1:
            raise ValueError("bad image ratio!!")
        # we have normalised positions and pre-scaled positions
        # need to offset the pre-scaled positions from the normalised ones
        top = self.top * img_height
        lower = top + new_height
        left = self.left * img_width
        right = left + new_width

        return ScreenPixelPositions(
            top=int(top),
            lower=int(lower),
            left=int(left),
            right=int(right)
        )


@dataclass
class UI_Behaviour_static():
    screen_normed_pos: ScreenNormalisedPositions
    channel: int
    border: bool


@dataclass
class UI_Behaviour_dynamic():
    screen_normed_pos: ScreenNormalisedPositions
    border: bool
    channel_A: int
    channel_B: int
    cut_off_value_norm: Union[float,  int]

    def get_channel(self, normalised_input_value):
        if normalised_input_value <= self.cut_off_value_norm:
            return self.channel_B
        return self.channel_A

@dataclass
class UI_ready_element:
    """UI element with positions to inject into an image"""
    name: str
    position: ScreenPixelPositions
    rotated_position: ScreenPixelPositions
    image: any # np array
    rotated_image: any # np array
    transform: any # affine matrix to transform element to position in output display NB element has to be rotated correctly first
    element_specifics: Union[UI_Behaviour_static, UI_Behaviour_dynamic]

# @dataclass
# class UI_playerInfo:
#     photo: ScreenNormalisedPositions
#     user_tagname: ScreenNormalisedPositions
#     user_info: ScreenNormalisedPositions


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
    size: int
    min_bbx_size: int
    shape: Shapes
    centre_x_y: list[int]
    _2d_samples: list[list]
    notes_for_debug_file: str
    decoded_id: int = -1
    

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
    def get_as_tuple(self):
        return (self.left, self.right, self.top, self.lower)
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

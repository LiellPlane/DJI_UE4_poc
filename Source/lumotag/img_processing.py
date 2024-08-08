import sys
import cv2
import shutil
from enum import Enum, auto
import os
import numpy as np
import time
from contextlib import contextmanager
from typing import Iterator, Literal, Annotated, Optional
from dataclasses import dataclass
from skimage.draw import line
from my_collections import CropSlicing, AffinePoints, UI_ready_element
from math import floor
try:# TODO FIX
    import utils
except:
    print("this is really bad please fix scambilight import issue")

Array3x3 = Annotated[np.ndarray, (3, 3)]
RED = (0, 0, 255)
BLUE = (255, 0, 0)


def interpolate_points(start, end, steps):
    return np.linspace(start, end, steps)

def ease_in_out_quad(t):
    return 2 * t**2 if t < 0.5 else 1 - (-2 * t + 2)**2 / 2

def ease_in_out_quart(t):
    return 8 * t**4 if t < 0.5 else 1 - (-2 * t + 2)**4 / 2

def ease_in_out_sine(t):
    return -(np.cos(np.pi * t) - 1) / 2

def ease_in_out_cubic(t):
    return np.where(t < 0.5, 4 * t**3, 1 - (-2 * t + 2)**3 / 2)

def interpolate_points_eased(start, end, steps):
    t = np.linspace(0, 1, steps)
    eased_t = ease_in_out_cubic(t)
    return start + eased_t[:, np.newaxis] * (end - start)

@dataclass
class CamDisplayTransform:
    cam_image_shape: tuple[int]
    

def darken_image(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def gray2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)



def radial_motion_blur(image, intensity=30):
    center = (image.shape[0]//2, image.shape[1]//2)
    h, w = image.shape
    y, x = np.ogrid[:h, :w]
    
    # Calculate angle and distance from center for each pixel
    angle = np.arctan2(y - center[1], x - center[0])
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Normalize distances
    max_dist = np.sqrt(w**2 + h**2)
    norm_dist = dist_from_center / max_dist
    
    # Create motion blur
    blurred = np.zeros_like(image, dtype=np.float32)
    for i in range(intensity):
        offset = i / intensity
        x_offset = offset * np.cos(angle) * norm_dist * intensity
        y_offset = offset * np.sin(angle) * norm_dist * intensity
        
        x_shift = np.round(x + x_offset).astype(int)
        y_shift = np.round(y + y_offset).astype(int)
        
        # Ensure we don't go out of bounds
        x_shift = np.clip(x_shift, 0, w-1)
        y_shift = np.clip(y_shift, 0, h-1)
        
        blurred += image[y_shift, x_shift] / intensity
    
    return blurred.astype(np.uint8)

@dataclass
class TransformsDetails:
    longrange_to_shortrange_perwarp: Array3x3 # warp calculated to seamlessly embed longrange image into closerange(quilt)
    closerange_to_display: CamDisplayTransform # transform to fit output display (with aspect ratio)
    longrange_to_display: CamDisplayTransform # transform to fit output display (with aspect ratio)
    transition_steps: int # moving between longe range and close range
    transition_time_secs: float # desired transition time to be LERPED
    display_image_shape: tuple[int]
    displayrotation: Literal[0, 90, 180, 270] # rotation of LCD screen on chassis
    slice_details_close_range: Optional[CropSlicing]
    slice_details_long_range: Optional[CropSlicing]


class CameraTransitionState(Enum):
    CLOSERANGE = auto()
    LONGRANGE = auto()
    TRANSITIONING = auto()


class TransformManager:
    def __init__(self, transformdetails: TransformsDetails):
        """all transitions assume that index 0 is closerange engated and last index is longrange engaged"""
        self._transitions_direction = 1
        self._triggered_time = time.perf_counter()
        self._current_managed_index = 0
        self.transformdetails: TransformsDetails = transformdetails
        # CR camera starts as fully engaged, then warps in to engage the LR. 
        # we need a starting point - so we use the points from the LR mapped to SR space (so embedded in the image)
        # then this transitions to fully engaged LR. These transforms can be used to warp the CR
        self.LR_lerp: list = self._get_close_to_long_transition_points()
        self.CR_transition_m: list[Array3x3] = self._get_transition_Matrices(
            target_array=self.LR_lerp,
            source_array=self.LR_lerp[:, 0]
            )
        # this transition is the LR from embedded in the CR to fully engaged.
        # we need the static transform of the embedded LR in CR space, and we also need the 
        # transforms lerping from this embedded to fully engaged. This needs 2 transforms
        self.LR_transition_m: list[Array3x3] = self._longrange_transition_calc_m(
            self.CR_transition_m,
            self.transformdetails.longrange_to_shortrange_perwarp
            )
        # there is a final transform which displays these images for the display. These are much smaller
        # so any blurring operations are best done here
        # we also have different aspect ratios of LR and CR, so the transition will have to include this
        # transform as well, so the display field is recalculated each transition with this new output shape
        self.displaytransition_lerp: list = self._get_display_transition_points()
        self.display_affine_transition_m = self._get_display_affine_transitions()
        self.display_warp_transition_m: list[Array3x3] = self._convert_affine_to_3x3(
            self.display_affine_transition_m
            )
        # # that didnt seem to work - so lets add a lerp between the two camera shapes as well
        # self.shape_transition_m: list[Array3x3] = self._get_transition_Matrices(
        #     target_array=self.displaytransition_lerp,
        #     source_array=self.displaytransition_lerp[:, 0]
        #     )
        self.CR_all_transition_m: list[Array3x3] = self._matmul_lists(
            list1=self.display_warp_transition_m,
            list2=self.CR_transition_m 
            )
        self.LR_all_transition_m: list[Array3x3] = self._matmul_lists(
            list1=self.display_warp_transition_m,
            list2=self.LR_transition_m 
            )
        # this is optional
        if self.transformdetails.slice_details_close_range and self.transformdetails.slice_details_long_range:
            self.lerped_slice_details = self.lerp_slice_details()

    def get_lerped_targetzone_slice(self, index):
        """the slices have to be provided or this will die"""
        return self.lerped_slice_details[index]

    def lerp_slice_details(self):
        cr_slice_np  = np.asarray(self.transformdetails.slice_details_close_range.get_as_tuple())
        lr_slice_np  = np.asarray(self.transformdetails.slice_details_long_range.get_as_tuple())
        lerped =  lerp_arrays(cr_slice_np,lr_slice_np,self.transformdetails.transition_steps)
        output = []
        for i in range(0, lerped.shape[1]):

            output.append(CropSlicing(
                left=lerped[:,i][0],
                right=lerped[:,i][1],
                top=lerped[:,i][2],
                lower=lerped[:,i][3]
                ))
        return output

    def get_display_affine_transformation(self, index):
        return self.display_affine_transition_m[index]
    
    def trigger_transition(self):
        '''alert manager that we want to it to generate transform indexes proportional to time delta'''
        self._transitions_direction *= -1
        self._triggered_time = time.perf_counter()


    def get_transition_state(self) -> CameraTransitionState:
        if self._current_managed_index <= 0:
            return CameraTransitionState.CLOSERANGE
        elif self._current_managed_index >= self.transformdetails.transition_steps-1: # zero based thing
            return CameraTransitionState.LONGRANGE
        else:
            return CameraTransitionState.TRANSITIONING
        

    def get_deltatime_transition(self):
        '''if you have triggered the trigger_transitions, use this to get current index proportional time delta and configured time span'''
        if any([
            (self._current_managed_index >= self.transformdetails.transition_steps) and self._transitions_direction == 1,
            (self._current_managed_index <= 0) and self._transitions_direction == -1
        ]):
            "don't bother calculating everything"
            pass
        else: 
            
            time_delta_sec = time.perf_counter() - self._triggered_time
            self.transformdetails.transition_time_secs
            self.transformdetails.transition_steps
            percent_done = time_delta_sec/self.transformdetails.transition_time_secs
            steps_in_timespan = int(self.transformdetails.transition_steps * percent_done) * self._transitions_direction
            self._current_managed_index += steps_in_timespan
            
            self._current_managed_index = min(max(0, self._current_managed_index), self.transformdetails.transition_steps)
        
        self._triggered_time = time.perf_counter()
        return self._current_managed_index

    @staticmethod
    def _matmul_lists(list1: list[Array3x3], list2: list[Array3x3]) -> list[Array3x3]:
        assert len(list1) == len(list2)
        matrices = []
        for mat1, mat2 in zip(list1, list2):
            matrices.append(np.matmul(mat1, mat2))
        return matrices

    def _convert_affine_to_3x3(self, affinetransforms: list) -> list [Array3x3]:
        matrices = []
        for affine_t in affinetransforms:
            warp_matrix = np.eye(3, dtype=np.float32)
            warp_matrix[:2, :] = affine_t
            matrices.append(warp_matrix)
        return matrices
    
    def _get_display_affine_transitions(self):
        transforms = []
        for i in range(0, self.displaytransition_lerp.shape[1]):
            corners = self.displaytransition_lerp[:, i]
            shape = (
                int(corners[:, 1].max()),
                int(corners[:, 0].max())
            )
            transforms.append(get_fitted_affine_transform(
                    cam_image_shape=shape,
                    display_image_shape=self.transformdetails.display_image_shape,
                    rotation=self.transformdetails.displayrotation
                    )
                )
        return transforms

    @staticmethod
    def _longrange_transition_calc_m(cr_transition_m: list[Array3x3], perpwarp: Array3x3):
        matrices = []
        for mat in cr_transition_m:
            matrices.append(np.matmul(mat, perpwarp))
        return matrices

    def _get_transition_Matrices(
            self,
            target_array: list[np.ndarray],
            source_array: np.ndarray):
        matrices = []
        for i in range(0, target_array.shape[1]):
            matrices.append(self._calc_perp_transform(
                src_points=source_array,
                dst_points=target_array[:, i]
                )
                )
        return matrices

    def _get_close_to_long_transition_points(self):
        long_range_corners = get_imagecorners_as_np_array(self.transformdetails.longrange_to_display.cam_image_shape)
        long_range_corners_in_SR_coords = mtransform_array_of_points(long_range_corners,self.transformdetails.longrange_to_shortrange_perwarp )
        #close_range_corners = get_imagecorners_as_np_array(self.transformdetails.closerange_to_display.cam_image_shape)
        lerped = self._get_lerped_points(long_range_corners_in_SR_coords, long_range_corners)
        return lerped

    def _get_display_transition_points(self):
        long_range_corners = get_imagecorners_as_np_array(self.transformdetails.longrange_to_display.cam_image_shape)
        close_range_corners = get_imagecorners_as_np_array(self.transformdetails.closerange_to_display.cam_image_shape)
        lerped = self._get_lerped_points(close_range_corners, long_range_corners)
        return lerped

    @staticmethod
    def _calc_perp_transform(src_points, dst_points) -> np.ndarray:

        return cv2.getPerspectiveTransform(
            np.array(src_points, dtype=np.float32),
            np.array(dst_points, dtype=np.float32)
            )

    def _get_lerped_points(self, startarray, endarray):
        """lerp between two sets of points, for instance provide 4 corners of one image and 4 corners of another and lerp between them"""
        interpolated_points = [
            interpolate_points_eased(start, end, self.transformdetails.transition_steps)
            for start, end
            in zip(
            startarray,
            endarray
            )
        ]
        return np.array(interpolated_points)

def lerp_arrays(a, b, num_points):
    t = np.linspace(0, 1, num_points)
    return np.array([np.interp(t, [0, 1], [a_i, b_i]) for a_i, b_i in zip(a, b)])

def get_imagecorners_as_np_array(imgshape: tuple[int]):
    return np.asarray([(0, imgshape[0]), (0,0), (imgshape[1], 0), (imgshape[1], imgshape[0])])


def mtransform_array_of_points(myarray:np.ndarray, mytransform: Array3x3) -> np.ndarray :
    homogeneous_points = np.column_stack((myarray, np.ones(len(myarray))))
    transformed_points = np.dot(homogeneous_points, mytransform.T)
    transformed_points_2d = transformed_points[:, :2] / transformed_points[:, 2:]
    return transformed_points_2d


def read_img(img_filepath):
    return cv2.imread(img_filepath)


def compute_and_apply_perpwarp(src_img, dst_img, src_points, dst_points):
    # Convert points to numpy arrays
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation to the source image
    height, width = dst_img.shape[:2]
    result = cv2.warpPerspective(src_img, matrix, (width, height))

    return result, matrix

def overlay_warped_image(background, warped):
    # Ensure the images have the same size and are mono
    assert background.shape == warped.shape, "Images must have the same dimensions"
    assert len(background.shape) == 2 and len(warped.shape) == 2, "Images must be mono (single channel)"
    
    # Ensure images are 8-bit unsigned integer type
    background = background.astype(np.uint8)
    warped = warped.astype(np.uint8)

    # Create a mask based on non-black pixels in the warped image
    _, mask = cv2.threshold(warped, 1, 255, cv2.THRESH_BINARY)

    # Black-out the area of warped image in background
    background_masked = cv2.bitwise_and(background, cv2.bitwise_not(mask))

    # Combine the background and warped image
    result = cv2.add(background_masked, warped)

    return result

def overlay_warped_image_alpha(background, warped, alpha=0.1):
    # Ensure the images have the same size and are mono
    assert background.shape == warped.shape, "Images must have the same dimensions"
    assert len(background.shape) == 2 and len(warped.shape) == 2, "Images must be mono (single channel)"
    
    # Ensure images are 8-bit unsigned integer type
    background = background.astype(np.uint8)
    warped = warped.astype(np.uint8)

    # Create a mask based on non-black pixels in the warped image
    _, mask = cv2.threshold(warped, 1, 255, cv2.THRESH_BINARY)

    # Convert mask to float and normalize
    mask = mask.astype(float) / 255.0

    # Apply alpha to the mask
    mask *= alpha

    # Invert the mask
    inv_mask = 1.0 - mask

    # Blend the images
    result = (background * inv_mask + warped * mask).astype(np.uint8)

    return result


def overlay_warped_image_alpha_feathered(background, warped, alpha=0.1, feather_amount=10):
    # Ensure the images have the same size and are mono
    assert background.shape == warped.shape, "Images must have the same dimensions"
    assert len(background.shape) == 2 and len(warped.shape) == 2, "Images must be mono (single channel)"
    
    # Ensure images are 8-bit unsigned integer type
    background = background.astype(np.uint8)
    warped = warped.astype(np.uint8)

    # Create a mask based on non-black pixels in the warped image
    _, mask = cv2.threshold(warped, 1, 255, cv2.THRESH_BINARY)

    # Apply feathering to the mask
    mask = cv2.GaussianBlur(mask, (feather_amount*2+1, feather_amount*2+1), 0)

    # Convert mask to float and normalize
    mask = mask.astype(float) / 255.0

    # Apply alpha to the mask
    mask *= alpha

    # Invert the mask
    inv_mask = 1.0 - mask

    # Blend the images
    result = (background * inv_mask + warped * mask).astype(np.uint8)

    return result


def apply_perp_transform(matrix, src_img, dst_img):
    
    # Apply the perspective transformation to the source image
    height, width = dst_img.shape[:2]
    result = cv2.warpPerspective(src_img, matrix, (width, height))

    return result


def concat_image(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation = cv2.INTER_NEAREST)
    return cv2.hconcat([img1,img2 ])

def clahe_equalisation(img, claheprocessor):
    if claheprocessor is None:
        claheprocessor = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(50,50))
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
    return cv2.GaussianBlur(img,(filtersize,filtersize),0)

def blur_average(img, filtersize = 7):
    kernel = np.ones((filtersize,filtersize),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    return dst

def normalise(img):
    image2_Norm = cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
    return image2_Norm

def threshold_img(img, low=0, high=255):
    #_ , th3 = cv2.threshold(img, low, 255,cv2.THRESH_BINARY)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,1)
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
    
    return gray_filtered

def simple_canny(blurred_img, lower, upper):
    # wide = cv2.Canny(blurred, 10, 200)
    # mid = cv2.Canny(blurred, 30, 150)
    # tight = cv2.Canny(blurred, 240, 250)
    return cv2.Canny(blurred_img, upper, lower, 7,L2gradient = False)

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
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)

    # return the resized image
    return resized

def get_resized_equalaspect(inputimage, screensize):

    tofit_height = inputimage.shape[0]
    tofit_width = inputimage.shape[1]
    to_fit_height2width = tofit_height / tofit_width
    screensize_height = screensize[0]
    screensize_width = screensize[1]

    # fit to height and check if width is ok
    test_height = screensize_height
    test_width = screensize_height / to_fit_height2width
    dim = None
    if test_width > screensize_width:
        test_width = screensize_width
        test_height = test_width * to_fit_height2width
        if test_height > screensize_height:
            raise Exception(
                "screen resize with aspect ratio has failed in heght & width cases, bad")

    dim = (
        int(np.floor(test_width)),
        int(np.floor(test_height)))

    return cv2.resize(inputimage, dim, interpolation = cv2.INTER_NEAREST)

def resize_centre_img(inputimage, screensize):

    # this is slow - might be faster passing in the image again?
    # TODO
    # empty is faster than zeros
    emptyscreen = np.zeros((screensize + (3,)), np.uint8)

    if screensize[0] < screensize[1]:
        image = image_resize_ratio(
            inputimage,
            height=screensize[0])
    else:
        image = image_resize_ratio(
            inputimage,
            width=screensize[1])


    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 

    offset_x = (emptyscreen.shape[0] - image.shape[0]) // 2
    offset_y = (emptyscreen.shape[1] - image.shape[1]) // 2
    emptyscreen[
        offset_x:image.shape[0]+offset_x,
        offset_y:image.shape[1]+offset_y,
        :] = image

    # should be equal scaling for dims as maintains aspect ratio
    scale_factor = (image.shape[0] / inputimage.shape[0])
    return emptyscreen, scale_factor


def add_cross_hair(image, adapt, lerp = 0):
    thick = 3
    vis_block = 30
    midx = image.shape[0] // 2
    midy = image.shape[1] // 2
    # TODO another potential lag point
    if adapt is True:
        col = int(
            image[midx-50:midx+50:4, midy-50:midy+50:4, :].mean())
    else:
        col = 255

    red = int(255 * lerp)
    green = int((1- lerp) * max(col, 50))

    image[midx - thick : midx + thick, 0:midy-vis_block, 2] = red
    image[midx - thick : midx + thick, 0:midy-vis_block, 1] = green
    image[midx - thick : midx + thick, midy+vis_block:-1, 2] = red
    image[midx - thick : midx + thick, midy+vis_block:-1, 1] = green

    image[0:midx-vis_block, midy - thick : midy + thick, 2] = red
    image[0:midx-vis_block, midy - thick : midy + thick, 1] = green
    image[midx+vis_block:-1, midy - thick : midy + thick, 2] = red
    image[midx+vis_block:-1, midy - thick : midy + thick, 1] = green
 

class lerped_add_crosshair():
    def __init__(self) -> None:
        self.lerper = utils.Lerp(
            start_value=0,
            end_value=1,
            duration=10,
            easing="ease_in_out_cubic"
            )

    def add_cross_hair(self, image, adapt, target_acquired=False):
        """wrap the add cross hair function so we can lerp it easily
        lerp in when target is acquired and lerp back out when lost"""
        pass
        # if target_acquired:
        #     if not self.lerper.is_running:
        #         self.lerper.start()
        # self.lerper.set_reverse_state(not target_acquired)
        # add_cross_hair(image, adapt,self.lerper.get_value())



def get_internal_section(imgshape, size: tuple[int, int]):
    midx = imgshape[0] // 2
    midy = imgshape[1] // 2
    regionx = size[0] // 2
    regiony = size[1] // 2
    left = max(midx-regionx, 0)
    right = min(midx+regionx, imgshape[0])
    top = max(midy-regiony, 0)
    lower = min(midy+regiony, imgshape[1])
    return CropSlicing(left=left, right=right, top=top, lower=lower)

def implant_internal_section(img, img_to_implant):

    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if len(img_to_implant.shape) < 3:
        img_to_implant = cv2.cvtColor(img_to_implant, cv2.COLOR_GRAY2RGB)
    #draw white border around area to implant
    img_to_implant[2:img_to_implant.shape[0]-2, 2,:] = 255
    img_to_implant[2:img_to_implant.shape[0]-2, img_to_implant.shape[1]-1,:] = 255
    img_to_implant[2, 2:img_to_implant.shape[1]-2,:] = 255
    img_to_implant[img_to_implant.shape[0]-2, 2:img_to_implant.shape[1]-1,:] = 255
    midx = img.shape[0] // 2
    midy = img.shape[1] // 2
    regionx = img_to_implant.shape[0] // 2
    regiony = img_to_implant.shape[1] // 2
    # specifying the area to implant is incase of odd sized half, so might miss 
    # a pixel leading to broadcast error
    img[midx-regionx:midx+regionx,
       midy-regiony:midy+regiony, :] = img_to_implant[0:regionx*2, 0:regiony*2]
    return img

# def bresenham_line_wikipedia(x0, y0, x1, y1):
# https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm


def bresenham_line_ski(x1,y1,x2, y2):
    rr, cc = line(x1,y1,x2, y2)
    return [i for i in zip(rr, cc)]


def get_affine_transform(pts1, pts2):
    """from 2 sets of 3 corresponding points
    calculate the affine transform"""
    return cv2.getAffineTransform(pts1, pts2)


def do_affine(img, T, row_cols: tuple[int, int]):
    return cv2.warpAffine(img, T, row_cols)


def rotate_pt_around_origin(point, origin, degrees):
    radians = np.deg2rad(degrees)
    x,y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + (cos_rad * adjusted_x) + (sin_rad * adjusted_y)
    qy = offset_y + (-sin_rad * adjusted_x) + (cos_rad * adjusted_y)
    return qx, qy


def add_ui_elements(
        image,
        element_package: UI_ready_element,
        fade_norm: float
        ) -> None:
    image[
            element_package.position.top:element_package.position.lower,
            element_package.position.left:element_package.position.right,
            0
        ] = (element_package.image * fade_norm).astype(np.uint8)


def resize_image(inputimage, width, height):
    return cv2.resize(inputimage, (width, height), interpolation = cv2.INTER_NEAREST)


def draw_pattern_output(image, patterndetails, debug=False): # ShapeItem - TODO 
    """draw graphics for user if a pattern is found
    TODO: maybe want floating numbers etc above this which
    will eventually need a user registry"""
    min_bbox = patterndetails.boundingbox_min
    cX, cY = patterndetails.centre_x_y
    closest_corners = patterndetails.closest_corners
    # corners of square
    cv2.circle(image, tuple(min_bbox[0]), 3, RED, 1)
    cv2.circle(image, tuple(min_bbox[2]), 3, RED, 1)
    cv2.circle(image, tuple(min_bbox[1]), 3, RED, 1)
    cv2.circle(image, tuple(min_bbox[3]), 3, RED, 1)


    # centre of pattern
    cv2.circle(image, (cX, cY), 5, RED, 1)
   
    # bounding box of contour - this does not handle perspective
    cv2.drawContours(image, [min_bbox], 0, RED)

    #draw barcode sampling lines - for illustration only
    # may not match exactly with generated sampled lines
    if debug is False:
        cv2.line(image, tuple(closest_corners[0]), tuple(closest_corners[2]), BLUE, 1) 
        cv2.line(image, tuple(closest_corners[1]), tuple(closest_corners[3]), BLUE, 1)
    else:
        for pos in patterndetails.sample_positions:

            cv2.circle(image, (pos[0],pos[1]), radius=0, color=RED, thickness=-1)


def load_img_set_transparency():
    #  Debug code until we have a user avatar delivery system
    imgfoler = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    imgfoler = imgfoler[:imgfoler.index("Source")]
    imgsource = f"{imgfoler}Source/lumotag/avatars/chick.png"
    return cv2.imread(imgsource, cv2.IMREAD_UNCHANGED)


def get_fitted_affine_transform(
    cam_image_shape,
    display_image_shape,
    rotation: Literal[0, -90, 90, 270, 180]
    ):
    """Get the matrix transform to rotate and scale a camera image to fit
    in a display image space, maintaining aspect ratio.
    
    cam image:
    
    _______    ^
    |^ ^ ^|    |
    |^ ^ ^|    | up vector
    |^_^_^|    |
    
    display LCD image:
    
    |-------------------|
    |                   |  <------- UP vector (user POV)
    |-------------------|

    and we want to rotate the cam image 90 degrees (so its correct for viewer)
    then scale and centralise to fit in display image

    |------|<-<-<-|------|
    |      |< < < |      |  <------- UP vector (user POV)
    |------|<-<-<-|------|

    Note the implanted camera image has been rotated to fit.

    this function only works for 90 degree angles
    """
    if rotation in [90, -90, 270]:
        reverse_output_shape = tuple(reversed(display_image_shape[0:2]))
        # if planning for 90 degrees, swap image dims
        input_targets, output_targets = get_affine_points(
            cam_image_shape,
            reverse_output_shape)
        output_targets = rotate_affine_targets(
            output_targets,
            rotation,
            reverse_output_shape)

        diffs = (np.asarray(reverse_output_shape) - np.asarray(display_image_shape[0:2]))/2
        output_targets.add_offset_h(diffs[1])
        output_targets.add_offset_w(diffs[0])

    elif rotation == 180:
        input_targets, output_targets = get_affine_points(
            cam_image_shape,
            display_image_shape)
        # have to flip output targets
        output_targets = rotate_affine_targets(
            output_targets,
            rotation,
            display_image_shape)

    elif rotation == 0:
        input_targets, output_targets = get_affine_points(
            cam_image_shape,
            (display_image_shape))

    return get_affine_transform(
        pts1=np.asarray(input_targets.as_array(), dtype="float32"),
        pts2=np.asarray(output_targets.as_array(), dtype="float32"))


def get_affine_points(incoming_img_dims, outgoing_img_dims) -> AffinePoints:
    """Return the corresponding points to fit the incoming image central to the
    view screen maintaining the aspect ratio, to be used to calculate affine
    transform
    
    inputs:
    incoming_img_dims: numpy array .shape
    outcoming_img_dims: numpy array .shape

    return source points, target points
    """
    incoming_w = incoming_img_dims[1]
    incoming_h = incoming_img_dims[0]
    outgoing_w = outgoing_img_dims[1]
    outgoing_h = outgoing_img_dims[0]
    incoming_pts = AffinePoints(
        top_left_w_h=(0, 0),
        top_right_w_h=(incoming_w, 0),
        lower_right_w_h=(incoming_w, incoming_h))
    # pick any ratio
    ratio = outgoing_h / incoming_h
    # if resizing with aspect ratio doesn't fit, do the other way
    if floor(incoming_w * ratio) > outgoing_w:
        ratio = outgoing_w / incoming_w
    output_fit_h = floor(incoming_h * ratio)
    output_fit_w = floor(incoming_w * ratio)
    # test to make sure aspect ratio is 
    if abs((incoming_h/incoming_w) - (outgoing_h/outgoing_w)) > 2:
        raise ValueError("error calculating output image dimensions")
    
    if abs(output_fit_w-outgoing_w)>1 and abs(output_fit_h-outgoing_h)>1:
        raise ValueError("error calculating output image dimensions")
    # get 3 corresponding points from the output view - keeping in mind
    # any rotation
    w_crop_in = (outgoing_w - output_fit_w) // 2
    h_crop_in = (outgoing_h - output_fit_h) // 2
    view_pts = AffinePoints(
        top_left_w_h=(w_crop_in, h_crop_in),
        top_right_w_h=(w_crop_in + output_fit_w, h_crop_in),
        lower_right_w_h=(w_crop_in + output_fit_w, h_crop_in + output_fit_h))

    return incoming_pts, view_pts


def rotate_affine_targets(targets, degrees, outputscreen_shape):
    mid_img = [int(x/2) for x in outputscreen_shape[0:2][::-1]] # get reversed dims
    new_target = AffinePoints(
                top_left_w_h=rotate_pt_around_origin(targets.top_left_w_h, mid_img, degrees),
                top_right_w_h=rotate_pt_around_origin(targets.top_right_w_h, mid_img, degrees),
                lower_right_w_h=rotate_pt_around_origin(targets.lower_right_w_h, mid_img, degrees))
    return new_target


def test_viewer(
        inputimage,
        pausetime_Secs=0,
        presskey=False,
        destroyWindow=True):

    cv2.imshow("img", inputimage)
    if presskey==True:
        cv2.waitKey(0); #any key

    if presskey==False:
        if cv2.waitKey(20) & 0xFF == 27:
                pass
    if pausetime_Secs>0:
        time.sleep(pausetime_Secs)
    if destroyWindow==True: cv2.destroyAllWindows()


def rotate_img_orthogonal(img, rotation: Literal[0, 90, -90, 180, 270]):
    if rotation in [0, 360]:
        return img
    if rotation in [90]:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotation in [-90, 270]:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation in [180]:
        return cv2.rotate(img, cv2.ROTATE_180)
    raise ValueError("bad rotation req", rotation)


def get_empty_lumodisplay_img(imgshape: tuple[int, int]):
    return np.zeros(
            (imgshape + (3,)), np.uint8)


def print_text_in_boundingbox(text: str, grayscale: bool):
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1.0
    FONT_THICKNESS = 2

    (label_width, label_height), baseline = cv2.getTextSize(
        text,
        FONT,
        FONT_SCALE,
        FONT_THICKNESS)

    label_patch = np.zeros((label_height + baseline, label_width, 3), np.uint8)

    cv2.putText(
        label_patch,
        text,
        (0, label_height),
        FONT,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS)
    
    if grayscale is True:
        label_patch = cv2.cvtColor(label_patch, cv2.COLOR_BGR2GRAY)

    return label_patch

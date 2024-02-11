import sys
import cv2
import shutil
from enum import Enum, auto
import os
import numpy as np
import time
from contextlib import contextmanager
from typing import Iterator, Literal
from dataclasses import dataclass
from skimage.draw import line
from my_collections import CropSlicing, AffinePoints, UI_ready_element
from math import floor

RED = (0, 0, 255)
BLUE = (255, 0, 0)

def read_img(img_filepath):
    return cv2.imread(img_filepath)

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

def add_cross_hair(image, adapt):
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

    image[midx - thick : midx + thick, 0:midy-vis_block, 2] = 0
    image[midx - thick : midx + thick, 0:midy-vis_block, 1] = max(col, 50)
    image[midx - thick : midx + thick, midy+vis_block:-1, 2] = 0
    image[midx - thick : midx + thick, midy+vis_block:-1, 1] = max(col, 50)

    image[0:midx-vis_block, midy - thick : midy + thick, 2] = 0
    image[0:midx-vis_block, midy - thick : midy + thick, 1] = max(col, 50)
    image[midx+vis_block:-1, midy - thick : midy + thick, 2] = 0
    image[midx+vis_block:-1, midy - thick : midy + thick, 1] = max(col, 50)
 
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


def add_ui_elements(image, element_package: UI_ready_element) -> None:
    image[
        element_package.position.top:element_package.position.lower,
          element_package.position.left:element_package.position.right,
          
          0
          ] = element_package.image


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

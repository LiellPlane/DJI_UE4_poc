import numpy as np
from abc import ABC, abstractmethod
import cv2
import time
import math
import os
from dataclasses import dataclass
from time import perf_counter
from contextlib import contextmanager
import random
import enum
import fisheye_lib
from typing import Optional

class Camera(ABC):
    
    @property
    def angle_vs_world_up(self):
        raise NotImplementedError

    def __init__(self) -> None:
        self.res_select = 0
        self.last_img = None

    @abstractmethod
    def gen_image(self):
       pass

    def __next__(self):
        img = self.gen_image()
        #if len(img.shape) == 3:
        #    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #self.last_img = img
        return img

    def __iter__(self):
        return self
    

class Fake_CSI_Camera(Camera):
    
    def get_res(self):
        pass

    def gen_image(self):
        blank_image = np.zeros((500, 500, 3), np.uint8)
        blank_image[:,:,:] = random.randint(0,255)
        blank_image = cv2.circle(
            blank_image,
            (250, 250),
            40,
            50,
            5)
        return blank_image
    

class CSI_Camera(Camera):
    
    #angle_vs_world_up = 90

    def __init__(self) -> None:
        super().__init__()
        self.picam2 = Picamera2()
        _config = self.picam2.create_video_configuration(
            main={"size": (640, 480),  "format": "YUV420"})
        #self.picam2.set_controls({"ExposureTime": 1000}) # for blurring - but can get over exposed at night
        self.picam2.configure(_config)
        self.picam2.start()
        time.sleep(0.1)

    def _gen_image(self):
        output = self.picam2.capture_array("main")
        return output

    def gen_image(self):
        return self._gen_image()

    def __del__(self):
        # this doesn't seem to end cleanly
        self.picam2.stop()

def get_platform():
    #  detect what OS we are on - test environment (Windows) or production (pi hardware)
    RASP_PI_4_OS = "armv7l"

    if hasattr(os, 'uname') is False:
        print("raspberry presence failed, probably Windows system")
        return _OS.WINDOWS
    elif os.uname()[-1] == RASP_PI_4_OS:
        print("raspberry presence detected, loading hardware libraries")
        return _OS.RASPBERRY
    else:
        raise Exception("Could not detect platform")


class _OS(str, enum.Enum):
    WINDOWS = "windows"
    RASPBERRY = "raspberry"

if get_platform() == _OS.RASPBERRY:
    # sorry not sorry
    import rpi_ws281x as leds
    from picamera2 import Picamera2


class Leds(ABC):

    def __init__(self):
        self.led_count      = 300 # number of led pixels.
        self.led_pin        = 18      # gpio pin connected to the pixels (18 uses pwm!).
        #led_pin        = 10      # gpio pin connected to the pixels (10 uses spi /dev/spidev0.0).
        self.led_freq_hz    = 800000  # led signal frequency in hertz (usually 800khz)
        self.led_dma        = 10      # dma channel to use for generating a signal (try 10)
        self.led_brightness = 60      # set to 0 for darkest and 255 for brightest
        self.led_invert     = False   # true to invert the signal (when using npn transistor level shift)
        self.led_channel    = 0
        #self.req_led_cols = [(0, 0, 0)] * self.led_count

    @abstractmethod
    def set_LED_values(self):
        pass

    @abstractmethod
    def execute_LEDS(self):
        pass


class SimLeds(Leds):

    def __init__(self):
        super().__init__()

    def set_LED_values(self, scambi_units: list):
        # don't do anything
        #if len(scambi_units) > self.led_count:
        #    raise Exception("Too many leds for configured strip")
        for index, scambiunit in enumerate(scambi_units):
            if index > self.led_count:
                break
            scambiunit.colour

    def execute_LEDS(self):
        pass

    def display(self, *args, **kwargs):
        ImageViewer_Quick_no_resize(*args, **kwargs)


class ws281Leds(Leds):
    
    def __init__(self):
        super().__init__()
        # Create NeoPixel object with configuration.
        self.strip = leds.Adafruit_NeoPixel(
            self.led_count,
            self.led_pin,
            self.led_freq_hz,
            self.led_dma,
            self.led_invert,
            self.led_brightness,
            self.led_channel)

        try:
        # Intialize the library (must be called once before other functions).
            self.strip.begin()
        except RuntimeError:
            print("**************")
            print("Try running as SUDO or ROOT user")
            print("**************")

        self.test_leds()

    def set_LED_values(self, scambi_units: list):
        #if len(scambi_units) > self.led_count:
        #    raise Exception("Too many leds for configured strip")
        for index, scambiunit in enumerate(scambi_units):
            if index > self.led_count:
                break
            self.strip.setPixelColor(
                index,
                leds.Color(*scambiunit.colour))

    def execute_LEDS(self):
        self.strip.show()

    def display(self, *args, **kwargs):
        #  no display - pass through
        pass

    def test_leds(self):
        for i in range (0, 50):
            for i in range(self.strip.numPixels()):
                color =  leds.Color(
                    random.randint(0,1)*255,
                    random.randint(0,1)*255,
                    random.randint(0,1)*255)
                self.strip.setPixelColor(i, color)
            self.execute_LEDS()
        for i in range (0, 50):
            for i in range(self.strip.numPixels()):
                color =  leds.Color(0, 0, 0)
                self.strip.setPixelColor(i, color)
            self.execute_LEDS()

@contextmanager
def time_it(comment):
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        toc: float = time.perf_counter()
        print(f"{comment}:proc time = {1000*(toc - tic):.3f}ms")
#@dataclass
class Scambi_unit():
    def __init__(self,
                 led_positionxy: tuple,
                 sample_area_left: int,
                 sample_area_right: int,
                 sample_area_top: int,
                 sample_area_lower: int,
                 inverse_warp_m: any,
                 img_shape: Optional[any],
                 img_circle: Optional[int]):

        """supply led and roi positions on a flat representation of
        the screen
        
        inverse_warp will map these positions with the inverse homography
        if the screen isn't presented flat
        
        img_shape and img_circle are optional parameters that will 
        additionally warp the already perspective warped rois
        to a fisheye of img_circle"""
        self.img_shape = img_shape
        self.img_circle = img_circle
        self.led_positionxy = led_positionxy
        self.sample_area_left = sample_area_left
        self.sample_area_right = sample_area_right
        self.sample_area_top = sample_area_top
        self.sample_area_lower = sample_area_lower
        self.inverse_warp_m = inverse_warp_m
        self.roi = []
        self.warped_led_pos = None
        self.warped_bounding_rect = None
        # this all needs to go in a dataclass or something
        self.bb_left = None
        self.bb_right = None
        self.bb_top = None
        self.bb_lower = None
        self.colour = (0, 0, 0)
        self.sample_area_lerp_contour = None
        self.convex_hulls_lerp_contour = None
        self.lerp_sample_area()
        self.warp_rois_homograpy()
        self.fisheye_rois()

    def lerp_sample_area(self):
        """Create interpolation between corners so we can
        warp in a non-linear fashion such as fish eye"""
        contours = []
        # along top left to right
        lin = np.linspace(self.sample_area_left, self.sample_area_right, 10)
        for x in lin:
            contours.append((x, self.sample_area_top))
        # down right side
        lin = np.linspace(self.sample_area_top, self.sample_area_lower, 10)
        for y in lin:
            contours.append((self.sample_area_right, y))
        # along bottom right to left
        lin = np.linspace(self.sample_area_right, self.sample_area_left, 10)
        for x in lin:
            contours.append((x, self.sample_area_lower))
        # up left side
        lin = np.linspace(self.sample_area_lower, self.sample_area_top, 10)
        for y in lin:
            contours.append((self.sample_area_left, y))
        cont_ints = [(int(i[0]), int(i[1])) for i in contours]
        self.sample_area_lerp_contour = cont_ints

    def warp_rois_homograpy(self):
        """warp rois if we just have a perspective warp"""
        # warp sample colour region
        temp_roi = []
        temp_roi.append(np.asarray([self.sample_area_left, self.sample_area_top, 1]))
        temp_roi.append(np.asarray([self.sample_area_left, self.sample_area_lower, 1]))
        temp_roi.append(np.asarray([self.sample_area_right, self.sample_area_lower, 1]))
        temp_roi.append(np.asarray([self.sample_area_right, self.sample_area_top, 1]))
        for pt in temp_roi:
            homog_coords = np.matmul(self.inverse_warp_m, pt)
            new_pt = list((np.floor(homog_coords[0:2]/homog_coords[-1])).astype(int))
            self.roi.append(new_pt)
        self.roi = np.asarray(self.roi).astype(np.int32)
        self.roi = self.roi.reshape((-1, 1, 2))

        # warp expected LED region
        temp_led_pos = np.asarray(self.led_positionxy + (1,))
        homog_coords = np.matmul(self.inverse_warp_m, temp_led_pos)
        self.warped_led_pos = tuple((np.floor(homog_coords[0:2]/homog_coords[-1])).astype(int))

        #get bounding box for ROI
        self.warped_bounding_rect = cv2.boundingRect(self.roi)
        left, top, w, h = self.warped_bounding_rect
        right = left + w
        lower = top + h
        self.bb_left = left
        self.bb_right = right
        self.bb_top = top
        self.bb_lower = lower

        #warp the lerped rois (rois with points between corners for non-linear warping)
        warped_lerp_roi = []
        for lerp_pt in self.sample_area_lerp_contour:
            temp_lerp_pos = np.asarray(lerp_pt + (1,))
            homog_coords = np.matmul(self.inverse_warp_m, temp_lerp_pos)
            warped_lerp_pt = tuple((np.floor(homog_coords[0:2]/homog_coords[-1])).astype(int))
            warped_lerp_roi.append(warped_lerp_pt)
        self.sample_area_lerp_contour = warped_lerp_roi
        
    def fisheye_rois(self):
        """warp the rois to fisheye - expected to have
        warped perspective first"""
        
        fisheyeser = fisheye_lib.fisheye_tool(
            img_width_height=tuple(reversed(self.img_shape[0:2])),
            image_circle_size=self.img_circle)
        
        temp_pts = []

        for pt in self.sample_area_lerp_contour:
            temp_pts.append(
                fisheyeser.brute_force_find_fisheye_pt(pt))

        self.sample_area_lerp_contour = temp_pts
        self.convex_hulls_lerp_contour = fisheye_lib.convert_pts_to_convex_hull(
            self.sample_area_lerp_contour)

        self.warped_led_pos = fisheyeser.brute_force_find_fisheye_pt(self.warped_led_pos)
        
        #recalculate
        fish_roi = np.asarray([np.asarray(fisheyeser.brute_force_find_fisheye_pt(list(pt[0]))) for pt in self.roi]).reshape(-1, 1, 2)
        self.warped_bounding_rect = cv2.boundingRect(fish_roi)
        left, top, w, h = self.warped_bounding_rect
        right = left + w
        lower = top + h
        self.bb_left = left
        self.bb_right = right
        self.bb_top = top
        self.bb_lower = lower
    
    def draw_lerp_contour(self, img):

        cv2.drawContours(
            image=img,
            contours=[self.convex_hulls_lerp_contour],
            contourIdx=-1,
            color=(100,200,250),
            thickness=1,
            lineType=cv2.LINE_AA)
        
        return img

    def draw_warped_roi(self, img):
        #for pt in self.roi:
        #    img[pt[1], pt[0], :] = (255, 255, 255)
        img = cv2.polylines(img,
                            [self.roi],
                      isClosed=True, color=(255,255,255),
                      thickness=1)
        
        #cv2.drawContours(img, [br], -1, (0,0,255), 1)
        #cv2.circle(img,(br[2], br[3]),2,(255,255,255),-1)
        #cv2.rectangle(img,(br[0], br[1]),(br[2], br[3]),(0,255,0),3)
        return img
    
    def draw_warped_boundingbox(self,img):

        x,y,w,h = self.warped_bounding_rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 1)

        return img

    def draw_led_pos(self, img, colour=None, offset=None, size=16):

        pos = self.led_positionxy
        if offset is not None:
            pos = (pos[0] + offset[0], pos[1] + offset[1])
        if colour is None:
            cv2.circle(img,pos,size,(255,0,100),-1)
        else:
            cv2.circle(img,pos,size,colour,-1)

    def draw_warped_led_pos(self, img, colour=None, offset=None, size=16):
    
        pos = self.warped_led_pos
        if offset is not None:
            pos = (pos[0] + offset[0], pos[1] + offset[1])
        if colour is None:
            cv2.circle(img,pos,size,(255,0,100),-1)
        else:
            cv2.circle(img,pos,size,colour,-1)

        return img

    def draw_rectangle(self, img):
        draw_rectangle(
            self.sample_area_left,
            self.sample_area_right,
            self.sample_area_top,
            self.sample_area_lower, img)

    def get_dominant_colour_perspective(self, img):
        pass

    def get_dominant_colour_flat(self, img):
        sample_area = img[
            self.bb_top:self.bb_lower,
            self.bb_left:self.bb_right,
            :]
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
        return tuple(dom_col)
        return (0, 0, 0)
    #def get_warped_area(self):

    
    #def draw_poly(self, img):
    #    return(cv2.fillPoly(img, [np.array(myROI)])


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
    
def ImageViewer_Quick_no_resize(inputimage,pausetime_Secs=0,presskey=False,destroyWindow=True):
    if inputimage is None:
        print("input image is empty")
        return
    ###handy quick function to view images with keypress escape andmore options
    cv2.imshow("img", inputimage.copy()); 


    if presskey==True:
        cv2.waitKey(0); #any key
   
    if presskey==False:
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                #for some reason
                pass
            
    if pausetime_Secs>0:
        time.sleep(pausetime_Secs)
    if destroyWindow==True: cv2.destroyAllWindows()

def img_width(img):
    return img.shape[1]

def img_height(img):
    return img.shape[0]
#cache this
def perimeter_spacing(img_dim, no_of_leds):
    perimeter_pxls = img_dim
    # maybe work out closest integer here 
    remainder = perimeter_pxls % no_of_leds
    spacing = (perimeter_pxls-remainder)/no_of_leds
    return int(spacing)

def get_led_perimeter_pos(img, no_leds_vert, no_leds_horiz):
    led_spacing_vert = perimeter_spacing(img_height(img), no_leds_vert)
    led_spacing_horiz = perimeter_spacing(img_width(img), no_leds_horiz)
    left = [(0,i ) for i in range(0, img_height(img), led_spacing_vert)]
    right = [(img_width(img), i) for i in range(0, img_height(img), led_spacing_vert)]
    top = [(i, 0) for i in range(0, img_width(img), led_spacing_horiz)]
    lower = [(i, img_height(img)) for i in range(0, img_width(img), led_spacing_horiz)]
    while True:
        for pos in left:
            yield pos, "left"
        for pos in top:
            yield pos, "top"
        for pos in right:
            yield pos, "right"
        for pos in lower:
            yield pos, "lower"
        break

def create_rectangle_from_centrepoint(centrepoint, edge):
    half_edge = int(edge/2)
    posx =centrepoint[0]
    posy = centrepoint[1]
    left = posx - half_edge
    right = posx + half_edge
    top = posy - half_edge
    lower = posy + half_edge
    return left, right, top, lower

def draw_rectangle(left, right, top, down, img):
    rec = cv2.rectangle(img,
                  (left, top),
                  (right, down),
                  (0,100,255),
                  8)
    return rec

class real_fish_eye_cam():
    vid = r"C:\Working\nonwork\SCAMBILIGHT\fisheye_taped.mp4"
    # 1296 * 976 or whatever mode it is
    # width = 1296
    # height = 972

    # fish_eye_circle = 1296 - 150
    width = 1269
    height = 972

    fish_eye_circle = 1269

    # these are corners from fisheye so need
    # to be converted/reversed
    #"C:\Working\nonwork\SCAMBILIGHT\fisheye_taped.png"
    corners = np.asarray([
        [105, 257],
        [1133, 358],
        [855, 603],
        [360, 572]], dtype="float32")

    targets = np.asarray([
        [0, 0],
        [width,0 ],
        [width, height],
        [0, height]], dtype="float32")

class get_homography():
    # class variables
    demo_img = r"C:\Working\nonwork\SCAMBILIGHT\calibrate_screen.png"
    demo_vid = r"C:\Working\nonwork\SCAMBILIGHT\fluid_sim.mp4"
    demo_width = 1920
    demo_height = 1080
    # demo_corners = np.asarray([
    #     [419, 204],
    #     [1707, 327],
    #     [1761, 1038],
    #     [91, 786]], dtype="float32")
    demo_corners = np.asarray([
        [640, 340],
        [1391, 292],
        [1413, 799],
        [669, 778]], dtype="float32")
    demo_targets = np.asarray([
        [0, 0],
        [demo_width,0 ],
        [demo_width, demo_height],
        [0, demo_height]], dtype="float32")

    def __init__(
            self,
            img_height_,
            img_width_,
            corners,
            target_corners,
            resize_ratio):
        """ calibrate_pts = np.asarray([
        [419, 204],
        [1707, 327],
        [1761, 1038],
        [91, 786]], dtype="float32")"""
        self._img_height = img_height_ * resize_ratio
        self._img_width = img_width_ * resize_ratio
        self._corners = corners * resize_ratio
        self._target_corners = target_corners * resize_ratio
        self.trans_matrix = cv2.getPerspectiveTransform(
                    self._corners,
                    self._target_corners)
        self.inverse_trans_matrix = np.linalg.inv(
            self.trans_matrix
            )

    def warp_img(self, img):
        return(cv2.warpPerspective(
            img,
            self.trans_matrix,
            list(img.shape[0:2]).reverse()
            ))

class Find_Screen():
    def __init__(self) -> None:
        self.motion_img = None
        self.firstFrame = None
        self.backSub = cv2.createBackgroundSubtractorKNN(
            history=5000,
            dist2Threshold=4000.0,
            detectShadows=False)
        self.kernel = np.ones((50,50),np.float32)/25
    def input_image(self, img):
        img_resize = cv2.resize(img,(640, 480))
        fgMask = self.backSub.apply(img_resize)
        if self.motion_img is None:
            self.motion_img = np.zeros_like(fgMask)
            self.motion_img = self.motion_img.astype(int)
        
        dst = cv2.filter2D(fgMask,-1,self.kernel)
        self.motion_img = np.add(self.motion_img, dst)
        
        #if self.motion_img.max() > 1000:
        self.motion_img = np.subtract(self.motion_img, 50)
        self.motion_img = np.clip(self.motion_img, 0, 2**32)
        #fgMask = cv2.fastNlMeansDenoising(fgMask)
        #fgMask = cv2.fastNlMeansDenoising(fgMask,None,10,10,7,21)
        print(self.motion_img.max())
        output = self.motion_img/(self.motion_img.max()/254)
        output = self.motion_img.astype("uint8")
        ImageViewer_Quick_no_resize(img,0,False,False)

def main_check_calib():
    fart = get_homography(
        img_height_=get_homography.demo_height,
        img_width_=get_homography.demo_width,
        corners=get_homography.demo_corners,
        target_corners=get_homography.demo_targets,
        resize_ratio=1)
     #calibrate_screen.png homography points
    calib_img = cv2.imread(r"C:\Working\nonwork\SCAMBILIGHT\calibrate_screen.png")
    ImageViewer_Quick_no_resize(calib_img,0,True,False)
    # calibrate_pts = np.zeros((4, 2), dtype="float32")
    # calibrate_pts = np.asarray([
    #     [419, 204],
    #     [1707, 327],
    #     [1761, 1038],
    #     [91, 786]], dtype="float32")

    # source_pts = np.asarray([
    #     [0, 0],
    #     [calib_img.shape[1],0 ],
    #     [calib_img.shape[1], calib_img.shape[0]],
    #     [0, calib_img.shape[0]]], dtype="float32")
    
    

    # for coord in calibrate_pts:
    #     cv2.circle(
    #         calib_img,
    #         tuple(np.asarray(coord, dtype="int")),
    #         16,
    #         (255,0,100),
    #         -1)

    # trans_m = cv2.getPerspectiveTransform(
    #                 calibrate_pts,
    #                 source_pts)

    warped = cv2.warpPerspective(
        calib_img,
        fart.trans_matrix,
        list(calib_img.shape[0:2]).reverse()
        )
    
    #ImageViewer_Quick_no_resize(warped,0,True,False)

    dewarped = cv2.warpPerspective(
        warped,
        fart.inverse_trans_matrix,
        list(warped.shape[0:2]).reverse()
        )

    ImageViewer_Quick_no_resize(dewarped,0,True,False)

def test_find_screen():
    input_vid = r"C:\Working\nonwork\SCAMBILIGHT\test_raspberrypi_v2.mp4"
    screen_finder = Find_Screen()
    cap = cv2.VideoCapture(input_vid)
    while True:
        suc, frame = cap.read()
        screen_finder.input_image(frame)



def main():
    system = get_platform()
    if system == _OS.WINDOWS:
        led_subsystem = SimLeds()
        cam = Fake_CSI_Camera()
    elif system == _OS.RASPBERRY:
        cam = CSI_Camera()
        led_subsystem = ws281Leds()
    else:
        raise Exception(system + " not supported")
    no_leds_vert = 20
    no_leds_horiz = 30
    input_vid_loc = [
        r"C:\Working\nonwork\SCAMBILIGHT\fisheye_taped.mp4",
        r"/home/scambilight/test_raspberrypi_v2.mp4"]
    input_vid = None
    for filepath in input_vid_loc:
        if os.path.exists(filepath):
            input_vid = filepath
            break
    if input_vid is None:
        raise Exception("No valid video paths found")
    #cap = cv2.VideoCapture("udp://127.0.0.1:1234?overrun_nonfatal=1&fifo_size=500000", cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(input_vid)
    move_in_horiz = 0.1
    move_in_vert = 0.1
    resize_ratio = 1.0 #expected input res 1080 * 1920
    sample_area_edge = 90 * resize_ratio
    demo_border_size = 10
    
    if resize_ratio != 1.0:
        raise Exception("any non unity values break the fisheye stuff in scambiunit")
    rfish = real_fish_eye_cam

    # fisheriser = fisheye_lib.fisheye_tool(
    #     img_width_height=(rfish.width, rfish.height),
    #     image_circle_size=rfish.fish_eye_circle)


    homography_tool = get_homography(
        img_height_=rfish.height,
        img_width_=rfish.width,
        corners=rfish.corners,
        target_corners=rfish.targets,
        resize_ratio=resize_ratio)



    suc, prev = cap.read()
    while suc is False:
        print("failed to grab image")
        time.sleep(0.05)
        suc, prev = cap.read()
    prev = cv2.resize(
            prev,
            (int(img_width(prev)*resize_ratio),int(img_height(prev)*resize_ratio)))


    scambi_units = []
    led_positions = get_led_perimeter_pos(prev, no_leds_vert, no_leds_horiz)
    for centre_, edgename in led_positions:
            centre_ = tuple((np.asarray(centre_)).astype(int))
            #cv2.circle(prev,plop,16,(255,0,100),-1)
            mid_screen = (np.array(tuple(reversed(prev.shape[:2])))/2).astype(int)[:2]
            vec_to_midscreen = mid_screen-np.asarray(centre_)
            #cv2.circle(prev,tuple(mid_screen),16,(255,0,100),-1)
            if edgename not in ["top", "lower", "left", "right"]:
                raise Exception("edge name " + edgename + "not valid")
            if edgename in ["top", "lower"]:
                new_pos = tuple((np.asarray(centre_) + (vec_to_midscreen * move_in_vert)).astype(int))
            if edgename in ["left", "right"]:
                new_pos = tuple((np.asarray(centre_) + (vec_to_midscreen * move_in_horiz)).astype(int))
            left, right, top, lower = create_rectangle_from_centrepoint(new_pos, edge=sample_area_edge)
            scambi_units.append(Scambi_unit(
                led_positionxy=centre_,
                sample_area_left=left,
                sample_area_right=right,
                sample_area_top=top,
                sample_area_lower=lower,
                inverse_warp_m=homography_tool.inverse_trans_matrix,
                img_shape=prev.shape,
                img_circle=real_fish_eye_cam.fish_eye_circle)
            )
    
    prev_copy = np.zeros_like(prev)
    empty_img = np.zeros((
        prev.shape[0]+(demo_border_size*2),
        prev.shape[1]+(demo_border_size*2),
        3), dtype="uint8")
    

    
    
    while True:
        suc, prev = cap.read()
        while suc is False:
            print("failed to grab image")
            time.sleep(0.05)
        #prev = fisheriser.fish_eye_image(prev, reverse=True)
        # prev = cv2.resize(
        #     prev,
        #     (int(img_width(prev)*resize_ratio),int(img_height(prev)*resize_ratio)))
        for unit in scambi_units:
            #unit.draw_led_pos(prev)
            #unit.draw_rectangle(prev)
            unit.colour = unit.get_dominant_colour_flat(prev)
            
            #prev = unit.draw_warped_roi(prev)
            prev = unit.draw_lerp_contour(prev)
            #prev = unit.draw_warped_boundingbox(prev)
            prev = unit.draw_warped_led_pos(
                prev,
                unit.colour,
                offset=(0, 0),
                size=40)
            
        ImageViewer_Quick_no_resize(prev,0,False,False)
















        continue

        led_background = empty_img.copy()
        mask = empty_img.copy()
        bigger_foreimage = empty_img.copy()
        pts = np.asarray([
            i+np.asarray([demo_border_size, demo_border_size])
            for i in homography_tool._corners.astype(int)])
        cv2.fillPoly(mask, pts=[pts], color=(255, 255, 255))
        mask=mask/255
        suc, prev = cap.read()

        prev = fisheriser.fish_eye_image(prev, reverse=True)
        #prev = homography_tool.warp_img(prev)
        prev = cv2.resize(
            prev,
            (int(img_width(prev)*resize_ratio),int(img_height(prev)*resize_ratio)))
        for unit in scambi_units:
            #unit.draw_led_pos(prev)
            #unit.draw_rectangle(prev)
            unit.colour = unit.get_dominant_colour_flat(prev)
            
            prev = unit.draw_warped_roi(prev)
            unit.draw_warped_led_pos(
                led_background,
                unit.colour,
                offset=(demo_border_size, demo_border_size),
                size=80)
            #unit.draw_led_pos(prev, dom_colour)
        led_subsystem.set_LED_values(scambi_units)
        led_subsystem.execute_LEDS()
        #ImageViewer_Quick_no_resize(prev,0,False,False)
        blurred = cv2.blur(led_background, (120, 120))

        with time_it(" get dom colours"):
            for unit in scambi_units:
                dom_colour = unit.get_dominant_colour_flat(prev)
        with time_it("get cam image"):
            test_img = next(cam)
        print("scambi_units n =", len(scambi_units))
        for unit in scambi_units:
                #unit.draw_led_pos(prev)
            
            #dom_colour = unit.get_dominant_colour_flat(prev)
            #unit.draw_warped_boundingbox(prev)
            

            unit.draw_warped_led_pos(
                blurred,
                unit.colour,
                offset=(demo_border_size, demo_border_size),
                size=8)
    

        
        bigger_foreimage[
            demo_border_size:prev.shape[0]+demo_border_size,
            demo_border_size:prev.shape[1]+demo_border_size,
            :] =prev[:,:,:]
        
        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(mask.astype('uint8'), bigger_foreimage)
        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1 - mask.astype('uint8'), blurred)
        # Add the masked foreground and background.
        outImage = cv2.add(foreground, background)
        outImage = cv2.flip(outImage, 0)


        #fart = prev_copy.copy()
        #for unit in scambi_units:
        #    for xxx, yyy in unit.sample_area_lerp_contour:
        #        fart[yyy, xxx, :] = (255,255,255)


        led_subsystem.display(outImage, 0, False, False)
        #ImageViewer_Quick_no_resize(outImage,0,False,False)
if __name__ == "__main__":
    #test_find_screen()
    main()

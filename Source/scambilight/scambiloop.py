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
from typing import Optional
import requests
import base64
import json
from typing import Literal
from libs.utils import (
    get_platform,
    _OS,
    TimeDiffObject,
    ImageViewer_Quick_no_resize,
    encode_img_to_str)
from libs.scambiunits import Scambi_unit, ScambiInit
from libs.collections import (
    LedSpacing,
    Edges,
    lens_details,
    LedsLayout,
    config_corner)
import libs.async_cam_lib as async_cam_lib
import libs.fisheye_lib as fisheye_lib
from libs.lighting import SimLeds, ws281Leds

PLATFORM = get_platform()
if PLATFORM == _OS.RASPBERRY:
    # sorry not sorry
    import rpi_ws281x as leds
    #from picamera2 import Picamera2



class DaisybankLedSpacing():
    def __init__(self) -> None:
        self.edges = {
            Edges.LEFT: LedsLayout(
                clockwise_start=73, clockwise_end=110),
            Edges.TOP: LedsLayout(
                clockwise_start=114, clockwise_end=181),
            Edges.RIGHT: LedsLayout(
                clockwise_start=185, clockwise_end=223),
            Edges.LOWER: LedsLayout(
                clockwise_start=229, clockwise_end=296)}
        



@contextmanager
def time_it(comment):
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        toc: float = time.perf_counter()
        if random.randint(1,100) < 4:
            print(f"{comment}:proc time = {1000*(toc - tic):.3f}ms")



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



def get_led_perimeter_pos(img, no_leds_vert, no_leds_horiz) -> LedSpacing:
    # imagine moving around the screen in a clockwise manner to
    # determine what is 0% and 100% of an edge
    led_spacing_vert = perimeter_spacing(img_height(img), no_leds_vert)
    led_spacing_horiz = perimeter_spacing(img_width(img), no_leds_horiz)
    x = 1
    y = 0
    reversed = 1
    while True:
        for pos in [(0,i ) for i in range(0, img_height(img), led_spacing_vert)]:
            yield LedSpacing(
                positionxy=pos,
                edge=Edges.LEFT,
                normed_pos_along_edge_mid=reversed - round(pos[x]/img_height(img),3),
                normed_pos_along_edge_start=reversed - round((pos[x]-(led_spacing_vert/2))/img_height(img),3),
                normed_pos_along_edge_end=reversed - round((pos[x]+(led_spacing_vert/2))/img_height(img),3))

        for pos in [(i, 0) for i in range(0, img_width(img), led_spacing_horiz)]:
            yield LedSpacing(
                positionxy=pos,
                edge=Edges.TOP,
                normed_pos_along_edge_mid=round(pos[y]/img_width(img),3),
                normed_pos_along_edge_start=round((pos[y]-(led_spacing_horiz/2))/img_width(img),3),
                normed_pos_along_edge_end=round((pos[y]+(led_spacing_horiz/2))/img_width(img),3))

        for pos in [(img_width(img), i) for i in range(0, img_height(img), led_spacing_vert)]:
            yield LedSpacing(
                positionxy=pos,
                edge=Edges.RIGHT,
                normed_pos_along_edge_mid=round(pos[x]/img_height(img),3),
                normed_pos_along_edge_start=round((pos[x]-(led_spacing_vert/2))/img_height(img),3),
                normed_pos_along_edge_end=round((pos[x]+(led_spacing_vert/2))/img_height(img),3))

        for pos in [(i, img_height(img)) for i in range(0, img_width(img), led_spacing_horiz)]:
            yield LedSpacing(
                positionxy=pos,
                edge=Edges.LOWER,
                normed_pos_along_edge_mid=reversed - round(pos[y]/img_width(img),3),
                normed_pos_along_edge_start=reversed - round((pos[y]-(led_spacing_horiz/2))/img_width(img),3),
                normed_pos_along_edge_end=reversed - round((pos[y]+(led_spacing_horiz/2))/img_width(img),3))
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


class get_homography():

    def __init__(
            self,
            img_height_,
            img_width_,
            corners,
            target_corners):
        """ calibrate_pts = np.asarray([
        [419, 204],
        [1707, 327],
        [1761, 1038],
        [91, 786]], dtype="float32")"""
        self._img_height = img_height_ 
        self._img_width = img_width_ 
        self._corners = np.asarray(corners,  dtype="float32") 
        self._target_corners = np.asarray(target_corners,  dtype="float32") 
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



def get_corners_from_remote_config(config, img):
    """find corners from disorder of inputs in format:
    {
                "clickx": 299
                "clicky": 339}
    """
    #min_x = min([i['clickx'] for i in config])
    corners = {}
    corners["top_left"] = [0, 0]
    corners["top_right"] = [img_width(img), 0 ]
    corners["lower_right"] = [img_width(img), img_height(img),]
    corners["lower_left"] =  [0, img_height(img)]
    list_config_pts = [[i['clickX'], i['clickY']] for i in config]
    for pt_id, pt_coord in corners.items():
        match_pt, list_config_pts = find_closest(pt_coord, list_config_pts)
        arse = config_corner(flat_corner=corners[pt_id], real_corner=match_pt)
        corners[pt_id] = arse
    return corners

def find_closest(testpt: list [int, int], input_pts:list):
    dists = {
        np.linalg.norm(np.asarray(testpt)-np.asarray(i)):i
        for i in input_pts}
    pt = dists[sorted(dists)[0]]
    return pt, [i for i in input_pts if i != pt]

    
def main():

    daisybank_hd_lens_details_HD = {
        'id': 'daisybank HD',
        'vid': 'C:\\Working\\nonwork\\SCAMBILIGHT\\fisheye_taped.mp4',
        'width': 1269,
        'height': 972,
        'fish_eye_circle': 1250,
        'corners': [[81, 234],[1157, 346],[860, 600],[363, 572]]}

    daisybank_hd_lens_details_LD = {
        'id': 'daisybank HD',
        'vid': 'C:\\Working\\nonwork\\SCAMBILIGHT\\fisheye_taped.mp4',
        'width': 640,
        'height': 480,
        'fish_eye_circle': 600,
        'corners': [[18, 114], [587, 173], [431, 301], [176, 285]]}

    rfish = lens_details(**daisybank_hd_lens_details_HD)
    system = get_platform()
    if system == _OS.WINDOWS:
        led_subsystem = SimLeds(DaisybankLedSpacing)
        cam = async_cam_lib.Synth_Camera_Async(async_cam_lib.ScambiLight_Cam_vidmodes)
        cores = 8
    elif system == _OS.RASPBERRY:
        cam = async_cam_lib.Scamblight_Camera_Async(async_cam_lib.ScambiLight_Cam_vidmodes)
        #cam = async_cam_lib.ScambilightCamImageGen([e.value for e in async_cam_lib.ScambiLight_Cam_vidmodes][0][1][0:2])
        led_subsystem = ws281Leds(DaisybankLedSpacing)
        cores = 3
    else:
        raise Exception(system + " not supported")
    no_leds_vert = 11
    no_leds_horiz = 20
    move_in_horiz = 0.2
    move_in_vert = 0.2
    #resize_ratio = 1.0 #expected input res 1080 * 1920
    sample_area_edge = 80
    subsample_cut = 15 # min edge pxls, we can subsample areas of image to speed up, but we don't want to subsample small areas into nothing
    cores_for_col_dect = cores
    img_upload_url = "https://yqnz152azi.execute-api.us-east-1.amazonaws.com/Prod/hello" # for AWS experiment

    #if resize_ratio != 1.0:
     #   raise Exception("any non unity values break the fisheye stuff in scambiunit")
    

    # fisheriser = fisheye_lib.fisheye_tool(
    #     img_width_height=(rfish.width, rfish.height),
    #     image_circle_size=rfish.fish_eye_circle)


    prev = next(cam)
    # upload before anything crashes - handy when changing res
    # send test image to aws
    fisheriser = fisheye_lib.fisheye_tool(
        img_width_height=(rfish.width, rfish.height),
        image_circle_size=rfish.fish_eye_circle)
    img_2_upload = fisheriser.fish_eye_image(next(cam), reverse=True)
    upload_img_to_aws(img_2_upload, img_upload_url, action = "raw")


    real_corners = rfish.corners
    positions = get_config_from_aws(img_upload_url)
    if len(positions) > 3:
        real_corners = get_corners_from_remote_config(positions, prev)
        real_corners = [real_corners['top_left'].real_corner,
        real_corners['top_right'].real_corner,
        real_corners['lower_right'].real_corner,
        real_corners['lower_left'].real_corner]
    else:
        print("not enough positions in remote config ", positions)
    #raise Exception("stop stop")

    print(positions)
    homography_tool = get_homography(
        img_height_=rfish.height,
        img_width_=rfish.width,
        corners=real_corners,
        target_corners=rfish.targets)



    # prev = cv2.resize(
    #         prev,
    #         (int(img_width(prev)*resize_ratio),int(img_height(prev)*resize_ratio)))

    scambi_units = []
    led_positions = get_led_perimeter_pos(prev, no_leds_vert, no_leds_horiz)
    print("got get_led_perimeter_pos")
    for index,  led in enumerate(led_positions):
            print(f"calculating scambiunit {index}/{no_leds_vert+no_leds_vert+no_leds_horiz+no_leds_horiz}")
            centre_ = tuple((np.asarray(led.positionxy)).astype(int))
            #cv2.circle(prev,plop,16,(255,0,100),-1)
            mid_screen = (np.array(tuple(reversed(prev.shape[:2])))/2).astype(int)[:2]
            vec_to_midscreen = mid_screen-np.asarray(centre_)
            #cv2.circle(prev,tuple(mid_screen),16,(255,0,100),-1)
            if led.edge  not in [Edges.TOP, Edges.LOWER, Edges.LEFT, Edges.RIGHT]:
                raise Exception("edge name " + led.edge + "not valid")
            if led.edge  in [Edges.TOP, Edges.LOWER]:
                new_pos = tuple((np.asarray(centre_) + (vec_to_midscreen * move_in_vert)).astype(int))
            if led.edge  in [Edges.LEFT, Edges.RIGHT]:
                new_pos = tuple((np.asarray(centre_) + (vec_to_midscreen * move_in_horiz)).astype(int))
            left, right, top, lower = create_rectangle_from_centrepoint(new_pos, edge=sample_area_edge)
            init = ScambiInit(led_positionxy=centre_,
                sample_area_left=left,
                sample_area_right=right,
                sample_area_top=top,
                sample_area_lower=lower,
                inverse_warp_m=homography_tool.inverse_trans_matrix,
                img_shape=prev.shape,
                img_circle=rfish.fish_eye_circle,
                edge=led.edge,
                position_normed=led.normed_pos_along_edge_mid,
                position_norm_start=led.normed_pos_along_edge_start,
                position_norm_end=led.normed_pos_along_edge_end,
                id=index)
            scambi_units.append(Scambi_unit(init)
            )

    for scambi in scambi_units:
        scambi.assign_physical_LED_pos(led_subsystem.get_LEDpos_for_edge_range(scambi))






    # prepare for main loop
    random.shuffle(scambi_units)
    scambis_per_core = int(len(scambi_units)/cores_for_col_dect)
    # chop up list of scambiunits for parallel processing
    proc_scambis = [
        async_cam_lib.Process_Scambiunits(
            scambiunits=scambi_units[i:i+scambis_per_core],
            subsample_cutoff=subsample_cut,
            flipflop=False)
        for i
        in range(0,len(scambi_units), scambis_per_core)]

    # get initialised scambiunits from parallel processing
    
    scambi_units = []
    for scamproc in proc_scambis:
        scambi_units.append(scamproc.initialised_scambis_q.get(block=True, timeout=None))
    
    # flatten nested list
    scambi_units = [item for sublist in scambi_units for item in sublist]
    # main loop
    index = 0
    flipflop = False
    sent_overlay = 10
    while True:
        subsampled = 0
        with time_it("main loop"):
            index += 1

            with time_it("get img"):
                prev = next(cam)
            flipflop = not flipflop

            with time_it(f"get {len(scambi_units)} colours"):
                for index, unit in enumerate(scambi_units):
                    # if flipflop is True:
                    #     if index%2 == 0:
                    #         continue
                    # if flipflop is False:
                    #     if index%2 == 1:
                    #         continue

                    unit.get_dom_colour_with_auto_subsample(prev, cut_off = subsample_cut)


            if PLATFORM == _OS.WINDOWS or sent_overlay > -1:
                if sent_overlay > -2:
                    sent_overlay -= 1
                display_img = prev.copy()
                with time_it(f"overlay"):
                    for index, unit in enumerate(scambi_units):
                        #display_img = unit.draw_warped_roi(display_img)
                        
                        unit.draw_warped_boundingbox(display_img)
                        display_img = unit.draw_lerp_contour(display_img)
                        display_img = unit.draw_warped_led_pos(
                            display_img,
                            unit.colour,
                            offset=(0, 0),
                            size=10)
                        

                if sent_overlay == 0:
                    before_warp = display_img.copy()
                    perp_warped = fisheriser.fish_eye_image(display_img.copy(), reverse=True)
                    for pt in homography_tool._corners:
                        perp_warped = cv2.circle(perp_warped, tuple(pt.astype(int)), 20, (255,0,0), -1)
                    display_img = fisheriser.fish_eye_image(display_img, reverse=True)
                    display_img = homography_tool.warp_img(display_img)
                    upload_img_to_aws(
                        np.vstack((before_warp,display_img, perp_warped)),
                        img_upload_url,
                        action = "overlay")

            if PLATFORM == _OS.WINDOWS:
                ImageViewer_Quick_no_resize(display_img,0,False,False)
    
            with time_it(f"subsampled {subsampled}/{len(scambi_units)}"):
                pass

            with time_it("set leds"):
                led_subsystem.set_LED_values(scambi_units)
                led_subsystem.execute_LEDS()




def upload_img_to_aws(img, url, action):
    
    print("uploading image")
    if action == "raw":
        action = "image_raw"
    elif action =="overlay":
        action = "image_overlay"
    else:
        raise Exception("bad action")
    img = clahe_equalisation(img, None)
    img_bytes = encode_img_to_str(img)
    myobj = {
        "authentication": "farts",
        "action": action,
        "payload": img_bytes
        }
    try:
        response = requests.post(url, json=myobj)
        print("Auto uploading image", response.text)
    except requests.exceptions.RequestException as e:
        print(e)
        print("could not connect first image upload to ", url)
    
def get_config_from_aws(url):
    print("getting config from aws")
    myobj = {
        "authentication": "farts",
        "action": "request_config"
        }
    positions = []
    try:
        response = requests.post(url, json=myobj)
        #TODO not good - why is this so arduous - can't be right
        clicked_positions = json.loads(json.loads(response.content)['config'])

        for elem in clicked_positions:
            # sorry
            positions.append({i:int((elem)[i]) for i in elem})
    except (requests.exceptions.RequestException, KeyError) as e:
        print(e)
        print("could not connect get config or find key from", url)

    return positions









def multiprocess_set_colours():
    # This doesnt work with raspberry pi w 2 - maybe if getting one
    # with more cores
    # some ROI are bigger than others
    random.shuffle(scambi_units)
    scambis_per_core = int(len(scambi_units)/cores_for_col_dect)
    # chop up list of scambiunits for parallel processing
    proc_scambis = [
        async_cam_lib.Process_Scambiunits(
            scambiunits=scambi_units[i:i+scambis_per_core],
            subsample_cutoff=subsample_cut,
            flipflop=False)
        for i
        in range(0,len(scambi_units), scambis_per_core)]

    while True:
        with time_it("main loop"):
            with time_it("get img"):
                prev = next(cam)
            with time_it("load imgs"):
                for scamproc in proc_scambis:
                    scamproc.in_queue.put(prev, block=True)
            scambis_cols = {}
            with time_it("wait for colors"):
                for scamproc in proc_scambis:
                    scambis_cols.update(scamproc.done_queue.get(block=True, timeout=None))
            with time_it("rebuild scambi colours"):
                for unit in scambi_units:
                    unit.colour = scambis_cols[unit.id]
            if PLATFORM == _OS.WINDOWS:
                for index, unit in enumerate(scambi_units):
                    unit.draw_warped_boundingbox(prev)
                    prev = unit.draw_warped_led_pos(
                        prev,
                        unit.colour,
                        offset=(0, 0),
                        size=10)
            with time_it("set colours"):
                led_subsystem.set_LED_values(scambi_units)
                led_subsystem.execute_LEDS()
            if PLATFORM == _OS.WINDOWS:
                ImageViewer_Quick_no_resize(prev,0,True,False)



    #index = 0
    #flipflop = False
    # while True:
    #     with time_it("main loop"):
    #         index += 1
    #         prev = next(cam)
    #         for index, unit in enumerate(scambi_units):
    #             unit.get_dominant_colour_flat(prev, subsample=1)
    #         if PLATFORM == _OS.WINDOWS:
    #             for index, unit in enumerate(scambi_units):
    #                 unit.draw_warped_boundingbox(prev)
    #                 prev = unit.draw_warped_led_pos(
    #                     prev,
    #                     unit.colour,
    #                     offset=(0, 0),
    #                     size=10)
    #         led_subsystem.set_LED_values(scambi_units)
    #         led_subsystem.execute_LEDS()
    #         if PLATFORM == _OS.WINDOWS:
    #             ImageViewer_Quick_no_resize(prev,0,False,False)








    # wokring loop
    while True:
        subsampled = 0
        with time_it("main loop"):
            index += 1
            #print(f"frame {index}")
            #prev = fisheriser.fish_eye_image(prev, reverse=True)
            # prev = cv2.resize(
            #     prev,
            #     (int(img_width(prev)*resize_ratio),int(img_height(prev)*resize_ratio)))
            with time_it("get img"):
                prev = next(cam)
            flipflop = not flipflop
            with time_it(f"get {len(scambi_units)} colours"):
                for index, unit in enumerate(scambi_units):
                    if flipflop is True:
                        if index%2 == 0:
                            continue
                    if flipflop is False:
                        if index%2 == 1:
                            continue
                    #unit.draw_led_pos(prev)
                    #unit.draw_rectangle(prev)

                    if (unit.bb_right - unit.bb_left) < subsample_cut or (unit.bb_lower-unit.bb_top) < subsample_cut:
                        unit.get_dominant_colour_flat(prev, subsample=1)
                    else:
                        unit.get_dominant_colour_flat(prev, subsample=2)
                        subsampled += 1

                    if PLATFORM == _OS.WINDOWS:
                        unit.draw_warped_boundingbox(prev)
                        prev = unit.draw_warped_led_pos(
                            prev,
                            unit.colour,
                            offset=(0, 0),
                            size=10)
            with time_it(f"subsampled {subsampled}/{len(scambi_units)}"):
                pass
            with time_it("set leds"):
                led_subsystem.set_LED_values(scambi_units)
                led_subsystem.execute_LEDS()
            #     #prev = unit.draw_warped_roi(prev)
            #     prev = unit.draw_lerp_contour(prev)
            # prev = unit.draw_warped_boundingbox(prev)

            #cv2.imwrite(f"/home/scambilight/0{index}.jpg", prev)
            if PLATFORM == _OS.WINDOWS:
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

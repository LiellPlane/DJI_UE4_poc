import numpy as np
import cv2
import time
from contextlib import contextmanager
import random
from libs.utils import (
    get_platform,
    _OS,
    TimeDiffObject,
    ImageViewer_Quick_no_resize,
    encode_img_to_str,
    img_height,
    img_width,
    time_it)
from libs.scambiunits import (
    Scambi_unit,
    ScambiInit,
    HomographyTool)
from libs.collections import (
    Edges,
    lens_details,
    LensConfigs)
import libs.async_cam_lib as async_cam_lib
import libs.fisheye_lib as fisheye_lib
from libs.lighting import SimLeds, ws281Leds, get_led_perimeter_pos
from libs.configs import (
    DaisybankLedSpacing,
    get_regions_config,
    get_lens_details)
from libs.external_data import (
    upload_img_to_aws,
    get_config_from_aws,
    get_corners_from_remote_config)

PLATFORM = get_platform()
if PLATFORM == _OS.RASPBERRY:
    # sorry not sorry
    import rpi_ws281x as leds


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





def main():

    rfish = get_lens_details(LensConfigs.DAISYBANK_HQ)
    system = get_platform()
    if system == _OS.WINDOWS:
        led_subsystem = SimLeds(DaisybankLedSpacing)
        cam = async_cam_lib.Synth_Camera_Async(async_cam_lib.ScambiLight_Cam_vidmodes)
        cores = 8
    elif system == _OS.RASPBERRY:
        cam = async_cam_lib.Scamblight_Camera_Async(async_cam_lib.ScambiLight_Cam_vidmodes)
        led_subsystem = ws281Leds(DaisybankLedSpacing)
        cores = 3
    else:
        raise Exception(system + " not supported")

    cores_for_col_dect = cores
    img_upload_url = "https://yqnz152azi.execute-api.us-east-1.amazonaws.com/Prod/hello" # for AWS experiment

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


    print(positions)
    homography_tool = HomographyTool(
        img_height_=rfish.height,
        img_width_=rfish.width,
        corners=real_corners,
        target_corners=rfish.targets)

    regions = get_regions_config()


    scambi_units = []
    led_positions = get_led_perimeter_pos(prev, regions.no_leds_vert, regions.no_leds_horiz)
    print("got get_led_perimeter_pos")
    for index,  led in enumerate(led_positions):
            print(f"calculating scambiunit {index}/{regions.no_leds_vert+regions.no_leds_vert+regions.no_leds_horiz+regions.no_leds_horiz}")
            centre_ = tuple((np.asarray(led.positionxy)).astype(int))
            #cv2.circle(prev,plop,16,(255,0,100),-1)
            mid_screen = (np.array(tuple(reversed(prev.shape[:2])))/2).astype(int)[:2]
            vec_to_midscreen = mid_screen-np.asarray(centre_)
            #cv2.circle(prev,tuple(mid_screen),16,(255,0,100),-1)
            if led.edge  not in [Edges.TOP, Edges.LOWER, Edges.LEFT, Edges.RIGHT]:
                raise Exception("edge name " + led.edge + "not valid")
            if led.edge  in [Edges.TOP, Edges.LOWER]:
                new_pos = tuple((np.asarray(centre_) + (vec_to_midscreen * regions.move_in_vert)).astype(int))
            if led.edge  in [Edges.LEFT, Edges.RIGHT]:
                new_pos = tuple((np.asarray(centre_) + (vec_to_midscreen * regions.move_in_horiz)).astype(int))
            left, right, top, lower = create_rectangle_from_centrepoint(new_pos, edge=regions.sample_area_edge)
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
            subsample_cutoff=regions.subsample_cut,
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

                    unit.get_dom_colour_with_auto_subsample(prev, cut_off = regions.subsample_cut)


            if PLATFORM == _OS.WINDOWS or sent_overlay > -1:
                if sent_overlay > -2:
                    sent_overlay -= 1
                display_img = prev.copy()
                with time_it("overlay"):
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


if __name__ == "__main__":
    #test_find_screen()
    main()

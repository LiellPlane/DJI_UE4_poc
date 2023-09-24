import os
import sys
#abs_path = os.path.dirname(os.path.abspath(__file__))
#scambi_path = abs_path + "/DJI_UE4_poc/Source/scambilight"
print( os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import time
import json
from collections import deque

from libs.utils import (
    get_platform,
    _OS,
    ImageViewer_Quick_no_resize,
    time_it_sparse)
from libs.scambiunits import (
    HomographyTool,
    generate_scambis)
from libs.collections import (
    LensConfigs,
    LEDColours)
import libs.async_cam_lib as async_cam_lib
import libs.fisheye_lib as fisheye_lib
from libs.lighting import SimLeds, ws281Leds
from libs.configs import (
    DaisybankLedSpacing,
    get_sample_regions_config,
    get_lens_details,
    ScambiLight_Cam_vidmodes,
    SCAMILIGHT_API)
from libs.external_data import (
    upload_img_to_aws,
    get_config_from_aws,
    get_region_config_from_aws,
    get_ext_corners_or_use_default,
    get_image_from_aws,
    ExternalDataWorker,
    ExternalDataWorker_dummy)
import os
PLATFORM = get_platform()


def get_cam(system: _OS, action: str):
    if action is not None:
        return async_cam_lib.Synth_Camera_sync(
            ScambiLight_Cam_vidmodes)
    if system == _OS.WINDOWS:
        return async_cam_lib.Synth_Camera_Async(
            ScambiLight_Cam_vidmodes)
    elif system == _OS.RASPBERRY:
        return async_cam_lib.Scamblight_Camera_Async(
            ScambiLight_Cam_vidmodes)
    elif system == _OS.LINUX:
        return async_cam_lib.Synth_Camera_Async(
            ScambiLight_Cam_vidmodes)
    else:
        raise Exception(system + " not supported")
    
def get_external_data_workr(action):
    if action is not None:
        return ExternalDataWorker_dummy(SCAMILIGHT_API)
    return ExternalDataWorker(SCAMILIGHT_API)


def main(action = None):
    
    optical_details = get_lens_details(
        LensConfigs.DAISYBANK_HQ)
    fisheye_compute = fisheye_lib.fisheye_tool(
        img_width_height=(optical_details.width, optical_details.height),
        image_circle_size=optical_details.fish_eye_circle)
    system = get_platform()
    cam = get_cam(system=system, action=action)
    if system == _OS.WINDOWS:
        led_subsystem = SimLeds(DaisybankLedSpacing)
        cores = 8
    elif system == _OS.RASPBERRY:
        led_subsystem = ws281Leds(DaisybankLedSpacing)
        cores = 3
    elif system == _OS.LINUX:
        led_subsystem = SimLeds(DaisybankLedSpacing)
        cores = 8
    else:
        raise Exception(system + " not supported")

    led_subsystem.display_info_colours(LEDColours.Red.value)
    cores_for_col_dect = cores

    # for incoming action, don't use external worker 
    ActionChecker = get_external_data_workr(action=action)
    ActionChecker._start()
    #event = check_events_from_aws(SCAMILIGHT_API)
    #print("purging old action requests", event)
    
    curr_img = next(cam)
    # upload image before anything crashes 
 
    aws_config = get_config_from_aws(SCAMILIGHT_API)
    led_subsystem.display_info_colours(LEDColours.Cyan.value)
    fish_img_corners = get_ext_corners_or_use_default(
        ext_click_data=aws_config.fish_eye_clicked_corners,
        default_corners=optical_details.corners,
        imgshape=curr_img.shape)
    led_subsystem.display_info_colours(LEDColours.Magenta.value)
    homography_tool = HomographyTool(
        img_height_=optical_details.height,
        img_width_=optical_details.width,
        corners=fish_img_corners,
        target_corners=optical_details.targets)

    led_subsystem.display_info_colours(LEDColours.Yellow.value)

    if (img_sample_controller := get_region_config_from_aws(SCAMILIGHT_API)) is None:
        print("Could not get sample region data - using default")
        time.sleep(2)
        img_sample_controller = get_sample_regions_config()

    print(f"Requested action: {action}")
    if action == None:
        scambi_units = generate_scambis(
            img_shape=curr_img.shape,
            regions=img_sample_controller,
            optical_details=optical_details,
            homography_tool=homography_tool,
            led_subsystem=led_subsystem,
            initialise=True,
            init_cores=cores_for_col_dect,
            progress_bar_func=led_subsystem.display_info_bar)
    else:
        scambi_units = generate_scambis(
            img_shape=curr_img.shape,
            regions=img_sample_controller,
            optical_details=optical_details,
            homography_tool=homography_tool,
            led_subsystem=led_subsystem,
            initialise=False,
            init_cores=cores_for_col_dect,
            progress_bar_func=led_subsystem.display_info_bar)

        for index, unit in enumerate(scambi_units):
            unit.initialise()


    if action is not None:
        prev = get_image_from_aws(SCAMILIGHT_API)
        for index, unit in enumerate(scambi_units):
            unit.draw_warped_boundingbox(prev)
            prev = unit.draw_lerp_contour(prev)
            unit.get_dom_colour_with_auto_subsample(
                prev, cut_off = img_sample_controller.subsample_cut)
            prev = unit.draw_warped_led_pos(
                prev,
                unit.colour,
                offset=(0, 0),
                size=10)
        upload_img_to_aws(
            prev,
            SCAMILIGHT_API,
            action = "overlay")
        # TODO can we wrap this somewhere nicely like an ATEXIT
        # or similar so it doesnt pollute the main thread?
        return {
            'statusCode': 200,
            'body': json.dumps(f"completed {action}")
        }


    # main loop
    index = 0
    flipflop = False
    #sent_overlay = 10
    while True:
        event = ActionChecker.check_for_action()
        #subsampled = 0
        with time_it_sparse("main loop"):
            index += 1

            with time_it_sparse("get img"):
                prev = next(cam)
                
            flipflop = not flipflop
            
            with time_it_sparse(f"get {len(scambi_units)} colours"):
                for index, unit in enumerate(scambi_units):
                    # if flipflop is True:
                    #     if index%2 == 0:
                    #         continue
                    # if flipflop is False:
                    #     if index%2 == 1:
                    #         continue

                    unit.get_dom_colour_with_auto_subsample(prev, cut_off = img_sample_controller.subsample_cut)


            if PLATFORM == _OS.WINDOWS:
                display_img = prev.copy()
                with time_it_sparse("overlay"):
                    for index, unit in enumerate(scambi_units):
                        #display_img = unit.draw_warped_roi(display_img)
                        
                        unit.draw_warped_boundingbox(display_img)
                        display_img = unit.draw_lerp_contour(display_img)
                        display_img = unit.draw_warped_led_pos(
                            display_img,
                            unit.colour,
                            offset=(0, 0),
                            size=10)
                        

                # if sent_overlay == 0:
                #     led_subsystem.display_info_colours(LEDColours.Blue.value)
                #     before_warp = display_img.copy()
                #     perp_warped = fisheye_compute.fish_eye_image(display_img.copy(), reverse=True)
                #     led_subsystem.display_info_colours(LEDColours.Red.value)
                #     for pt in homography_tool._corners:
                #         perp_warped = cv2.circle(perp_warped, tuple(pt.astype(int)), 20, (255,0,0), -1)
                #     display_img = fisheye_compute.fish_eye_image(display_img, reverse=True)
                #     led_subsystem.display_info_colours(LEDColours.Blue.value)
                #     display_img = homography_tool.warp_img(display_img)
                #     upload_img_to_aws(
                #         np.vstack((before_warp, display_img, perp_warped)),
                #         SCAMILIGHT_API,
                #         action = "overlay")
            if event == "update_image_all":
                led_subsystem.display_info_colours(LEDColours.Magenta.value)
                display_img = prev.copy()
                perp_warped = fisheye_compute.fish_eye_image(display_img.copy(), reverse=True)
                upload_img_to_aws(perp_warped, SCAMILIGHT_API, action = "raw")
                display_img = prev.copy()
                for index, unit in enumerate(scambi_units):
                        #display_img = unit.draw_warped_roi(display_img)
                        
                        unit.draw_warped_boundingbox(display_img)
                        display_img = unit.draw_lerp_contour(display_img)
                        display_img = unit.draw_warped_led_pos(
                            display_img,
                            unit.colour,
                            offset=(0, 0),
                            size=10)              
                led_subsystem.display_info_colours(LEDColours.Blue.value)
                before_warp = display_img.copy()
                perp_warped = fisheye_compute.fish_eye_image(display_img.copy(), reverse=True)
                led_subsystem.display_info_colours(LEDColours.Yellow.value)
                for pt in homography_tool._corners:
                    perp_warped = cv2.circle(perp_warped, tuple(pt.astype(int)), 20, (255,0,0), -1)
                #display_img = fisheye_compute.fish_eye_image(display_img, reverse=True)
                led_subsystem.display_info_colours(LEDColours.Blue.value)
                display_img = homography_tool.warp_img(perp_warped)
                upload_img_to_aws(
                    np.vstack((before_warp, display_img, perp_warped)),
                    SCAMILIGHT_API,
                    action = "overlay")
            if event == "update_image":
                display_img = prev.copy()
                upload_img_to_aws(
                    display_img,
                    SCAMILIGHT_API,
                    action = "raw")
                for index, unit in enumerate(scambi_units):
                    #display_img = unit.draw_warped_roi(display_img)
                    unit.draw_warped_boundingbox(display_img)
                    display_img = unit.draw_lerp_contour(display_img)
                    display_img = unit.draw_warped_led_pos(
                        display_img,
                        unit.colour,
                        offset=(0, 0),
                        size=10)
                upload_img_to_aws(
                    display_img,
                    SCAMILIGHT_API,
                    action = "overlay")
                

            if event == "reset":
                led_subsystem.display_info_colours(LEDColours.Red.value)
                if PLATFORM == _OS.RASPBERRY:
                    os.system("sudo reboot")
                else:
                    raise Exception("reboot requested")
            if PLATFORM == _OS.WINDOWS:
                ImageViewer_Quick_no_resize(display_img,0,False,False)
    
            # with time_it_sparse(f"subsampled {subsampled}/{len(scambi_units)}"):
            #     pass

            with time_it_sparse("set leds"):
                led_subsystem.set_LED_values(scambi_units)
                led_subsystem.execute_LEDS()
            

if __name__ == "__main__":
    main()


def handler(event, context):
    print("boom")
    main(action = "Sim Scambis")

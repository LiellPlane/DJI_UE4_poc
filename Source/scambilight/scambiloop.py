import os
import sys
import copy
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
from math import floor
from datetime import datetime
from libs.utils import (
    get_platform,
    _OS,
    ImageViewer_Quick_no_resize,
    time_it_sparse,
    create_progress_image)
from libs.scambiunits import (
    HomographyTool,
    generate_scambis,
    Scambi_unit_LED_only)
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
    SCAMILIGHT_API,
    UploadImageTypes)
from libs.external_data import (
    upload_img_to_aws,
    get_config_from_aws,
    get_region_config_from_aws,
    get_ext_corners_or_use_default,
    get_image_from_aws,
    get_lens_details_external,
    ExternalDataWorker,
    ExternalDataWorker_dummy,
    cors_headers)
import os
PLATFORM = get_platform()


def get_cam(system: _OS, action: str):
    if action is not None:
        return async_cam_lib.Synth_Camera_sync(
            ScambiLight_Cam_vidmodes)
    if system == _OS.WINDOWS:
        return async_cam_lib.Synth_Camera_Async_buffer(
            ScambiLight_Cam_vidmodes)
    elif system == _OS.RASPBERRY:
        return async_cam_lib.Scamblight_Camera_Async_buffer(
            ScambiLight_Cam_vidmodes)
    elif system == _OS.LINUX:
        return async_cam_lib.Synth_Camera_Async(
            ScambiLight_Cam_vidmodes)
    elif system == _OS.MAC_OS:
        return async_cam_lib.Synth_Camera_Async_buffer(
            ScambiLight_Cam_vidmodes)
    else:
        raise Exception(system + " not supported")
    
def get_external_data_workr(action):
    if action is not None:
        return ExternalDataWorker_dummy(SCAMILIGHT_API)
    return ExternalDataWorker(SCAMILIGHT_API)


def main(action = None):
    optical_details = get_lens_details_external(SCAMILIGHT_API)
    # optical_details = get_lens_details(
    #     LensConfigs.DAISYBANK_LQ)
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
        cores = 2 # tends to crash higher than 2
    elif system == _OS.LINUX:
        led_subsystem = SimLeds(DaisybankLedSpacing)
        cores = 8
    elif system == _OS.MAC_OS:
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
    if action is None:
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

        update_cnter = -1
        for index, unit in enumerate(scambi_units):
            unit.initialise()
            progress = int((index/len(scambi_units)) * 100)
            updaterate = floor(progress/10)
            if updaterate != update_cnter:
                upload_img_to_aws(
                    create_progress_image(progress_percent=progress),
                    SCAMILIGHT_API,
                    action = UploadImageTypes.OVERLAY.value)
            update_cnter = updaterate

    if action is not None:
        prev = get_image_from_aws(SCAMILIGHT_API)
        for index, unit in enumerate(scambi_units):
            unit.draw_warped_boundingbox(prev)
            prev = unit.draw_lerp_contour(prev)
            # grab colour first or averaging gives 
            # a non useful colour
            colour = unit.get_dominant_colour_flat(prev, subsample=1)
            unit.colour = tuple(int(i) for i in colour)
            # now grab colour as its done in the main loop
            unit.set_dom_colour_with_auto_subsample(
                prev, cut_off = img_sample_controller.subsample_cut)
            prev = unit.draw_warped_led_pos(
                prev,
                unit.colour,
                offset=(0, 0),
                size=10)

        upload_img_to_aws(
            prev,
            SCAMILIGHT_API,
            action = UploadImageTypes.OVERLAY.value)
        # TODO can we wrap this somewhere nicely like an ATEXIT
        # or similar so it doesnt pollute the main thread?
        return{
            'statusCode': 201,
            'headers': cors_headers,
            'body': json.dumps({'message': f"completed {action}"})
        }



    # main loop
    index = 0
    #sent_overlay = 10

    # this is being updated constantly by the camera class
    # and luckily we can read frrom it without mem errors


    # things with queues can break the AWS lambda container images
    if action is None:
        proc_scambis = async_cam_lib.RunScambisWithAsyncImage(
            scambiunits=copy.deepcopy(scambi_units[0:len(scambi_units)//2]),
            curr_img=curr_img,
            async_image_buf=cam.shared_mem_handler.mem_ids["0"],
            Scambi_unit_LED_only=Scambi_unit_LED_only,
            subsample_cutoff=img_sample_controller.subsample_cut
        )
        # half scambis to parallel process half to main proces
        scambi_units = scambi_units[len(scambi_units)//2:]




    while True:
        event = ActionChecker.check_for_action()
        #subsampled = 0
        with time_it_sparse("main loop"):
            index += 1

            # with time_it_sparse("get img"):
            #     prev = next(cam)

            # get next image buffer 
            cam.release_next_image()
            prev: np.ndarray = np.ndarray(
                curr_img.shape,
                dtype=curr_img.dtype,
                buffer=cam.get_img_buffer())
            
            if PLATFORM == _OS.WINDOWS or PLATFORM == _OS.MAC_OS:
                display_img = prev.copy()
                time.sleep(0.1)
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

            if event == "update_image_all":
                led_subsystem.display_info_colours(LEDColours.Magenta.value)
                display_img = prev.copy()
                upload_img_to_aws(
                    display_img,
                    SCAMILIGHT_API,
                    action = UploadImageTypes.RAW.value)
                perp_warped = fisheye_compute.fish_eye_image(display_img.copy(), reverse=True)
                upload_img_to_aws(perp_warped, SCAMILIGHT_API, action = UploadImageTypes.PERPWARPED.value)
                display_img = prev.copy()
                for index, unit in enumerate(scambi_units):
                        unit.draw_warped_boundingbox(display_img)
                        display_img = unit.draw_lerp_contour(display_img)
                        display_img = unit.draw_warped_led_pos(
                            display_img,
                            unit.colour,
                            offset=(0, 0),
                            size=10)              
                led_subsystem.display_info_colours(LEDColours.Blue.value)
                before_warp = display_img.copy()
                #perp_warped = fisheye_compute.fish_eye_image(display_img.copy(), reverse=True)
                led_subsystem.display_info_colours(LEDColours.Yellow.value)
                for pt in homography_tool._corners:
                    perp_warped = cv2.circle(perp_warped, tuple(pt.astype(int)), 20, (255,0,0), -1)
                #display_img = fisheye_compute.fish_eye_image(display_img, reverse=True)
                led_subsystem.display_info_colours(LEDColours.Blue.value)
                display_img = homography_tool.warp_img(perp_warped)
                upload_img_to_aws(
                    np.vstack((before_warp, display_img, perp_warped)),
                    SCAMILIGHT_API,
                    action = UploadImageTypes.OVERLAY.value)
            if event == "update_image":
                display_img = prev.copy()
                upload_img_to_aws(
                    display_img,
                    SCAMILIGHT_API,
                    action = UploadImageTypes.RAW.value)
                perp_warped = fisheye_compute.fish_eye_image(display_img.copy(), reverse=True)
                upload_img_to_aws(perp_warped, SCAMILIGHT_API, action = UploadImageTypes.PERPWARPED.value)
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
                upload_img_to_aws(
                    display_img,
                    SCAMILIGHT_API,
                    action = UploadImageTypes.OVERLAY.value)
                
            if event.startswith("ERROR"):
                raise Exception(event)

            if event == "reset":
                led_subsystem.display_info_colours(LEDColours.Red.value)
                if PLATFORM == _OS.RASPBERRY:
                    os.system("sudo reboot")
                else:
                    raise Exception("reboot requested")
            if PLATFORM == _OS.WINDOWS or PLATFORM == _OS.MAC_OS:
                ImageViewer_Quick_no_resize(display_img,0,False,False)
    
            # put this here incase we can grab an image if everyhthing is messed up
            scambiunits_led_info = []
            with time_it_sparse(f"get {len(scambi_units)} colours"):
                for index, unit in enumerate(scambi_units):
                    # if flipflop is True:
                    #     if index%2 == 0:
                    #         continue
                    # if flipflop is False:
                    #     if index%2 == 1:
                    #         continue
                    unit.set_dom_colour_with_auto_subsample(prev, cut_off = img_sample_controller.subsample_cut)
                    scambiunits_led_info.append(Scambi_unit_LED_only(
                        colour=unit.colour,
                        physical_led_pos=unit.physical_led_pos))
                    
                #TODO all these conditions are not good code
                # probably take it all out
                if action is None: #  not running in container (which doesn't like queues)
                    scambiunits_led_info += proc_scambis.done_queue.get(block=True)
                    proc_scambis.handshake_queue.put("done", block=True, timeout=None)

            with time_it_sparse("set leds"):
                led_subsystem.set_LED_values(scambiunits_led_info)
                led_subsystem.execute_LEDS()

if __name__ == "__main__":
    main()


def handler(event, context):
    print("boom")
    main(action = "Sim Scambis")

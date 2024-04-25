import os
import sys
import copy
#abs_path = os.path.dirname(os.path.abspath(__file__))
#scambi_path = abs_path + "/DJI_UE4_poc/Source/scambilight"
#
# print( os.path.dirname(os.path.abspath(__file__)))
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
    time_it_return_details,
    create_progress_image)
from libs.scambiunits import (
    HomographyTool,
    generate_scambis)
from libs.collections import LEDColours, Scambi_unit_LED_only
import libs.async_cam_lib as async_cam_lib
import libs.fisheye_lib as fisheye_lib
from libs.lighting import SimLeds, ws281Leds, RemoteLeds
from libs.configs import (
    DaisybankLedSpacing,
    ScambiLight_Cam_vidmodes,
    SCAMILIGHT_API,
    UploadImageTypes)
from libs.external_data import (
    upload_img_to_aws,
    calculate_which_corner,
    get_image_from_aws,
    get_all_config_external,
    ExternalDataWorker,
    ExternalDataWorker_dummy,
    cors_headers,
    sim_file_system,
    raspberry_file_system)
import os
PLATFORM = get_platform()


def get_cam(system: _OS, action: str):
    if action is not None:
        return async_cam_lib.Synth_Camera_sync(
            ScambiLight_Cam_vidmodes)
    if system == _OS.WINDOWS:
        return async_cam_lib.Synth_Camera_sync(
            ScambiLight_Cam_vidmodes)
    elif system == _OS.RASPBERRY:
        return async_cam_lib.Scambi_Camera_sync(
            ScambiLight_Cam_vidmodes)
    elif system == _OS.LINUX:
        return async_cam_lib.Synth_Camera_sync(
            ScambiLight_Cam_vidmodes)
    elif system == _OS.MAC_OS:
        return async_cam_lib.Synth_Camera_sync(
            ScambiLight_Cam_vidmodes)
    else:
        raise Exception(system + " not supported")


def get_external_data_workr(action, sessiontoken):
    if action is not None:
        return ExternalDataWorker_dummy(
            SCAMILIGHT_API,
            sessiontoken
            )
    return ExternalDataWorker(
        SCAMILIGHT_API,
        sessiontoken
        )


def get_file_system(system: _OS):
    if system == _OS.RASPBERRY:
        return raspberry_file_system()
    else:
        return sim_file_system()

def main_test():
    cam = async_cam_lib.Scambi_Camera_sync(
            ScambiLight_Cam_vidmodes)
    timings = deque(maxlen=100)
    while True:
        
        with time_it_return_details("set leds", timings):
            x = next(cam)
            print(x[1,1,1])
            print(x.shape)
        if len(timings) > timings.maxlen-1:
            print('\n'.join(timings))
            timings.clear()

def main(action = None, sessiontoken = None):
    timings = deque(maxlen=100)
    system = get_platform()
    file_system =get_file_system(system=system)
    if sessiontoken is None:
        sessiontoken = file_system.get_session_token_file

    optical_details = get_all_config_external(SCAMILIGHT_API, sessiontoken)

    fisheye_compute = fisheye_lib.fisheye_tool(
        img_width_height=(optical_details.lens_details.width,optical_details.lens_details.height),
        image_circle_size=optical_details.lens_details.fish_eye_circle)
    
    cam = get_cam(system=system, action=action)


    if system == _OS.WINDOWS:
        led_subsystem = RemoteLeds(DaisybankLedSpacing)#RemoteLeds(DaisybankLedSpacing)
        cores_for_col_dect = 8
    elif system == _OS.RASPBERRY:
        led_subsystem = ws281Leds(DaisybankLedSpacing)#ws281Leds
        cores_for_col_dect = 2 # tends to crash higher than 2
    elif system == _OS.LINUX:
        led_subsystem = RemoteLeds(DaisybankLedSpacing)
        cores_for_col_dect = 8 
    elif system == _OS.MAC_OS:
        led_subsystem = RemoteLeds(DaisybankLedSpacing)
        cores_for_col_dect = 8
    else:
        raise Exception(system + " not supported")

    led_subsystem.display_info_colours(LEDColours.Red.value)


    # for incoming action, don't use external worker 
    ActionChecker = get_external_data_workr(
        action=action,
        sessiontoken=sessiontoken)
    ActionChecker._start()
    #event = check_events_from_aws(SCAMILIGHT_API)
    #print("purging old action requests", event)
    
    curr_img = next(cam)
    # upload image before anything crashes 
 
    led_subsystem.display_info_colours(LEDColours.Cyan.value)

    fish_img_corners = calculate_which_corner(
        ext_click_data=optical_details.clicked_corners.fish_eye_clicked_corners,
        imgshape=curr_img.shape
        )

    led_subsystem.display_info_colours(LEDColours.Magenta.value)

    homography_tool = HomographyTool(
        img_height_=optical_details.lens_details.height,
        img_width_=optical_details.lens_details.width,
        corners=fish_img_corners,
        target_corners=optical_details.lens_details.targets)

    led_subsystem.display_info_colours(LEDColours.Yellow.value)

    img_sample_controller = optical_details.sample_regions

    print(f"Requested action: {action}")
    if action is None:
        scambi_units = generate_scambis(
            img_shape=curr_img.shape,
            regions=img_sample_controller,
            optical_details=optical_details.lens_details,
            homography_tool=homography_tool,
            led_subsystem=led_subsystem,
            initialise=True,
            init_cores=cores_for_col_dect,
            progress_bar_func=led_subsystem.display_info_bar)
    else:
        scambi_units = generate_scambis(
            img_shape=curr_img.shape,
            regions=img_sample_controller,
            optical_details=optical_details.lens_details,
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
                    action = UploadImageTypes.OVERLAY.value,
                    sessiontoken=sessiontoken)
            update_cnter = updaterate

    if action is not None:
        prev = get_image_from_aws(SCAMILIGHT_API, sessiontoken)
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
            action = UploadImageTypes.OVERLAY.value,
            sessiontoken=sessiontoken)
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
        # for the raspberry pi we want optimal parallism
        if PLATFORM == _OS.RASPBERRY:
            proc_scambis = async_cam_lib.RunScambisWithAsyncImage(
                scambiunits=copy.deepcopy(scambi_units[0:len(scambi_units)//2]),
                curr_img=curr_img,
                async_image_buf=cam.get_mem_buffers()[0],
                Scambi_unit_LED_only=Scambi_unit_LED_only,
                subsample_cutoff=img_sample_controller.subsample_cut
            )
            # half scambis to parallel process half to main proces
            scambi_units = scambi_units[len(scambi_units)//2:]
        else:# if we are running locally we want to see all the regions - but at least make sure
            # parallel processing is working, so give the parallel process 1 region to process
            proc_scambis = async_cam_lib.RunScambisWithAsyncImage(
                scambiunits=copy.deepcopy(scambi_units[0:1]),
                curr_img=curr_img,
                async_image_buf=cam.get_mem_buffers()[0],
                Scambi_unit_LED_only=Scambi_unit_LED_only,
                subsample_cutoff=img_sample_controller.subsample_cut
            )
            # half scambis to parallel process half to main proces
            scambi_units = scambi_units[1:]



    while True:
        event = ActionChecker.check_for_action()
        #subsampled = 0
        with time_it_return_details("TOTAL", timings):
            index += 1

            # with time_it_sparse("get img"):
            #     prev = next(cam)

            # get next image buffer
            with time_it_return_details("get img", timings):
                prev = next(cam)
                # prev: np.ndarray = np.ndarray(
                #     curr_img.shape,
                #     dtype=curr_img.dtype,
                #     buffer=cam.get_img_buffer())
            
            if PLATFORM == _OS.WINDOWS or PLATFORM == _OS.MAC_OS:
                display_img = prev.copy()
                time.sleep(0.1)
                with time_it_return_details("overlay", timings):
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
                    action = UploadImageTypes.RAW.value,sessiontoken=sessiontoken)
                perp_warped = fisheye_compute.fish_eye_image(display_img.copy(), reverse=True)
                upload_img_to_aws(perp_warped, SCAMILIGHT_API, action = UploadImageTypes.PERPWARPED.value,sessiontoken=sessiontoken)
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
                    action = UploadImageTypes.OVERLAY.value,
                    sessiontoken=sessiontoken)
            if event == "update_image":
                display_img = prev.copy()
                upload_img_to_aws(
                    display_img,
                    SCAMILIGHT_API,
                    action = UploadImageTypes.RAW.value,
                    sessiontoken=sessiontoken)
                perp_warped = fisheye_compute.fish_eye_image(display_img.copy(), reverse=True)
                upload_img_to_aws(perp_warped, SCAMILIGHT_API, action = UploadImageTypes.PERPWARPED.value, sessiontoken=sessiontoken)
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
                    action = UploadImageTypes.OVERLAY.value,
                    sessiontoken=sessiontoken)

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
            with time_it_return_details(f"get {len(scambi_units)} colours", timings):
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
                    
                # TODO all these conditions are not good code
                # probably take it all out
                if action is None: #  not running in container (which doesn't like queues)
                    scambiunits_led_info += proc_scambis.done_queue.get(block=True)
                    proc_scambis.handshake_queue.put("done", block=True, timeout=None)

            with time_it_return_details("set leds", timings):
                led_subsystem.set_LED_values_alternating(scambiunits_led_info)
            with time_it_return_details("execute leds", timings):
                led_subsystem.execute_LEDS()

            if len(timings) > timings.maxlen-1:
                print('\n'.join(timings))
                timings.clear()

def handler(event, context):
    print("boom")
    print(event)
    event_body = json.loads(event['body'])
    print(event_body)
    main(
        action=event_body['action'],
        sessiontoken=event_body['sessiontoken']
    )

if __name__ == "__main__":

    main()
    #main_test()
    
    # body = json.dumps({
    #     'sessiontoken': "admin",
    #     'action': "do_something"
    #     })
    # event = {"body": body}
    # handler(event, None)





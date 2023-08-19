import numpy as np
import cv2
from libs.utils import (
    get_platform,
    _OS,
    ImageViewer_Quick_no_resize,
    time_it)
from libs.scambiunits import (
    HomographyTool,
    generate_scambis)
from libs.collections import (
    LensConfigs)
import libs.async_cam_lib as async_cam_lib
import libs.fisheye_lib as fisheye_lib
from libs.lighting import SimLeds, ws281Leds
from libs.configs import (
    DaisybankLedSpacing,
    get_regions_config,
    get_lens_details)
from libs.external_data import (
    upload_img_to_aws,
    get_config_from_aws,
    get_corners_from_remote_config,
    get_ext_corners_or_use_default)

PLATFORM = get_platform()
if PLATFORM == _OS.RASPBERRY:
    # sorry not sorry
    import rpi_ws281x as leds

def main():

    lens_details = get_lens_details(LensConfigs.DAISYBANK_HQ)
    fisheriser = fisheye_lib.fisheye_tool(
        img_width_height=(lens_details.width, lens_details.height),
        image_circle_size=lens_details.fish_eye_circle)
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

    curr_img = next(cam)
    # upload image before anything crashes 

    img_2_upload = fisheriser.fish_eye_image(next(cam), reverse=True)
    upload_img_to_aws(img_2_upload, img_upload_url, action = "raw")

    aws_config = get_config_from_aws(img_upload_url)

    fish_img_corners = get_ext_corners_or_use_default(
        ext_click_data=aws_config.fish_eye_clicked_corners,
        default_corners=lens_details.corners,
        imgshape=curr_img.shape)

    homography_tool = HomographyTool(
        img_height_=lens_details.height,
        img_width_=lens_details.width,
        corners=fish_img_corners,
        target_corners=lens_details.targets)

    regions = get_regions_config()

    scambi_units = generate_scambis(
        img_shape=curr_img.shape,
        regions=regions,
        lens_details=lens_details,
        homography_tool=homography_tool,
        led_subsystem=led_subsystem,
        initialise=True,
        init_cores=cores_for_col_dect)


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

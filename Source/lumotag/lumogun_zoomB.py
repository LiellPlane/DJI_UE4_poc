# need to add switch for reversed camera in shared memory
# its a bad technique but will have to make do

import factory
import random
from functools import partial
import msgs
import time
#import decode_clothID_v2 as decode_clothID
import analyse_lumotag
import img_processing
from utils import time_it, get_platform
import img_processing
from my_collections import _OS
# need this import to detect lumogun types (subclasses)
import configs
#  detect what OS we are on - test environment (Windows) or production (pi hardware)
PLATFORM = get_platform()
PRINT_DEBUG = configs.get_lumofind_config(PLATFORM).PRINT_DEBUG

if PLATFORM == [_OS.WINDOWS]:
    print("raspberry presence failed, loading test libraries")
    import fake_raspberry_hardware as lumogun
    import sound as sound
elif PLATFORM == _OS.RASPBERRY:
    print("raspberry presence detected, loading hardware libraries")
    import raspberry5_hardware as lumogun
    import sound as sound
elif PLATFORM == _OS.MAC_OS:
    print("disgusting Mac detected, loading fake hardware libraries")
    import fake_raspberry_hardware as lumogun
    import sound_fake as sound
else:
    import fake_raspberry_hardware as lumogun
    import sound_fake as sound

    #raise Exception("Could not detect platform")

# load config depending on if simulated, or if on hardware,
# model ID from file on device
model = lumogun.get_my_info(factory.gun_config.DETAILS_FILE)
GUN_CONFIGURATION  = factory.get_config(model)
def ping_pong_manual(start, end):
    current = start
    step = 1
    while True:
        yield current
        if current == end:
            step = -1
        elif current == start:
            step = 1
        current += step

# import pickle
# import os
# script_path = os.path.abspath(__file__)
# parent_dir = os.path.dirname(script_path)
# pickle_file_path = os.path.join(parent_dir, lumogun.get)
# with open(pickle_file_path, 'rb') as f:
#     perp_details = pickle.load(f)


import numpy as np
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
def main():
    file_system = lumogun.filesystem()
    LR_to_CR_warp_matrix = file_system.get_closerange_to_longrange_transform()
    image_capture = lumogun.CSI_Camera_async_flipflop(GUN_CONFIGURATION.video_modes)
    image_capture_closerange = lumogun.CSI_Camera_async_flipflop(GUN_CONFIGURATION.video_modes_closerange)
    display = lumogun.display(GUN_CONFIGURATION)

    longrangedetails = img_processing.CamDisplayTransform(
        cam_image_shape= next(image_capture).shape
    )
    closerangedetails = img_processing.CamDisplayTransform(
        cam_image_shape= next(image_capture_closerange).shape
    )
    transform_details = img_processing.TransformsDetails(
        longrange_to_shortrange_perwarp=file_system.get_closerange_to_longrange_transform(),
        closerange_to_display=closerangedetails,
        longrange_to_display=longrangedetails,
        transition_steps=25,
        display_image_shape=GUN_CONFIGURATION.screen_size,
        displayrotation=GUN_CONFIGURATION.screen_rotation
    )

    transform_manager = img_processing.TransformManager(transformdetails=transform_details)


    while True:
        # generate lerp from target positions to corners (zoom in?)
        
        # create new transform from original target positions to new lerped ones

        # execute transform for close range

        # execute transform to long range into these new coordinates, so will need to do it twice

        cap_img = next(image_capture)
        cap_img_closerange = next(image_capture_closerange)



        iterator = ping_pong_manual(0, transform_manager.transformdetails.transition_steps-1)
        for _ in range(100000):
            i = next(iterator)
            cap_img = next(image_capture)
            cap_img_closerange = next(image_capture_closerange)

            with time_it("execute affine transform", debug=PRINT_DEBUG):

                mat = transform_manager.CR_all_transition_m[i]
                cr_img = img_processing.apply_perp_transform(mat, cap_img_closerange, display.emptyscreen)

                mat = transform_manager.LR_all_transition_m[i]
                lr_img = img_processing.apply_perp_transform(mat, cap_img, display.emptyscreen)

                percent_done = i/(transform_manager.transformdetails.transition_steps-1)
                cr_img = img_processing.darken_image(cr_img, 1-percent_done)
                combo_image = img_processing.overlay_warped_image_alpha_feathered(cr_img, lr_img, percent_done)
                
                combo_image = img_processing.radial_motion_blur(combo_image)
                combo_image = img_processing.gray2rgb(combo_image)
            with time_it("add graphics: crosshair/analyics", debug=PRINT_DEBUG):
                display.add_crosshair_and_analytics_graphics(combo_image, [])
            with time_it("display image", debug=PRINT_DEBUG):
                display.display_method(combo_image)

            

if __name__ == '__main__':
    main()
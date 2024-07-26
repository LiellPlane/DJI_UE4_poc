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

import pickle
import os
script_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(script_path)
pickle_file_path = os.path.join(parent_dir, 'warp_transform.pkl')
with open(pickle_file_path, 'rb') as f:
    perp_details = pickle.load(f)


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
    image_capture = lumogun.CSI_Camera_async_flipflop(GUN_CONFIGURATION.video_modes)
    image_capture_closerange = lumogun.CSI_Camera_async_flipflop(GUN_CONFIGURATION.video_modes_closerange)
    display = lumogun.display(GUN_CONFIGURATION)

    while True:
        # generate lerp from target positions to corners (zoom in?)
        
        # create new transform from original target positions to new lerped ones

        # execute transform for close range

        # execute transform to long range into these new coordinates, so will need to do it twice

        cap_img = next(image_capture)
        cap_img_closerange = next(image_capture_closerange)


        long_range_ppints = np.asarray([(0, cap_img.shape[0]-1), (0,0), (cap_img.shape[1]-1, 0), (cap_img.shape[1]-1, cap_img.shape[0]-1)])
        homogeneous_points = np.column_stack((long_range_ppints, np.ones(len(long_range_ppints))))
        transformed_points = np.dot(homogeneous_points, perp_details["warpmatrix"].T)
        transformed_points_2d = transformed_points[:, :2] / transformed_points[:, 2:]
        original_form = np.round(transformed_points_2d).astype(int)


        interpolated_points = [
             interpolate_points_eased(start, end, 25)
             for start, end
             in zip(
                original_form,
                np.asarray([(0, cap_img_closerange.shape[0]-1), (0,0), (cap_img_closerange.shape[1]-1, 0), (cap_img_closerange.shape[1]-1, cap_img_closerange.shape[0]-1)])
             )
         ]
        interpolated_points = np.array(interpolated_points)


        iterator = ping_pong_manual(0, interpolated_points.shape[1]-1)
        for _ in range(100000):
            i = next(iterator)
            cap_img = next(image_capture)
            cap_img_closerange = next(image_capture_closerange)
            # this gets the transformation to slowly stretch the long range pov to full screen dims
            # watch out here - as the two cameras have different dims!
            img, mat = img_processing.compute_and_apply_perpwarp(cap_img_closerange, cap_img_closerange,original_form, interpolated_points[:, i])
            # combine the matrices - so we don't have to double up on warps
            # this is the warp which squahes the long range into the centre of the close rnage, then the
            # transition matrix above which unwarps the 
            combined_mat = np.matmul(mat, perp_details["warpmatrix"])
            wraped_img = img_processing.apply_perp_transform(combined_mat,cap_img,cap_img_closerange)
            combo_image = img_processing.overlay_warped_image(img, wraped_img)
            #display.display_method(combo_image)
            #time.sleep(0.001)
        # plop = img_processing.apply_perp_transform(perp_details["warpmatrix"],cap_img,cap_img_closerange)
        # #cap_img_closerange[:,:] = 255
        # for i in range(0,interpolated_points.shape[0]):
        #     plop[list(interpolated_points[i][0].astype(int))[1], list(interpolated_points[i][0].astype(int))[0]] =255
        #     #plop[list(interpolated_points[i][1].astype(int))[1], list(interpolated_points[i][1].astype(int))[0]] =255
        #     #plop[list(interpolated_points[i][2].astype(int))[1], list(interpolated_points[i][2].astype(int))[0]] =255
        #     #plop[list(interpolated_points[i][3].astype(int))[1], list(interpolated_points[i][3].astype(int))[0]] =255

        
        #wraped_img = img_processing.apply_perp_transform(perp_details["warpmatrix"],cap_img,cap_img_closerange)
            #combo_image = img_processing.overlay_warped_image(cap_img_closerange, wraped_img)
            with time_it("execute affine transform", debug=PRINT_DEBUG):
                combo_image = display.generate_output_affine(combo_image)

            with time_it("display image", debug=PRINT_DEBUG):
                display.display_method(combo_image)


if __name__ == '__main__':
    main()
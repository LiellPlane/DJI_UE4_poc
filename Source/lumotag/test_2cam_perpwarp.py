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

import pickle
import os
script_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(script_path)
pickle_file_path = os.path.join(parent_dir, 'warp_transform.pkl')
with open(pickle_file_path, 'rb') as f:
    matrix_perp_transform = pickle.load(f)

def main():
    image_capture = lumogun.CSI_Camera_async_flipflop(GUN_CONFIGURATION.video_modes)
    image_capture_closerange = lumogun.CSI_Camera_async_flipflop(GUN_CONFIGURATION.video_modes_closerange)
    display = lumogun.display(GUN_CONFIGURATION)

    while True:
        cap_img = next(image_capture)
        cap_img_closerange = next(image_capture_closerange)
        wraped_img = img_processing.apply_perp_transform(matrix_perp_transform,cap_img,cap_img_closerange)
        combo_image = img_processing.overlay_warped_image(cap_img_closerange, wraped_img)
        with time_it("execute affine transform", debug=PRINT_DEBUG):
            img = display.generate_output_affine(combo_image)

        with time_it("display image", debug=PRINT_DEBUG):
            display.display_method(img)


if __name__ == '__main__':
    main()
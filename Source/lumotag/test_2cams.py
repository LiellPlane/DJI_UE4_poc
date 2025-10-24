# need to add switch for reversed camera in shared memory
# its a bad technique but will have to make do

import factory
from utils import time_it, get_platform

#  detect what OS we are on - test environment (Windows) or production (pi hardware)
PLATFORM = get_platform()

import raspberry5_hardware as lumogun


# load config depending on if simulated, or if on hardware,
# model ID from file on device
model = lumogun.get_my_info(factory.gun_config.DETAILS_FILE)
GUN_CONFIGURATION  = factory.get_config(model)


def main():
    image_capture = lumogun.CSI_Camera_Synchro(GUN_CONFIGURATION.video_modes)
    image_capture2 = lumogun.CSI_Camera_Synchro(GUN_CONFIGURATION.video_modes_closerange)
    while True:
        cap_img = next(image_capture)
        print(cap_img[1,1])
        cap_img2 = next(image_capture2)
        print(cap_img2[1,1])
if __name__ == '__main__':
    main()

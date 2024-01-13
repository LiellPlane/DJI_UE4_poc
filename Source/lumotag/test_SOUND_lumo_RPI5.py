# need to add switch for reversed camera in shared memory
# its a bad technique but will have to make do

import factory
from utils import time_it, get_platform

#  detect what OS we are on - test environment (Windows) or production (pi hardware)
PLATFORM = get_platform()

import raspberry5_hardware as lumogun

import sound
import time
# load config depending on if simulated, or if on hardware,
# model ID from file on device
model = lumogun.get_my_info(factory.gun_config.DETAILS_FILE)
GUN_CONFIGURATION  = factory.get_config(model)


def main():
    voice = sound.Voice()
    while True:
        
        voice.speak("fart time")
        print("speaking")
        time.sleep(4)

if __name__ == '__main__':
    main()

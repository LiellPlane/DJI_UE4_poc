import time
import re
from subprocess import Popen, PIPE
from os import kill
from signal import SIGKILL
import cv2
import numpy as np
import enum
import functools
# for finding pinout, type pinout in terminal
import RPi.GPIO as GPIO
import time
import factory
import rabbit_mq
from picamera2 import Picamera2
#accelerometer
# adafruit board library forces BCM mode!!
import board
import digitalio
import busio
import adafruit_lis3dh
import json
import img_processing
#import imutils

GPI_MODE_SET = False

def set_GPIO_mode(is_set):
    try:
        if is_set is False:
            GPIO.setmode(GPIO.BCM)
            print("setting GPIO MODE", "GPIO.setmode(GPIO.BCM)")
            is_set = True
    except Exception as e:
        print(e)
        print("attempting to continue - accelerometer may have taken precedence")


def lumo_viewer(
        inputimage,
        pausetime_Secs=0,
        presskey=False,
        destroyWindow=True):
    try:
        cv2.imshow("img", inputimage)
        cv2.moveWindow("img", 600, 0)
        if presskey==True:
            cv2.waitKey(0); #any key
    
        if presskey==False:
            if cv2.waitKey(20) & 0xFF == 27:
                    pass
        if pausetime_Secs>0:
            time.sleep(pausetime_Secs)
        if destroyWindow==True: cv2.destroyAllWindows()

    except Exception as e:
        print(e)


class Accelerometer(factory.Accelerometer):
    def __init__(self) -> None:
        super().__init__()
        self._disp_val_lim_max = 9
        self._disp_val_lim_min = -9
        #using l2c not spi!!
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.int1 = digitalio.DigitalInOut(board.D24)
        self.lis3dh = adafruit_lis3dh.LIS3DH_I2C(
            self.i2c,
            int1=self.int1)

    def update_vel(self):
        x, y, z = self.lis3dh.acceleration
        self._last_xyz = (x, y, z)
        # reverse polarity is to match with
        # LT display - not good place to have it
        return (
            self.round(x*-1),
            self.round(y*-1),
            self.round(z*-1))

    def get_visual(self):
        return super().get_visual()


class display(factory.display):

    def display_output(self, output):
        # quicker in theory to resize first then rotate as
        # input image is expected to be much larger than display size
        if self.display_rotate == 90:
            output = cv2.resize(output, tuple(reversed(self.screen_size)))
            output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
        elif self.display_rotate == -90 or self.display_rotate == 270:
            output = cv2.resize(output, tuple(reversed(self.screen_size)))
            output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.display_rotate == 180:
            output = cv2.resize(output, self.screen_size)
            output = cv2.rotate(output, cv2.ROTATE_180)
        elif self.display_rotate == 0:
            output = img_processing.resize_centre_img(
               output,
               self.screen_size)
            output = img_processing.add_cross_hair(output, adapt=True)
            #output = cv2.resize(output, self.screen_size)
        else:
            raise Exception("incorrect display rotate value", self.display_rotate)


        #output = cv2.normalize(output, output,0, 255, cv2.NORM_MINMAX)

        #output = cv2.applyColorMap(output, cv2.COLORMAP_JET)

        lumo_viewer(output,0,False,False)


class Triggers(factory.Triggers):

    def __init__(self, _gun_config) -> None:
        super().__init__(_gun_config)
        set_GPIO_mode(GPI_MODE_SET)
        for trig, gpio in self.gun_config.TRIGGER_IO.items():
            GPIO.setup(gpio, GPIO.IN)
            print(f"GPIO {gpio} set for trig {trig}")

    def test_states(self):
        outputs = {pos:gpio for pos, gpio
                   in self.gun_config.TRIGGER_IO.items()}
        for _, (pos, gpio) in enumerate(
            self.gun_config.TRIGGER_IO.items()):
            if GPIO.input(gpio) == GPIO.LOW:
                outputs[pos] = True
            else:
                outputs[pos] = False
        return outputs


class Relay(factory.Relay):

    def __init__(self, _gun_config) -> None:
        super().__init__(_gun_config)
        set_GPIO_mode(GPI_MODE_SET)
        for relay, gpio in self.gun_config.RELAY_IO.items():
            GPIO.setup(gpio, GPIO.OUT)
            self.debouncers[relay] = factory.Debounce()
            self.debouncers_1shot[relay] = factory.Debounce()
            print(f"GPIO {gpio} set for relay {relay}")

    def set_relay(
            self,
            relaypos: int,
            state: bool,
            strobe_cnt: int):

        # sometimes we need to strobe the relays for special
        # hardware - for instance IR light that has 3 modes
        debouncer = self.debouncers[relaypos]

        #  functions as variables to make it a bit easier to read
        debounce_on = functools.partial(
                    debouncer.trigger,
                    GPIO.output,
                    self.gun_config.RELAY_IO[relaypos],
                    GPIO.HIGH)
        debounce_off = functools.partial(
                    debouncer.trigger,
                    GPIO.output,
                    self.gun_config.RELAY_IO[relaypos],
                    GPIO.LOW)

        if (strobe_cnt == 0) or (state is False):
            if state:
                return debounce_on()
            else:
                return debounce_off()

        if strobe_cnt == 0 or state is False:
            raise Exception("Bad logic to relay strobe")

        # different logic for strobing
        strobe_state = True

        for _ in range ((strobe_cnt * 2) - 1):
            print(debouncer.debouncetime_sec)
            while not debouncer.can_trigger():
                time.sleep(0.005)
            if strobe_state:
                debounce_on()
            else:
                debounce_off()
            strobe_state = not strobe_state

        if strobe_state is True:
            raise Exception("should always end here high!")
        return True


class CSI_Camera(factory.Camera):

    def __init__(self, video_modes) -> None:
        super().__init__()
        self.cam_res = video_modes
        self.picam2 = Picamera2()
        _config = self.picam2.create_video_configuration(
            main={"size": self.get_res(),  "format": "YUV420"})#, controls={"FrameDurationLimits": (233333, 233333)})
        #self.picam2.set_controls({"ExposureTime": 1000}) # for blurring - but can get over exposed at night
        self.picam2.configure(_config)
        self.picam2.start()
        time.sleep(0.1)
    
    def get_res(self):
        return [e.value for e in self.cam_res][self.res_select][1]

    def _gen_image(self):
        output = self.picam2.capture_array("main")
        (x, y) = self.get_res()#  Need to do this for YUV!
        output = output[0:y, 0:x]#  Need to do this for YUV!
        return output

    def gen_image(self):
        return self._gen_image()

    def __del__(self):
        # this doesn't seem to end cleanly
        self.picam2.stop()


class KillProcess(factory.KillProcess):
    def clean_up_processes(self, cmds, rec_depth=0):
        rec_depth += 1
        if rec_depth > 10:
            raise RecursionError(
                "cannot clean up previous session streaming processes")
        process = Popen(['ps', '-eo', 'pid,args'], stdout=PIPE, stderr=PIPE)
        stdout, _ = process.communicate()
        for line in stdout.splitlines():
            match_list = re.findall(cmds, str(line))
            if len(match_list) > 0:
                print(f"PROCESS {str(line)}")
                pid = int(str(line).split()[1])
                print(f"PID {pid}")
                kill(pid, SIGKILL)
                time.sleep(1)
                self.clean_up_processes(cmds, rec_depth)
                break


Messenger = rabbit_mq.Messenger


def get_my_info(file):
    with open(file, 'r') as file:
        data =  json.load(file)
        MY_ID = data["MY_ID"]

    return MY_ID
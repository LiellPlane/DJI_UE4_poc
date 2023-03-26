import time
import re
from subprocess import Popen, PIPE
from os import kill
from signal import SIGKILL
import cv2
import numpy as np
import enum
# for finding pinout, type pinout in terminal
import RPi.GPIO as GPIO
import time
import factory
from picamera2 import Picamera2
#accelerometer
# adafruit board library forces BCM mode!!
import board
import digitalio
import busio
import adafruit_lis3dh

GPI_MODE_SET = False


class HQ_Cam_vidmodes(enum.Enum):
    _4 = ["640 × 480",(640, 480)] #0.3MP
    _2 = ["2028 × 1080p50,",(2020, 1080)] # 2.0MP  this is not losing res -  turn camera 90 degrees - probably want this one
    _3 = ["1332 × 990p120",(1332, 990)] 
    _1 = ["2028 × 1520p40",(2020, 1520)]


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
        cv2.imshow("img", inputimage.copy()); 
        cv2.moveWindow("img", 0, 0)
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
        output = cv2.resize(output,factory.screensizes.pi_4.value)
        #output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
        output = cv2.normalize(output, output,0, 255, cv2.NORM_MINMAX)
        lumo_viewer(output,0,False,False)


class Triggers(factory.Triggers):

    def __init__(self) -> None:
        super().__init__()
        set_GPIO_mode(GPI_MODE_SET)
        for trig, gpio in factory.config.TRIGGER_IO.items():
            GPIO.setup(gpio, GPIO.IN)
            print(f"GPIO {gpio} set for trig {trig}")

    def test_states(self):
        outputs = {pos:gpio for pos, gpio
                   in factory.config.TRIGGER_IO.items()}
        for _, (pos, gpio) in enumerate(
            factory.config.TRIGGER_IO.items()):
            if GPIO.input(gpio) == GPIO.LOW:
                outputs[pos] = True
            else:
                outputs[pos] = False
        return outputs


class config(factory.config):
    env_name = "integration"

    def loop_wait(self):
        pass


class Relay(factory.Relay):

    def __init__(self) -> None:
        super().__init__()
        set_GPIO_mode(GPI_MODE_SET)
        for relay, gpio in factory.config.RELAY_IO.items():
            GPIO.setup(gpio, GPIO.OUT)
            self.debouncers[relay] = factory.Debounce()
            print(f"GPIO {gpio} set for relay {relay}")

    def set_relay(self, relaypos:int, state:bool):
        if state:
            self.debouncers[relaypos].trigger(
                GPIO.output,
                factory.config.RELAY_IO[relaypos],
                GPIO.HIGH)
        else:
            self.debouncers[relaypos].trigger(
                GPIO.output,
                factory.config.RELAY_IO[relaypos],
                GPIO.LOW)


class CSI_Camera(factory.Camera):

    angle_vs_world_up = 90

    def __init__(self) -> None:
        super().__init__()
        self.picam2 = Picamera2()
        _config = self.picam2.create_video_configuration(
            main={"size": self.get_res(),  "format": "YUV420"})#, controls={"FrameDurationLimits": (233333, 233333)})
        self.picam2.set_controls({"ExposureTime": 10000}) # for blurring - but can get over exposed at night
        self.picam2.configure(_config)
        self.picam2.start()
        time.sleep(0.1)
    
    def get_res(self):
        return [e.value for e in HQ_Cam_vidmodes][self.res_select][1]

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
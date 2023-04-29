import time
import random
import cv2
import numpy as np
import time
import decode_clothID_v1 as decode_clothID
import factory
import math
import msgs

def lumo_viewer(
        inputimage,
        pausetime_Secs=0,
        presskey=False,
        destroyWindow=True):
    try:
        cv2.imshow("img", inputimage.copy())
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


class Triggers(factory.Triggers):
    def __init__(self, _gun_config) -> None:
        super().__init__(_gun_config)
        self.blink_timer = factory.TimeDiffObject()
        self.flipflop = False
    def test_states(self):
        if self.blink_timer.get_dt() > 0.2:
            self.flipflop = not self.flipflop
            self.blink_timer.reset()
        outputs = {pos:gpio for pos, gpio
                   in self.gun_config.TRIGGER_IO.items()}
        for _, (pos, _) in enumerate(
            self.gun_config.TRIGGER_IO.items()):
            outputs[pos] = self.flipflop
        return outputs


class Relay(factory.Relay):
    
    def __init__(self, _gun_config) -> None:
        super().__init__(_gun_config)
        self.relay_mem = {}
        for relay, gpio in self.gun_config.RELAY_IO.items():
            self.debouncers[self.gun_config.RELAY_IO[relay]] = factory.Debounce()
            self.relay_mem[self.gun_config.RELAY_IO[relay]] = False
            print(f"GPIO {gpio} set for relay {relay}")

    def set_relay(self, relaypos:int, state:bool):
        return self.debouncers[self.gun_config.RELAY_IO[relaypos]].trigger(
            self._set_fake_relay,
            self.gun_config.RELAY_IO[relaypos],
            state)
            
    def _set_fake_relay(self, relay, state):
        if relay not in self.relay_mem:
            raise Exception("relay position does not exist!", relay)
        self.relay_mem[relay] = state


class CSI_Camera(factory.Camera):

    def get_res(self):
        pass

    def gen_image(self):
        blank_image = np.zeros((500, 500, 3), np.uint8)
        blank_image[:,:,:] = random.randint(0,255)
        blank_image = cv2.circle(
            blank_image,
            (250, 250),
            40,
            50,
            5)
        return blank_image


class display(factory.display):
    def display_output(self, output):
        output = cv2.resize(output,factory.screensizes.windows_laptop.value)
        #  simulate rotation of lumotag 
        #output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #TODO this might be slow
        lumo_viewer(output, 0, False, False)


class KillProcess(factory.KillProcess):
    def clean_up_processes(self, cmds, rec_depth=0):
        pass


class Accelerometer(factory.Accelerometer):
    def __init__(self) -> None:
        super().__init__()
        self._x = 1
        self._y = -1
        self._z = 0
        self._callcnt = 0
        self._last_xyz = (0, 0, 0)

    def update_vel(self):
        self._callcnt += 1
        self._x += 0.1
        self._y += 0.1
        self._z += 0.1
        if self._x > 9999999:
            self._x = 0
        if self._y > 9999999:
            self._y = 0
        if self._z > 9999999:
            self._z = 0
        real_accel_range = 30
        self._last_xyz = (
            self.round(math.sin(self._x)*real_accel_range),
            self.round(math.sin(self._y)*real_accel_range),
            self.round(math.sin(self._z)*real_accel_range))
        return (
            self.round(math.sin(self._x)*real_accel_range),
            self.round(math.sin(self._y)*real_accel_range),
            self.round(math.sin(self._z)*real_accel_range))
    

class messenger(factory.messenger):

    def __init__(self, config) -> None:
        super().__init__(config=config)
    
    def _in_box_worker(self, in_box, config, scheduler):
        cnt = 0
        while True:
            cnt += 1
            time.sleep(4)
            if in_box._qsize() >= in_box.maxsize:
                print("can't push on any more test messages")
                continue
            in_box.put(
                msgs.create_test_msg(),
                block=False)

    def _out_box_worker(self, out_box, config, scheduler):
        while True:
            message = out_box.get(block=True)
            print("sending into void", message)
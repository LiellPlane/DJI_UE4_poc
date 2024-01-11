# for finding pinout, type pinout in terminal
import RPi.GPIO as GPIO
import time
import factory
import functools
import os
from rasp_hardware_common import *
import utils

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
        # GPIO.setup(port_or_pin, GPIO.OUT)
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

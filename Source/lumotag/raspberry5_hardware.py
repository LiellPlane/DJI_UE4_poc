# for finding pinout, type pinout in terminal
import gpiozero
import time
import factory
import functools
from rasp_hardware_common import *


class Triggers(factory.Triggers):

    def __init__(self, _gun_config) -> None:
        super().__init__(_gun_config)
        self.IO = {}
        for trig, gpio in self.gun_config.TRIGGER_IO.items():
            self.IO[trig] = gpiozero.Button(f"BCM{gpio}")
            print(f"GPIO {gpio} set for trig {trig}")

    def test_states(self):

        outputs = {
            pos: None for pos, gpio
            in self.gun_config.TRIGGER_IO.items()
                   }

        for pos, _ in self.gun_config.TRIGGER_IO.items():
            if self.IO[pos].is_pressed:
                outputs[pos] = True
            else:
                outputs[pos] = False

        return outputs


class Relay(factory.Relay):

    def getOutputDevice(self, gpio):
        return gpiozero.OutputDevice(
                f"BCM{gpio}",
                active_high=True,
                initial_value=False)
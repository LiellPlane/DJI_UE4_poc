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

    def __init__(self, _gun_config) -> None:
        super().__init__(_gun_config)
        self.relays = {}
        for relay, gpio in self.gun_config.RELAY_IO.items():
            self.relays[relay] = gpiozero.OutputDevice(
                f"BCM{gpio}",
                active_high=True,
                initial_value=False)
            self.debouncers[relay] = factory.Debounce()
            self.debouncers_1shot[relay] = factory.Debounce()
            print(f"GPIO {gpio} set for relay {relay}")

    def force_set_relay(
        self,
        relaypos: int,
        state: bool
    ):
        debouncer = self.debouncers[relaypos]

        if state is True:
            debouncer.trigger(self.relays[relaypos].on)
        else:
            debouncer.trigger(self.relays[relaypos].off)
        return  None

    def set_relay(
            self,
            relaypos: int,
            state: bool,
            strobe_cnt: int):

        # sometimes we need to strobe the relays for special
        # hardware - for instance IR light that has 3 modes
        debouncer = self.debouncers[relaypos]

        #  functions as variables to make it a bit easier to read
        #  GPIO.setup(port_or_pin, GPIO.OUT)
        debounce_on = functools.partial(
                    debouncer.trigger,
                    self.relays[relaypos].on)

        debounce_off = functools.partial(
                    debouncer.trigger,
                    self.relays[relaypos].off)

        if (strobe_cnt == 0) or (state is False):
            if state:
                return debounce_on()
            else:
                return debounce_off()

        if strobe_cnt == 0 or state is False:
            raise Exception("Bad logic to relay strobe")

        # strobing isn't used anymore
        # different logic for strobing
        strobe_state = True

        for _ in range((strobe_cnt * 2) - 1):
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

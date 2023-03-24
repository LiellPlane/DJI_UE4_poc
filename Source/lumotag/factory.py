from abc import ABC, abstractmethod
import numpy as np
import time
from enum import Enum
import cv2

RELAY_IO_BOARD = {1:29, 2:31, 3:16}
RELAY_IO_BCM = {1:5, 2:6, 3:23}
RELAY_IO = RELAY_IO_BCM
TRIGGER_IO_BOARD = {1:15, 2:13}
TRIGGER_IO_BCM = {1:22, 2:27}
TRIGGER_IO = TRIGGER_IO_BCM


class RelayFunction(Enum):
    torch = 1
    unused_1 = 2
    unused_2 = 3


class display(ABC):
    @abstractmethod
    def display_output(self):
        pass


class Accelerometer(ABC):

    @abstractmethod
    def get_vel(self) -> tuple:
        pass


class Triggers(ABC):

    @abstractmethod
    def test_states(self) -> list [bool]:
        pass


class GetImage(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.res_select = 0

    @abstractmethod
    def gen_image(self):
       pass

    def __next__(self):
        img = self.gen_image()
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def __iter__(self):
        return self


class Relay(ABC):
    def __init__(self) -> None:
        self.debouncers = {}
    @abstractmethod
    def set_relay(self):
        pass


class KillProcess(ABC):
    @abstractmethod
    def kill(self):
        pass


class Debounce:

    def __init__(self) -> None:
        self.debouncetime_sec = 0.05
        self.debouncer = TimeDiffObject()

    def trigger(self, triggerfunc, *args):
        if self.debouncer.get_dt() < self.debouncetime_sec:
            return False
        else:
            triggerfunc(*args)
            self.debouncer.reset()
            return True


class TimeDiffObject:
    """stopwatch function"""

    def __init__(self) -> None:
        self._start_time = time.perf_counter()

    def get_dt(self) -> float:
        """gets time in seconds since last reset/init"""
        self._stop_time = time.perf_counter()
        difference_ms = self._stop_time-self._start_time
        return difference_ms

    def reset(self):
        self._start_time = time.perf_counter()

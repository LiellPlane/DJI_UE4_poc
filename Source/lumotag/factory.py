from abc import ABC, abstractmethod
import numpy as np
import time

class Accelerometer(ABC):

    @abstractmethod
    def get_vel(self) -> np.array:
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
    def get_image(self):
        pass

class Relay(ABC):

    @abstractmethod
    def set_outputs(self):
        pass

class Trigger:
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
        self._stop_time = time.perf_counter()

    def get_dt(self) -> float:
        """gets time in seconds since last reset/init"""
        self._stop_time = time.perf_counter()
        difference_ms = self._stop_time-self._start_time
        return difference_ms

    def reset(self):
        self._start_time = time.perf_counter()

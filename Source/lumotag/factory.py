from abc import ABC, abstractmethod
import numpy as np
import time
from enum import Enum
import cv2

RELAY_IO_BOARD = {1:29, 3:31, 2:16}
RELAY_IO_BCM = {1:5, 3:6, 2:23}
RELAY_IO = RELAY_IO_BCM

TRIGGER_IO_BOARD = {1:15, 2:13}
TRIGGER_IO_BCM = {1:22, 2:27}
TRIGGER_IO = TRIGGER_IO_BCM


class screensizes(Enum):
    pi_4 = (480, 740)
    windows_laptop = (1000, 1000)


class RelayFunction(Enum):
    torch = 1
    unused_1 = 2
    unused_2 = 3


class display(ABC):
    @abstractmethod
    def display_output(self):
        pass


class config(ABC):
    @property
    def env_name(self):
        raise NotImplementedError
    @abstractmethod
    def loop_wait(self):
        raise NotImplementedError

class Accelerometer(ABC):

    def __init__(self) -> None:
        self._last_xyz = None
        self._display_size = 20
        self._disp_val_lim_max = 20
        self._disp_val_lim_min = -20

    @abstractmethod
    def update_vel(self) -> tuple:
        pass

    @staticmethod
    def round(val):
        return round(val, 4)

    def interp_pos_in_img(self, xyz: np.array):
            output = []
            for el in xyz:
                output.append(
                    np.interp(
                        el,
                        [self._disp_val_lim_min, self._disp_val_lim_max],
                        [0, self._display_size]))
            output = np.asarray(output)
            output = np.clip(
                output,
                0,
                self._display_size)
            return output

    def get_visual(self):
        ds = self._display_size
        visual = np.ones((ds,ds,3))
        if self._last_xyz is None:
            return visual
        input_vec = np.asarray(self._last_xyz)
        half_ds = int(ds/2)
        # rectify and stretch to size of output
        input_vec = np.clip(
            input_vec,
            self._disp_val_lim_min,
            self._disp_val_lim_max)

        lerp_input_vec = self.interp_pos_in_img(input_vec)
        x = 0
        y = 1
        z = 2
        
        cv2.line(
            visual,
            (half_ds, int(lerp_input_vec[x])),
            (half_ds, half_ds),
            (255 ,0, 0),
            1)
        cv2.line(
            visual,
            (int(lerp_input_vec[y]), half_ds),
            (half_ds, half_ds),
            (0 ,255, 0),
            1)
        cv2.line(
            visual,
            (int(lerp_input_vec[y]), int(lerp_input_vec[y])),
            (half_ds, half_ds),
            (0 ,0, 255),
            1)
        return visual
    

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

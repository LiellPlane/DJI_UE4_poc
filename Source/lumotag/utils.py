import time
from contextlib import contextmanager
from typing import Iterator
import enum
import os
import platform
from my_collections import _OS
import shutil
import math

def DeleteFiles_RecreateFolder(FolderPath):
    Deltree(FolderPath)
    os.mkdir(FolderPath)


def Deltree(Folderpath):
      # check if folder exists
    if len(Folderpath)<6:
        raise("Input:" + str(Folderpath),"too short - danger")
        raise ValueError("Deltree error - path too short warning might be root!")
        return
    if os.path.exists(Folderpath):
         # remove if exists
         shutil.rmtree(Folderpath)
    else:
         # throw your exception to handle this special scenario
         #raise Exception("Unknown Error trying to Deltree: " + Folderpath)
         pass
    return

def get_platform():
    #  detect what OS we are on - test environment (Windows) or production (pi hardware)
    RASP_PI_4_OS = "armv7l"
    RASP_PI_MACHINE = "aarch64"
    try:
        if RASP_PI_MACHINE.lower() in platform.machine().lower():
            print(f"probably a raspberry pi - {RASP_PI_MACHINE}")
            return _OS.RASPBERRY
    except Exception:
        pass

    try:
        if hasattr(os, 'uname') is False:
            print("scambiloop raspberry presence failed, probably Windows system")
            return _OS.WINDOWS
    except Exception:
        pass

    try:
        if os.uname()[-1] == RASP_PI_4_OS:
            print("scambiloop raspberry presence detected, loading hardware libraries")
            return _OS.RASPBERRY
    except Exception:
        pass

    try:
        if os.name == 'posix' and platform.system() == 'Linux':
            print("EWWWWWWWWWWWW linux, disgusting")
            return _OS.LINUX
    except Exception:
        pass

    try:
        if os.name == 'posix' and platform.system() == 'Darwin':
            print("EWWWWWWWWWWWW Mac, disgusting")
            return _OS.MAC_OS
    except Exception:
        pass

    raise Exception(
        "Could not detect platform",
        os.name,
        platform.system(),
        platform.release())

def get_epoch_timestamp():
    return str(time.time()).replace(".", "_")

def custom_print(comment, debug=False):
    if debug is False: return
    print(comment)

@contextmanager
def time_it(comment, debug=False) -> Iterator[None]:
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        if debug is True:
            toc: float = time.perf_counter()
            output = f"{comment}:Computation time"
            time_ = f"{1000*(toc - tic):.3f}ms"
            if "total" in comment.lower():
                buff = 65
            else:
                buff = 60
            buffer = ''.join(["="] * (buff-len(output)))
            shiftbuff = len(str(int((1000*(toc - tic))//1)))
            buffer = buffer[0:-shiftbuff]
            print(output + buffer + time_)

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


class Lerp:
    def __init__(self, start_value, end_value, duration, easing='linear'):
        """
        linear_lerp = Lerp(start_value=0, end_value=100, duration=5)

        quad_ease_in_lerp = Lerp(start_value=0, end_value=100, duration=5, easing='ease_in_quad')

        cubic_ease_out_lerp = Lerp(start_value=0, end_value=100, duration=5, easing='ease_out_cubic')

        sine_ease_in_out_lerp = Lerp(start_value=0, end_value=100, duration=5, easing='ease_in_out_sine')

        - In your main loop or update function:
        while not lerp.is_complete():
            current_value = lerp.get_value()
            print(f"Current value: {current_value}")

        -Reverse the lerp halfway through
            if current_value >= 50 and not lerp.is_reversed:
                lerp.reverse()

        """
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration
        self.start_time = None
        self.is_running = False
        self.easing = easing
        self.is_reversed = False
        self.reverse_time = None
        
        self.easing_functions = {
            'linear': self._linear,
            'ease_in_quad': self._ease_in_quad,
            'ease_out_quad': self._ease_out_quad,
            'ease_in_out_quad': self._ease_in_out_quad,
            'ease_in_cubic': self._ease_in_cubic,
            'ease_out_cubic': self._ease_out_cubic,
            'ease_in_out_cubic': self._ease_in_out_cubic,
            'ease_in_sine': self._ease_in_sine,
            'ease_out_sine': self._ease_out_sine,
            'ease_in_out_sine': self._ease_in_out_sine,
        }

    def start(self):
        self.start_time = time.time()
        self.is_running = True
        self.is_reversed = False
        self.reverse_time = None

    def stop(self):
        self.is_running = False

    def reset(self):
        self.start_time = None
        self.is_running = False
        self.is_reversed = False
        self.reverse_time = None

    def reverse(self):
        if self.is_running:
            self.is_reversed = not self.is_reversed
            self.reverse_time = time.time()

    def set_reverse_state(self, state: bool):
        if self.is_running:
            self.is_reversed = state
            self.reverse_time = time.time()

    def get_value(self):
        if not self.is_running:
            return self.start_value if self.is_reversed else self.end_value

        current_time = time.time()
        if self.is_reversed:
            if self.reverse_time is None:
                self.reverse_time = current_time
            elapsed_time = self.reverse_time - self.start_time
            reverse_elapsed = current_time - self.reverse_time
            t = max(0, (elapsed_time - reverse_elapsed) / self.duration)
        else:
            elapsed_time = current_time - self.start_time
            t = min(1.0, elapsed_time / self.duration)

        if (self.is_reversed and t <= 0) or (not self.is_reversed and t >= 1):
            self.is_running = False
            return self.start_value if self.is_reversed else self.end_value

        eased_t = self.easing_functions[self.easing](t)
        return self.start_value + (self.end_value - self.start_value) * eased_t

    def is_complete(self):
        if not self.is_running:
            return True
        current_time = time.time()
        if self.is_reversed:
            if self.reverse_time is None:
                return False
            return current_time - self.reverse_time >= self.reverse_time - self.start_time
        else:
            return current_time - self.start_time >= self.duration

    def _linear(self, t):
        return t

    def _ease_in_quad(self, t):
        return t * t

    def _ease_out_quad(self, t):
        return 1 - (1 - t) * (1 - t)

    def _ease_in_out_quad(self, t):
        return 2 * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 2) / 2

    def _ease_in_cubic(self, t):
        return t * t * t

    def _ease_out_cubic(self, t):
        return 1 - pow(1 - t, 3)

    def _ease_in_out_cubic(self, t):
        return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

    def _ease_in_sine(self, t):
        return 1 - math.cos((t * math.pi) / 2)

    def _ease_out_sine(self, t):
        return math.sin((t * math.pi) / 2)

    def _ease_in_out_sine(self, t):
        return -(math.cos(math.pi * t) - 1) / 2

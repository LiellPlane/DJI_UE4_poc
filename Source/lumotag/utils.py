import time
from contextlib import contextmanager
from typing import Iterator
import enum
import os
import platform
from my_collections import _OS
import shutil
import math
from collections import deque

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

def mapFromTo(x,a,b,c,d):
    y=(x-a)/(b-a)*(d-c)+c
    return y

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
        self.start_time = time.perf_counter()
        self.duration = duration
        self.start_value = start_value
        self.end_value = end_value
        self.easing = easing
        self.direction = 1
        self.unity_value = 0
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

    def set_direction_forward(self, state: bool):
        if state is False:
            self.direction = -1
        else:
            self.direction = 1
    
    def _get_value(self):
        if self.direction == 1 and self.unity_value >= 1:
            pass
        elif self.direction == -1 and self.unity_value <=0:
            pass
        else:
            time_difference = time.perf_counter() - self.start_time
            step_unity = time_difference / self.duration
            self.unity_value += (step_unity * self.direction)
        self.unity_value = min(1, max(0, self.unity_value))
        self.start_time = time.perf_counter()
        return self.unity_value

    def get_value(self):
        val = self._get_value()
        eased_val = self.easing_functions[self.easing](val)
        mapped_val = mapFromTo(abs(eased_val),0,1,self.start_value,self.end_value)
        return round(mapped_val, 5)

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


# class SequenceDetector:
#     def __init__(self, max_length, target_sequence):
#         self.buffer = deque(maxlen=max_length)
#         self.target_sequence = target_sequence

#     def add(self, value):
#         self.buffer.append(value)
#         return self.check_sequence()

#     def check_sequence(self):
#         return len(self.buffer) == self.buffer.maxlen and list(self.buffer) == self.target_sequence


# import time
# lerper = Lerp(start_value=0,end_value=2,duration=2,easing='ease_in_out_cubic')

# cnt = 0
# while True:
#     print(lerper.get_value())
#     time.sleep(0.1)
#     cnt += 1
#     if cnt == 5:
#         lerper.set_direction_forward(False)
#     if cnt == 25:
#         lerper.set_direction_forward(True)
#     if cnt == 50:
#         lerper.set_direction_forward(False)
#     if cnt == 99:
#         lerper.set_direction_forward(True)
#     # print(cnt)
#     # print(lerper.get_value())

import time
from contextlib import contextmanager
from typing import Iterator
import enum
import os
import platform


class _OS(str, enum.Enum):
    WINDOWS = "windows"
    RASPBERRY = "raspberry"
    LINUX = "digusting linux"


def get_platform():
    #  detect what OS we are on - test environment (Windows) or production (pi hardware)
    RASP_PI_4_OS = "armv7l"

    if hasattr(os, 'uname') is False:
        print("scambiloop raspberry presence failed, probably Windows system")
        return _OS.WINDOWS
    elif os.uname()[-1] == RASP_PI_4_OS:
        print("scambiloop raspberry presence detected, loading hardware libraries")
        return _OS.RASPBERRY
    elif os.name == 'posix' and platform.system() == 'Linux':
        print("EWWWWWWWWWWWW linux, disgusting")
        return _OS.LINUX
    else:
        raise Exception(
            "Could not detect platform",
            os.name,
            platform.system(),
            platform.release())

def get_epoch_timestamp():
    return str(time.time()).replace(".", "_")

@contextmanager
def time_it(comment) -> Iterator[None]:
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        pass
        # toc: float = time.perf_counter()
        # output = f"{comment}:Computation time"
        # time_ = f"{1000*(toc - tic):.3f}ms"
        # buffer = ''.join(["="] * (60-len(output)))
        # shiftbuff = len(str(int((1000*(toc - tic))//1)))
        # buffer = buffer[0:-shiftbuff]
        # print(output + buffer + time_)
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

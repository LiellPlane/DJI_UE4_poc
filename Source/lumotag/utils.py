import time
from contextlib import contextmanager
from typing import Iterator
import enum
import os
import platform
from my_collections import _OS
import shutil


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

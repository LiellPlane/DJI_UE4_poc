import time
from contextlib import contextmanager
from typing import Iterator

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

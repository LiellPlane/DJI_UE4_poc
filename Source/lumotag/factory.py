from abc import ABC, abstractmethod
import numpy as np
import time
from enum import Enum
import cv2
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from queue import Queue


@contextmanager
def time_it(process):
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        toc: float = time.perf_counter()
        print(f"time for {process} = {1000*(toc - tic):.3f}ms")


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


# class config(ABC):
#     torch = 1
#     triggerclick = 2
    
#     RELAY_IO_BOARD = {1:29, 3:31, 2:16}
#     RELAY_IO_BCM = {1:5, 3:6, 2:23}
#     RELAY_IO = RELAY_IO_BCM

#     TRIGGER_IO_BOARD = {1:15, 2:13}
#     TRIGGER_IO_BCM = {1:22, 2:27}
#     TRIGGER_IO = TRIGGER_IO_BCM

#     @property
#     def env_name(self):
#         raise NotImplementedError
#     @abstractmethod
#     def loop_wait(self):
#         raise NotImplementedError


class gun_config(ABC):

    def __init__(self) -> None:
        self.relay_map = {
            "laser" : 2,
            "torch" : 1,
            "clicker" : 3}
        self.messaging_config = {
            'username' : 'guest',
            'password' : 'guest',
            'host' : 'lumotagHQ.local',
            'port' : 5672,
            'virtual_host' : '/'
        }

    @property
    @abstractmethod
    def rly_torch(self):
        ...
    @property
    @abstractmethod
    def rly_triggerclick(self):
        ...
    @property
    @abstractmethod
    def model_name(self):
        ...
    @property
    @abstractmethod
    def RELAY_IO(self):
        ...
    @property
    @abstractmethod
    def TRIGGER_IO(self):
        ...
    @property
    @abstractmethod
    def screen_rotation(self):
        ...
    @abstractmethod
    def loop_wait(self):
        ...


class stryker_config(gun_config):
    
    def __init__(self) -> None:
        super().__init__()
        #for reference on rasperry pi 4
        self.RELAY_IO_BOARD = {1:29, 3:31, 2:16}
        self.RELAY_IO_BCM = {1:5, 3:6, 2:23}
        self.TRIGGER_IO_BOARD = {1:15, 2:13}
        self.TRIGGER_IO_BCM = {1:22, 2:27}

    @property
    def rly_torch(self):
        return 1

    @property
    def rly_triggerclick(self):
        return 2
    
    @property
    def model_name(self):
        return ("stryker")
    
    @property
    def RELAY_IO(self):
        return(self.RELAY_IO_BCM)
    
    @property
    def TRIGGER_IO(self):
        return (self.TRIGGER_IO_BCM)
    
    @property
    def screen_rotation(self):
        return(0)

    def loop_wait(self):
        pass


class TZAR_config(gun_config):

    def __init__(self) -> None:
        super().__init__()
        #for reference on rasperry pi 4
        self.RELAY_IO_BOARD = {1:29, 3:31, 2:16}
        self.RELAY_IO_BCM = {1:5, 3:6, 2:23}
        self.TRIGGER_IO_BOARD = {1:15, 2:13}
        self.TRIGGER_IO_BCM = {1:22, 2:27}

    @property
    def rly_torch(self):
        return 1

    @property
    def rly_triggerclick(self):
        return 2
    
    @property
    def model_name(self):
        return ("TZAR")
    
    @property
    def RELAY_IO(self):
        return(self.RELAY_IO_BCM)
    
    @property
    def TRIGGER_IO(self):
        return (self.TRIGGER_IO_BCM)
    
    @property
    def screen_rotation(self):
        return(90)

    def loop_wait(self):
        pass


class simitzar_config(gun_config):
    
    def __init__(self) -> None:
        super().__init__()
        #for reference on rasperry pi 4
        self.RELAY_IO_BOARD = {1:29, 3:31, 2:16}
        self.RELAY_IO_BCM = {1:5, 3:6, 2:23}
        self.TRIGGER_IO_BOARD = {1:15, 2:13}
        self.TRIGGER_IO_BCM = {1:22, 2:27}

    @property
    def rly_torch(self):
        return 1

    @property
    def rly_triggerclick(self):
        return 2
    
    @property
    def model_name(self):
        return ("sim tzaaar")
    
    @property
    def RELAY_IO(self):
        return(self.RELAY_IO_BCM)
    
    @property
    def TRIGGER_IO(self):
        return (self.TRIGGER_IO_BCM)
    
    @property
    def screen_rotation(self):
        return(0)

    def loop_wait(self):
        time.sleep(0.3)


class Accelerometer(ABC):

    def __init__(self) -> None:
        self._last_xyz = None
        self._display_size = 100
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
        
        # Using cv2.putText() method
        visual = cv2.putText(
            visual,
            '^THIS WAY UP^',
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.3,
            (255, 0, 0),
            1,
            cv2.LINE_AA)

        cv2.line(
            visual,
            (int(lerp_input_vec[y]), half_ds),
            (half_ds, half_ds),
            (255 ,0, 0),
            1)
        cv2.line(
            visual,
            (half_ds, int(lerp_input_vec[x])),
            (half_ds, half_ds),
            (0 ,255, 0),
            1)
        cv2.line(
            visual,
            (int(lerp_input_vec[z]), int(lerp_input_vec[z])),
            (half_ds, half_ds),
            (0 ,0, 255),
            1)
        # get 
        cv2.line(
            visual,
            (int(lerp_input_vec[y]), int(lerp_input_vec[x])),
            (half_ds, half_ds),
            (0 ,0, 0),
            2)

        return visual
    

class Triggers(ABC):

    def __init__(self, _gun_config) -> None:
        self.gun_config = _gun_config
    @abstractmethod
    def test_states(self) -> list [bool]:
        pass


class Camera(ABC):

    @property
    def angle_vs_world_up(self):
        raise NotImplementedError

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
    def __init__(self, _gun_config) -> None:
        self.debouncers = {}
        self.gun_config = _gun_config
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


class messenger(ABC):

    def __init__(self, config: gun_config) -> None:
        """msg_worker can be None, or a specific messaging
        implementation such as RabbitMQ"""
        self._in_box = Queue(maxsize = 3)
        self._out_box = Queue(maxsize = 3)
        self._config = config

        self.inbox_worker = threading.Thread(
            target=self._in_box_worker,
            args=(self._in_box, ))
        self.inbox_worker.start()

        self.outbox_worker = threading.Thread(
            target=self._out_box_worker,
            args=(self._out_box, ))
        self.outbox_worker.start()

    @abstractmethod
    def _in_box_worker(self, in_box):
        pass

    @abstractmethod
    def _out_box_worker(self, out_box):
        pass

    def send_message(self, message: str) -> bool:
        if self._out_box._qsize() >= self._out_box.maxsize - 1:
            print("Message outbox full!!")
            return
        self._out_box.put(
            message,
            block=False)

    def check_in_box(self):
        message = None
        if self._in_box._qsize()>0:
            try:
                message = self._in_box.get(block=False)
            except Queue.Empty:
                pass
        return message
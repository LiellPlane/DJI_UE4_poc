from abc import ABC, abstractmethod
import numpy as np
import time
from enum import Enum
import cv2
from contextlib import contextmanager
from dataclasses import dataclass
import threading
#from queue import Queue
import queue
import uuid
from enum import Enum
from multiprocessing import Process, Queue, shared_memory
from functools import reduce
import img_processing
from math import floor
RELAY_BOUNCE_S = 0.02


class AutoStrEnum(str, Enum):
    """
    StrEnum where auto() returns the field name.
    See https://docs.python.org/3.9/library/enum.html#using-automatic-values
    """
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name






# @contextmanager
# def time_it(process):
#     tic: float = time.perf_counter()
#     try:
#         yield
#     finally:
#         toc: float = time.perf_counter()
#         print(f"time for {process} = {1000*(toc - tic):.3f}ms")


class RelayFunction(Enum):
    torch = 1
    unused_1 = 2
    unused_2 = 3

def create_id():
    return str(uuid.uuid4())


class gun_config(ABC):
    model = "NOT OVERRIDDEN!"
    DETAILS_FILE = '/boot/MY_INFO.txt'
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

        self.my_id = create_id()

        self.trigger_debounce = Debounce(
            debounce_sec=0.1)

        self.msg_heartbeat_s = 20

        self.torch_debounce = Debounce(
            debounce_sec=1.0)


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
    @property
    @abstractmethod
    def screen_size(self):
        ...
    @property
    @abstractmethod
    def opencv_window_pos(self):
        ...
    @abstractmethod
    def loop_wait(self):
        ...

    # UNIQUEFIRE T65 IR light has 3 modes
    # need to cycle through them each time
    @abstractmethod
    def light_strobe_cnt(self):
        ...
    @abstractmethod
    def internal_img_crop(self):
        ...
    @property
    @abstractmethod
    def video_modes(self):
        ...


class filesystem(ABC):
    @abstractmethod
    def save_image(self):
        pass


class display(ABC):
    
    def __init__(self,  _gun_config: gun_config) -> None:
        self.display_rotate = _gun_config.screen_rotation
        self.screen_size = _gun_config.screen_size
        self.opencv_win_pos = _gun_config.opencv_window_pos
        self.emptyscreen = np.zeros(
            ( _gun_config.screen_size + (3,)), np.uint8)
        self.draw_test_rect()

    def draw_test_rect(self):
        buffer = int(self.emptyscreen.shape[0]/100)
        self.emptyscreen = cv2.rectangle(
            self.emptyscreen,
            (buffer, buffer),
            tuple(np.asarray(list(reversed(self.emptyscreen.shape[0:2]))) - np.asarray([buffer, buffer])),
            (255,255,255),
            min(int(buffer/2),2))

    @abstractmethod
    def display_method(image, self):
        pass

    def set_image_in_centre(self, inputimg):
        if inputimg.shape[0] == self.emptyscreen.shape[0]:
            offset = floor((self.emptyscreen.shape[1] - inputimg.shape[1]) /2)
            self.emptyscreen[:, offset:inputimg.shape[1]+offset, 0] = inputimg
            self.emptyscreen[:, offset:inputimg.shape[1]+offset, 1] = inputimg
            self.emptyscreen[:, offset:inputimg.shape[1]+offset, 2] = inputimg
        elif inputimg.shape[1] == self.emptyscreen.shape[1]:
            offset = floor((self.emptyscreen.shape[0] - inputimg.shape[0]) /2)
            self.emptyscreen[offset:inputimg.shape[0]+offset, :, 0] = inputimg
            self.emptyscreen[offset:inputimg.shape[0]+offset, :, 1] = inputimg
            self.emptyscreen[offset:inputimg.shape[0]+offset, :, 2] = inputimg
        else:
            raise Exception("Warning, bad resized image shapes", inputimg.shape,  self.emptyscreen.shape )

    def display_output(self, output):
        # quicker in theory to resize first then rotate as
        # input image is expected to be much larger than display size
        if self.display_rotate == 90:
            #output = cv2.resize(output, tuple(reversed(self.screen_size)))
            #output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
            output = img_processing.get_resized_equalaspect(
                output,
                tuple(reversed(self.screen_size)))
            output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
            
        elif self.display_rotate == -90 or self.display_rotate == 270:
            output = img_processing.get_resized_equalaspect(
                output,
                tuple(reversed(self.screen_size)))
            output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)

        elif self.display_rotate == 180:
            output = img_processing.get_resized_equalaspect(
                output,
                self.screen_size)
            output = cv2.rotate(output, cv2.ROTATE_180)

        elif self.display_rotate == 0:
            # output, scale_factor = img_processing.resize_centre_img(
            #    output,
            #    self.screen_size)
            #output = img_processing.add_cross_hair(output, adapt=True)
            #output = cv2.resize(output, self.screen_size)
            output = img_processing.get_resized_equalaspect(
                output,
                (self.screen_size))
            

        else:
            raise Exception("incorrect display rotate value", self.display_rotate)
        #output = img_processing.add_cross_hair(output, adapt=True)

        
        self.set_image_in_centre(output)
        img_processing.add_cross_hair(self.emptyscreen, adapt=True)
        self.display_method(self.emptyscreen)

    @abstractmethod
    def display_output_with_implant(self):
        pass


class Accelerometer(ABC):

    def __init__(self) -> None:
        self._last_xyz = None
        self._display_size = 100
        self._disp_val_lim_max = 20
        self._disp_val_lim_min = -20
        self._fifo = []
        self._timer = TimeDiffObject()

    @abstractmethod
    def update_vel(self) -> tuple:
        pass

    def update_fifo(self):
        return
        if self._last_xyz is not None:
            self._fifo.append(
                np.asarray(self._last_xyz))
        if self._timer.get_dt() > 0.01:
            plop=1


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
        input_vec = self._last_xyz
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
    def test_states(self):
        pass


class ImageGenerator(ABC):
    @abstractmethod
    def get_image(self):
        pass


class Camera(ABC):

    def __init__(self, video_modes) -> None:
        self.res_select = 0
        self.last_img = None
        self.cam_res = video_modes

    @abstractmethod
    def gen_image(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    def __iter__(self):
        return self

    def get_res(self):
        return [
            e.value for e in self.cam_res][self.res_select][1]


class Camera_synchronous(Camera):
    
    def __init__(self, video_modes, imagegen_cls) -> None:
        super().__init__(video_modes)
        self.imagegen_cls = imagegen_cls(self.get_res())

    def gen_image(self):
        return self.imagegen_cls.get_image()

    def __next__(self):
        img = self.gen_image()
        self.last_img = img
        return img


class Camera_async(Camera):
    
    def __init__(self, video_modes, imagegen_cls) -> None:
        self.res_select = 0
        self.last_img = None
        self.handshake_queue = Queue(maxsize=1)
        self.process = None
        self.shared_mem_handler = None
        self.cam_res = video_modes
        # this has to be after initialising self.cam_res
        self.imagegen_cls = imagegen_cls
        # this would be nice to have in a __post_init__ type thing
        self.configure_shared_memory()
 
    def configure_shared_memory(self):
        # we need to get shape of image first to
        # create memory buffer
        # don't call this before everything else has been initialised!

        img_byte_size = reduce(
            lambda acc, curr: acc * curr, self.get_res())


        self.shared_mem_handler = SharedMemory(
                            obj_bytesize=img_byte_size,
                            discrete_ids=[str(self.res_select)]
                                        )

        memblock = self.shared_mem_handler.mem_ids[str(self.res_select)]

        func_args = (
            self.handshake_queue,
            memblock,
            self.get_res(),
            self.imagegen_cls)

        process = Process(
            target=self.async_img_loop,
            args=func_args,
            daemon=True)

        process.start()

    def __next__(self):
        return self.gen_image()

    def gen_image(self):
        # popping the queue item unblocks image sender
        _ = self.handshake_queue.get(
                        block=True,
                        timeout=None
                        )

        strm_buff = self.shared_mem_handler.mem_ids[str(self.res_select)].buf
        res = self.get_res() 
        #range returns nothing if out of bounds - eg if using grayscale image
        #(w, h, c)
        if len(res) == 2:
            _shape = (res[1], res[0])
        else:
            _shape = (res[0], res[1], res[2])
        
        img_buff = np.frombuffer(
            strm_buff,
            dtype=('uint8')
                ).reshape(_shape)

        #if len(img_buff.shape) == 3:
        #    img_buff = cv2.cvtColor(img_buff, cv2.COLOR_BGR2GRAY)

        self.last_img = img_buff

        return img_buff

    def async_img_loop(
        self,
        myqueue: Queue,
        shared_mem_object: shared_memory.SharedMemory,
        res: tuple,
        img_gen: ImageGenerator):

        _img_gen = img_gen(res)

        shared_mem = None

        while True:
            img = _img_gen.get_image()
            # one-time initialise buffer
            if shared_mem is None:
                shared_mem: np.ndarray = np.ndarray(
                img.shape,
                dtype=img.dtype,
                buffer=shared_mem_object.buf
            )

            shared_mem[:] = img[:]
            #blocking put until consumer handshakes 
            myqueue.put("image_ready", block=True, timeout=None)



class Relay(ABC):
    def __init__(self, _gun_config) -> None:
        self.debouncers = {}
        self.debouncers_1shot = {}
        self.gun_config = _gun_config
    @abstractmethod
    def set_relay(self):
        pass


class KillProcess(ABC):
    @abstractmethod
    def kill(self):
        pass


class Debounce:

    def __init__(self, debounce_sec = RELAY_BOUNCE_S) -> None:
        self.debouncetime_sec = debounce_sec
        self.debouncer = TimeDiffObject()
        self._statemem = False
        self._stateheld = False
        self._configuration = None

    def set_check_config(self, funcname):
        if self._configuration is None:
            self._configuration = funcname
        else:
            if self._configuration != funcname:
                print("debounce config:", self._configuration)
                print("attempted reconfig:", funcname)
                raise Exception("debouncer config mix-up, multiple configs")
        
    def can_trigger(self):
        return self.debouncer.get_dt() >= self.debouncetime_sec
    
    def get_memstate(self):
        return self._statemem

    def get_heldstate(self):
        return self._stateheld

    def trigger(self, triggerfunc, *args):
        self.set_check_config("trigger")
        if self.can_trigger() is False:
            return False
        else:
            triggerfunc(*args)
            self.debouncer.reset()
            return True

    def trigger_oneshot(self, boolstate, triggerfunc, *args):
        """needs to be released before retriggering with
        symmetrical delay"""
        self.set_check_config("trigger_oneshot")

        if self.can_trigger() is True:
            if self._statemem != boolstate:
                triggerfunc(*args)
                self.debouncer.reset()
                return True
            self._statemem = boolstate
        return False

    def trigger_oneshot_simple(self, boolstate):
        """needs to be released before retriggering with
        symmetrical delay but you handle the function
        yourself, for more complicated events"""
        self.set_check_config("trigger_oneshot_simple")

        if self.can_trigger() is True:
            if self._statemem != boolstate:
                self.debouncer.reset()
                self._statemem = boolstate
                return True
            self._statemem = boolstate
        return False

    def trigger_1shot_simple_High(self, boolstate):
        """needs to be released before retriggering but
        upon release has no wait period
        
        use with get mem state"""
        self.set_check_config("trigger_1shot_simple_High")
        # in this condition we can turn it straight back on
        if boolstate is True and self._statemem is False and self._statemem is False:
            self.debouncer.reset()
            self._stateheld = True
            self._statemem = True
            return True

        # here we are still holding mem HIGH until
        # time out
        if self.can_trigger() is True:
            self._stateheld = False
            self._statemem = boolstate

        return False


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


class Messenger(ABC):

    def __init__(self,
                 config: gun_config) -> None:
        self._in_box = queue.Queue(maxsize=2)
        self._out_box = queue.Queue(maxsize=2)
        self._schedule = queue.Queue(maxsize=1)
        self._config = config

        self.inbox_worker = threading.Thread(
            target=self._in_box_worker,
            args=(self._in_box, self._config, self._schedule, ))
        self.inbox_worker.start()

        self.outbox_worker = threading.Thread(
            target=self._out_box_worker,
            args=(self._out_box, self._config, self._schedule, ))
        self.outbox_worker.start()

        self.heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(self._out_box,  self._config, ))
        self.heartbeat.start()

    @abstractmethod
    def _heartbeat(self, out_box, config):
        """specfically for rabbitMQ but lets keep it here for
         test implementations to inherit"""
        pass

    @abstractmethod
    def _in_box_worker(self, in_box, config, scheduler):
        pass

    @abstractmethod
    def _out_box_worker(self, out_box, config, scheduler):
        pass

    def send_message(
            self,
            message: bytes) -> bool:

        if self._out_box._qsize() >= self._out_box.maxsize:
            print("Message outbox full!!")
            return

        self._out_box.put(
            message,
            block=False)

    def check_in_box(self, blocking=False):
        messages = []
        try:
            # just get one but keep in list structure for convenience
            messages.append(self._in_box.get(block=blocking))
        except queue.Empty:
            pass

        return messages

def get_config(model) -> gun_config:
    for subclass_ in gun_config.__subclasses__():
        if subclass_.model.lower() == model.lower():
            return subclass_()
    raise Exception("No config found for model ID ", str(model))


class SharedMemory():
    def __init__(self, obj_bytesize: int,
                 discrete_ids: list[str]
                 ):
        """Memory which can be shared between processes.

            obj_bytesize: expected size of payload

            discrete_ids: for each element create a
            shared memory object and associate with ID"""
        self._bytesize = obj_bytesize
        self.mem_ids = {}

        for my_id in discrete_ids:
            try:
                self.mem_ids[my_id] = (shared_memory.SharedMemory(
                    create=True,
                    size=obj_bytesize,
                    name=my_id))

            except FileExistsError:
                print(f"Warning: shared memory {my_id} has not been cleaned up")
                self.mem_ids[my_id] = (shared_memory.SharedMemory(
                    create=False,
                    size=obj_bytesize,
                    name=my_id))


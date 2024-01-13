from abc import ABC, abstractmethod
import numpy as np
import time
from enum import Enum
from functools import lru_cache
import cv2
import os
import threading
import random
#from queue import Queue
import queue
import uuid
from enum import Enum
from multiprocessing import Process, Queue, shared_memory
from functools import reduce
import img_processing
from math import floor
from functools import reduce
from my_collections import AffinePoints, ShapeItem, CropSlicing

from my_collections import SharedMem_ImgTicket

try:
    pass
except Exception as e:
    # TODO
    print("this must be scambilight - bad solution please fix TODO")


RELAY_BOUNCE_S = 0.02

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
        self._affine_transform = None

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

    # def set_image_in_centre(self, inputimg):
    #     if inputimg.shape[0] == self.emptyscreen.shape[0]:
    #         offset = floor((self.emptyscreen.shape[1] - inputimg.shape[1]) /2)
    #         self.emptyscreen[:, offset:inputimg.shape[1]+offset, 0] = inputimg
    #         self.emptyscreen[:, offset:inputimg.shape[1]+offset, 1] = inputimg
    #         self.emptyscreen[:, offset:inputimg.shape[1]+offset, 2] = inputimg
    #     elif inputimg.shape[1] == self.emptyscreen.shape[1]:
    #         offset = floor((self.emptyscreen.shape[0] - inputimg.shape[0]) / 2)
    #         self.emptyscreen[offset:inputimg.shape[0]+offset, :, 0] = inputimg
    #         self.emptyscreen[offset:inputimg.shape[0]+offset, :, 1] = inputimg
    #         self.emptyscreen[offset:inputimg.shape[0]+offset, :, 2] = inputimg
    #     else:
    #         raise Exception(
    #             "Warning, bad resized image shapes",
    #             inputimg.shape,
    #             self.emptyscreen.shape)

    @staticmethod
    @lru_cache
    def get_affine_points(incoming_img_dims, outgoing_img_dims) -> AffinePoints:
        """Return the corresponding points to fit the incoming image central to the
        view screen maintaining the aspect ratio, to be used to calculate affine
        transform
        
        inputs:
        incoming_img_dims: numpy array .shape
        outcoming_img_dims: numpy array .shape

        return source points, target points
        """
        incoming_w = incoming_img_dims[1]
        incoming_h = incoming_img_dims[0]
        outgoing_w = outgoing_img_dims[1]
        outgoing_h = outgoing_img_dims[0]
        incoming_pts = AffinePoints(
            top_left_w_h=(0,0),
            top_right_w_h=(incoming_w , 0),
            lower_right_w_h=(incoming_w , incoming_h))
        # pick any ratio
        ratio = outgoing_h / incoming_h
        # if resizing with aspect ratio doesn't fit, do the other way
        if floor(incoming_w * ratio) > outgoing_w:
            ratio = outgoing_w / incoming_w
        output_fit_h = floor(incoming_h * ratio)
        output_fit_w = floor(incoming_w * ratio)
        # test to make sure aspect ratio is 
        if abs((incoming_h/incoming_w) - (outgoing_h/outgoing_w)) > 2:
            raise ValueError("error calculating output image dimensions")
        # get 3 corresponding points from the output view - keeping in mind
        # any rotation
        w_crop_in = (outgoing_w - output_fit_w) // 2
        h_crop_in = (outgoing_h - output_fit_h) // 2
        view_pts = AffinePoints(
            top_left_w_h=(w_crop_in, h_crop_in),
            top_right_w_h=(w_crop_in + output_fit_w, h_crop_in),
            lower_right_w_h=(w_crop_in + output_fit_w, h_crop_in + output_fit_h))

        return incoming_pts, view_pts 

    @staticmethod
    def rotate_affine_targets(targets, degrees, outputscreen_shape):
        mid_img = [int(x/2) for x in outputscreen_shape[0:2][::-1]] # get reversed dims
        new_target = AffinePoints(
                    top_left_w_h=img_processing.rotate_pt_around_origin(targets.top_left_w_h, mid_img, degrees),
                    top_right_w_h=img_processing.rotate_pt_around_origin(targets.top_right_w_h, mid_img, degrees),
                    lower_right_w_h=img_processing.rotate_pt_around_origin(targets.lower_right_w_h, mid_img, degrees))
        return new_target


    def generate_output_affine(self, output):
        """use affine transform to resize and rotate image in one calculation
        need 2 sets of 3 corresponding points to create calculation"""

        if self._affine_transform is None:        
            if self.display_rotate == 90 or self.display_rotate == -90 or self.display_rotate == 270:
                reverse_output_shape = tuple(reversed(self.emptyscreen.shape[0:2]))
                # if planning for 90 degrees, swap image dims
                input_targets, output_targets = self.get_affine_points(
                    output.shape,
                    reverse_output_shape)
                output_targets = self.rotate_affine_targets(
                    output_targets,
                    self.display_rotate,
                    reverse_output_shape)

                diffs = (np.asarray(reverse_output_shape) - np.asarray(self.emptyscreen.shape[0:2]))/2
                output_targets.add_offset_h(diffs[1])
                output_targets.add_offset_w(diffs[0])

            elif self.display_rotate == 180:
                input_targets, output_targets = self.get_affine_points(
                    output.shape,
                    self.emptyscreen.shape)
                # have to flip output targets
                output_targets = self.rotate_affine_targets(
                    output_targets,
                    self.display_rotate,
                    self.emptyscreen.shape)

            elif self.display_rotate == 0:
                input_targets, output_targets = self.get_affine_points(
                    output.shape,
                    (self.emptyscreen.shape))

            self._affine_transform = img_processing.get_affine_transform(
                pts1=np.asarray(input_targets.as_array(), dtype="float32"),
                pts2=np.asarray(output_targets.as_array(), dtype="float32"))
        # get matrix multiplication here to transform graphics to fit image

        row_cols = self.emptyscreen.shape[0:2][::-1]
        outptu_img = img_processing.do_affine(output, self._affine_transform, row_cols)
        outptu_img = cv2.cvtColor(outptu_img, cv2.COLOR_GRAY2BGR)
        return outptu_img

    def add_internal_section_region(self, inputimg, _slice: CropSlicing):

        left_top = tuple(
            np.matmul(self._affine_transform, np.array([_slice.left,_slice.top,1])).astype(int))
        right_low = tuple(
            np.matmul(self._affine_transform, np.array([_slice.right,_slice.lower,1])).astype(int))
        inputimg = cv2.rectangle(inputimg, left_top, right_low, (255,255,255), 2)
        #inputimg[int(left_top[1]):int(right_low[1]), int(right_low[1])] = 100

    def display_output_with_graphics(self, output, graphics: ShapeItem):
        img_processing.add_cross_hair(
            output,
            adapt=True)
        for c in graphics:
            c.transform_points(self._affine_transform)
            img_processing.draw_pattern_output(
                image=output,
                patterndetails=c)

        self.display_method(output)
 

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
        self._is_reversed = None
        self._res = None

    @abstractmethod
    def gen_image(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    def __iter__(self):
        return self

    def get_res(self):
        if self._res is None:
            self._res =  [e.value for e in self.cam_res][self.res_select].res_width_height
        return self._res
        #return tuple(reversed([e.value for e in self.cam_res][self.res_select][1]))

    def get_is_reversed(self):
        if self._is_reversed is None:
            self._is_reversed =  [e.value for e in self.cam_res][self.res_select].shared_mem_reversed
        return self._is_reversed


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


class Camera_async_flipflop(Camera):
    """for each iteration call, the shared memory buffer is
    alternated to give other processes time to analyse
    
    another shared memory mechanism is used to determine which is
    the static buffer (not to be overwritten) during async capture of
    next image"""
    def __init__(self, video_modes, imagegen_cls) -> None:
        super().__init__(video_modes)
        self.res_select = 0
        self.last_img = None
        self.handshake_queue = Queue(maxsize=1)
        self.handshake_queue2 = Queue(maxsize=1)
        self.process = None
        self.shared_mem_handler = []
        self.shared_mem_index = None
        self._shared_id_index_name = "whatever"
        self._store_res = None
        self.get_safe_mem_details = None
        if not self.get_is_reversed():
            self._store_res = self.get_res()
        else:
            self._store_res = tuple(reversed(self.get_res()))
        # this has to be after initialising self.cam_res
        self.imagegen_cls = imagegen_cls
        # this would be nice to have in a __post_init__ type thing
        self.configure_shared_memory()
        #  hack to get around confusion with different combinations
        #  of screens orientations and camera resolutions


    def get_mem_buffers(self) -> dict:
        return (
            {0: self.shared_mem_handler[0].mem_ids["0"],
            1: self.shared_mem_handler[1].mem_ids["1"]})

    def configure_shared_memory(self):
        # we need to get shape of image first to
        # create memory buffer
        # don't call this before everything else has been initialised!

        img_byte_size = reduce(
            lambda acc, curr: acc * curr, self.get_res())


        # we add more than 1 instance of shared memory
        self.shared_mem_handler.append(SharedMemory(
                            obj_bytesize=img_byte_size,
                            discrete_ids=["0"]
                                        ))

        self.shared_mem_handler.append(SharedMemory(
                            obj_bytesize=img_byte_size,
                            discrete_ids=["1"]
                                        ))
        self.shared_mem_index = SharedMemory(
                            obj_bytesize=1,
                            discrete_ids=[self._shared_id_index_name]
                                        )

        memblock_0 = self.shared_mem_handler[0].mem_ids["0"]
        memblock_1 = self.shared_mem_handler[1].mem_ids["1"]
        memblock_index = self.shared_mem_index.mem_ids[
            self._shared_id_index_name]

        func_args = (
            self.handshake_queue,
            self.handshake_queue2,
            memblock_0, memblock_1, memblock_index,
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
        self.handshake_queue2.put("please rename this", block=True, timeout=None)
        mem_details = self.handshake_queue.get(
                        block=True,
                        timeout=None
                        )
        #print("FLIPFLOP Requested Image, NP incoming:", mem_details)
        
        strm_buff = self.shared_mem_handler[
            int(mem_details.index)].mem_ids[str(mem_details.index)].buf

        img_byte_size = reduce(
            lambda acc, curr: acc * curr, self._store_res)
        
        img_buff = np.frombuffer(
            strm_buff,
            dtype=('uint8')
                )[0:img_byte_size].reshape(self._store_res)

        self.last_img = img_buff

        self.get_safe_mem_details = SharedMem_ImgTicket(
            index=mem_details.index,
            res=mem_details.res,
            buf_size=mem_details.buf_size,
            id = mem_details.id)
        #print("FLIPFLOP saving record for analyis", self.get_safe_mem_details)
        return img_buff

    def async_img_loop(
        self,
        myqueue: Queue,
        handshake_queue: Queue,
        memblock_0: shared_memory.SharedMemory,
        memblock_1: shared_memory.SharedMemory,
        memblock_index: shared_memory.SharedMemory,
        res: tuple,
        img_gen: ImageGenerator):

        _img_gen = img_gen(res)

        shared_mem_0 = None
        shared_mem_1 = None
        shared_curr_id_quick = np.ndarray(
            [1],
            'i1',
            memblock_index.buf)
        

        while True:
            img = _img_gen.get_image()
            
            # one-time initialise buffer
            if shared_mem_0 is None:
                shared_mem_0: np.ndarray = np.ndarray(
                img.shape,
                dtype=img.dtype,
                buffer=memblock_0.buf
            )
            if shared_mem_1 is None:
                shared_mem_1: np.ndarray = np.ndarray(
                img.shape,
                dtype=img.dtype,
                buffer=memblock_1.buf
            )

            output = SharedMem_ImgTicket(
                index=shared_curr_id_quick[0],
                res=self._store_res,
                buf_size=[memblock_1.buf.shape, memblock_1.buf.shape],
                id=random.randint(1111,9999))

            if shared_curr_id_quick == [1]:
                #print("FLIPFLOP WRITING ASYNC image to 1")
                shared_mem_1[:] = img[:]
                shared_curr_id_quick = [0]
            elif shared_curr_id_quick == [0]:
                #print("FLIPFLOP WRITING ASYNC image to 0")
                shared_mem_0[:] = img[:]
                shared_curr_id_quick = [1]
            else:
                raise Exception("Invalid buffer ID")
            #blocking put until consumer handshakes
            #print("FLIPFLOP waiting to send ASYNC outgoing:", output)
            _ = handshake_queue.get(block=True, timeout=None)
            myqueue.put(output, block=True, timeout=None)
            #print("FLIPFLOP sent!! ASYNC outgoing:", output)


class Camera_async(Camera):
    
    def __init__(self, video_modes, imagegen_cls) -> None:
        super().__init__(video_modes)
        self.res_select = 0
        self.last_img = None
        self.handshake_queue = Queue(maxsize=1)
        self.process = None
        self.shared_mem_handler = None
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

        _product = reduce((lambda x, y: x * y), self.get_res())

        if not self.get_is_reversed():
            img_buff = np.frombuffer(
                strm_buff,
                dtype=('uint8')
                    )[0:_product].reshape(self.get_res())  # some systems have page size granularity of 4096 bytes (?)
        else:
            img_buff = np.frombuffer(
                strm_buff,
                dtype=('uint8')
                    )[0:_product].reshape(tuple(reversed(self.get_res())))  # some systems have page size granularity of 4096 bytes (?)

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


class Camera_async_buffer(Camera_async):

    def __init__(self, video_modes, imagegen_cls) -> None:
        super().__init__(video_modes, imagegen_cls)

    def get_img_buffer(self):
        return self.shared_mem_handler.mem_ids["0"].buf

    def release_next_image(self):
        _ = self.handshake_queue.get(
                block=True,
                timeout=None
                )

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
            if not self._in_box.empty():
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
            #try:
            try:
                # if shrd memory already exists, tidy it up or crash out
                tidy_mem = (shared_memory.SharedMemory(
                    create=False,
                    name=my_id))
                tidy_mem.close()
                tidy_mem.unlink()
            except FileNotFoundError:
                # shared memory has been tidied up previously
                pass

            self.mem_ids[my_id] = (shared_memory.SharedMemory(
                create=True,
                size=obj_bytesize,
                    name=my_id))

            # except FileExistsError:
            #     print(f"Warning: shared memory {my_id} has not been cleaned up")

class ImageLibrary(ImageGenerator):
    
    def __init__(self, res) -> None:
        self.blank_image = np.zeros(tuple(reversed(res)), np.uint8)
        imgfoler = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        #imgfoler = r"D:\OutputImages"
        self.images = jpgs_in_folder(imgfoler)
        #self.images = [i for i in self.images if "0290" in i]#0290
        #self.images = [i for i in self.images if "1704404023_3112135" in i]
        self.images = [i for i in self.images if "wearable" in i]
        self.res = res
        if len(self.images) < 1:
            raise Exception("could not find images in folder")


    def get_image(self):
        img_to_load = random.choice(self.images)
        
        img = cv2.imread(img_to_load)
        print(f"img {img_to_load}")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.resize(img, tuple(self.res[0:2]))
        self.blank_image[:] = img
        return self.blank_image

def jpgs_in_folder(directory):
    allFiles = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name[-4:len(name)] == '.jpg':
                allFiles.append(os.path.join(root, name))
    return allFiles

class VoiceBase(ABC):

    def __init__(self) -> None:
        """Class to provide synthetic
        voice prompts or alerts"""
        self.in_box = Queue(maxsize=4)
        self.t = threading.Thread(
            target=self.speaker,
            args=(self.in_box,))
        self.t.start()

    def speak(
            self,
            message: str):
        # use  in_box._qsize() to prevent
        # blowing it up
        if self.in_box.qsize() >= self.in_box._maxsize - 1:
            self.in_box.queue.clear()
            self.in_box.put(
                "Voice buffer overflow",
                block=False)
        else:
            self.in_box.put(
                message,
                block=False)

    def speaker(self, in_box):
        pass

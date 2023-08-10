#test code copied from lumotag
# test by copying and pasting using ssh
# then create a COMMON library

from abc import ABC, abstractmethod
import numpy as np
import time
from enum import Enum
import os
import cv2
from contextlib import contextmanager
from dataclasses import dataclass
import threading
#from queue import Queue
import queue
import uuid
from enum import Enum,auto
from multiprocessing import Process, Queue, shared_memory
from functools import reduce

import random


@contextmanager
def time_it(comment):
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        toc: float = time.perf_counter()
        if random.randint(1,100) < 4:
            print(f"{comment}:asynclib proc time = {1000*(toc - tic):.3f}ms")

def get_platform():
    #  detect what OS we are on - test environment (Windows) or production (pi hardware)
    RASP_PI_4_OS = "armv7l"

    if hasattr(os, 'uname') is False:
        print("async_cam_lib raspberry presence failed, probably Windows system")
        return _OS.WINDOWS
    elif os.uname()[-1] == RASP_PI_4_OS:
        print("async_cam_lib raspberry presence detected, loading hardware libraries")
        return _OS.RASPBERRY
    else:
        raise Exception("Could not detect platform")


class _OS(str, Enum):
    WINDOWS = "windows"
    RASPBERRY = "raspberry"

if get_platform() == _OS.RASPBERRY:
    # sorry not sorry
    from picamera2 import Picamera2

class ImageGenerator(ABC):
    @abstractmethod
    def get_image(self):
        pass


class Process_Scambiunits():

    def __init__(
            self,
            scambiunits,
            subsample_cutoff: int,
            flipflop: bool) -> None:
        self.in_queue = Queue(maxsize=1)
        self.done_queue = Queue(maxsize=1)
        self.initialised_scambis_q = Queue(maxsize=1)
        self.scambiunits = scambiunits
        self.subsample_cutoff = subsample_cutoff
        self.flipflop = flipflop
        print(
            "Process_Scambiunits received",
            len(scambiunits),
            "scambis")

        self._start()

    def _start(self):

        process = Process(
            target=self._run,
            args=(),
            daemon=True)

        process.start()

    def _run(self):
        for unit in self.scambiunits:
            unit.initialise()
        # we will init scambis here and pull off
        # do not have to use the output but so we 
        # can experiment with the rasberry pi
        self.initialised_scambis_q.put(
            self.scambiunits, block=True, timeout=None)

        while True:
            image = self.in_queue.get(
                block=True,
                timeout=None
                )
            return_dic = {}
            with time_it("process scambis"):
                for unit in self.scambiunits:
                    if (unit.bb_right - unit.bb_left) < self.subsample_cutoff or (unit.bb_lower-unit.bb_top) < self.subsample_cutoff:
                        color = unit.get_dominant_colour_flat(image, subsample=1)
                    else:
                        color = unit.get_dominant_colour_flat(image, subsample=2)
                    return_dic[unit.id] = color
            self.done_queue.put(
                return_dic, block=True, timeout=None)


class CameraAsync(ABC):
    
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
            lambda acc, curr: acc * curr,self.get_res())


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
        # popping the queue item unblocks image sender
        _ = self.handshake_queue.get(
                        block=True,
                        timeout=None
                        )

        strm_buff = self.shared_mem_handler.mem_ids[str(self.res_select)].buf

        img_buff = np.frombuffer(
            strm_buff,
            dtype=('uint8')
                ).reshape(self.get_res())

        #if len(img_buff.shape) == 3:
        #    img_buff = cv2.cvtColor(img_buff, cv2.COLOR_BGR2GRAY)

        self.last_img = img_buff

        return img_buff

    def __iter__(self):
        return self

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

    def get_res(self):
        return [e.value for e in self.cam_res][self.res_select][1]


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


class CsiCameraImageGen(ImageGenerator):
    
    def __init__(self, res) -> None:
        self.cam_res = res
        self.picam2 = Picamera2()
        res_xy = res[0:2]
        _config = self.picam2.create_video_configuration(
                    main={"size": res_xy,  "format": "YUV420"})#, controls={"FrameDurationLimits": (233333, 233333)})
                #self.picam2.set_controls({"ExposureTime": 1000}) # for blurring - but can get over exposed at night
        self.picam2.configure(_config)
        #  set_controls must come after config!!
        self.picam2.set_controls({"AnalogueGain": 10.0})
        self.picam2.start()
        time.sleep(0.2)

    def get_image(self):
        output = self.picam2.capture_array("main")
        # some dims will be (x,y) and some (x,y, 3)
        x, y, *_ = self.cam_res
        #x = res[0]
        #y = res[1]
        output = output[0: y, 0: x]#  Need to do this for YUV!
        return output


class SynthImgGen(ImageGenerator):
    
    def __init__(self, res) -> None:
        self.blank_image = np.zeros(res, np.uint8)

    def get_image(self):
        self.blank_image[:,:,:] = random.randint(0,255)
        self.blank_image = cv2.circle(
            self.blank_image,
            (self.blank_image.shape[1]//2, self.blank_image.shape[0]//2),
            self.blank_image.shape[0]//10,
            50,
            -1)
        return self.blank_image


def jpgs_in_folder(directory):
    allFiles = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name[-4:len(name)] == '.jpg':
                allFiles.append(os.path.join(root, name))
    return allFiles

class ImageLibrary(ImageGenerator):
    
    def __init__(self, res) -> None:
        self.blank_image = np.zeros(res, np.uint8)
        self.images = jpgs_in_folder(r"C:\VMs\SharedFolder\temp_get_imgs\get_imgs_LQ")
        self.res = res
        if len(self.images) < 1:
            raise Exception("could not find images in folder")


    def get_image(self):
        img_to_load = random.choice(self.images)
        latch = cv2.imread(img_to_load)
        latch = cv2.resize(latch, list(reversed(self.res[0:2])))
        return latch
    

class ScambilightCamImageGen(ImageGenerator):
    
    def __init__(self, res) -> None:
        
        self.cam_res = res
        self.picam2 = Picamera2()
        # have to reverse as quirk of ov5647 camera
        res_xy = tuple(reversed(res[0:2]))
        _config = self.picam2.create_video_configuration(
                    main={"size": res_xy, "format": "RGB888"})#, controls={"FrameDurationLimits": (233333, 233333)})
        self.picam2.configure(_config)
        #  set_controls must come after config!!
        #self.picam2.set_controls({"AnalogueGain": 30.0})
        #self.picam2.set_controls({"ExposureTime": 1000000}) # for blurring - but can get over exposed at night
        #self.picam2.set_controls({"FrameDurationLimits": (1000,1000)})
        self.picam2.set_controls({"ExposureTime": 100000000, "AnalogueGain": 1.0})
        self.picam2.start()
        time.sleep(0.2)

    def get_image(self):
        output = self.picam2.capture_array("main")
        return output


class CSI_Camera_Async(CameraAsync):

    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ScambilightCamImageGen)


class Scamblight_Camera_Async(CameraAsync):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ScambilightCamImageGen)


class Synth_Camera_Async(CameraAsync):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ImageLibrary)


class HQ_GS_Cam_vidmodes(Enum):
    """global shutter model"""
    _2 = ["1456 × 1088p50,",(1456, 1088)]


class ScambiLight_Cam_vidmodes(Enum):
    """scambilight fisheye ov5647"""
    # dimensions are reversed (h, w) due to quirk of ov5647
    _2 = ["640x480 [58.92 fps - (16, 0)/2560x1920 crop]",(480, 640, 3)]
    _1 = ["1296x972 [43.25 fps - (0, 0)/2592x1944 crop]",(972, 1296 , 3)]

    
    _3 = ["1920x1080 [30.62 fps - (348, 434)/1928x1080 crop]",(1080, 1920 , 3)]
    _4 = ["2592x1944 [15.63 fps - (0, 0)/2592x1944 crop]",(1944, 2592, 3)]


def lumo_viewer(
        inputimage,
        move_windowx,
        move_windowy,
        pausetime_Secs=0,
        presskey=False,
        destroyWindow=True):
    try:
        cv2.imshow("img", inputimage)
        cv2.moveWindow("img", move_windowx, move_windowy)
        if presskey==True:
            cv2.waitKey(0); #any key
    
        if presskey==False:
            if cv2.waitKey(20) & 0xFF == 27:
                    pass
        if pausetime_Secs>0:
            time.sleep(pausetime_Secs)
        if destroyWindow==True: cv2.destroyAllWindows()

    except Exception as e:
        print(e)

# def main():
#     image_capture2 = Scamblight_Camera_Async(ScambiLight_Cam_vidmodes)

#     weewee = 0
#     while True:
#         fart = next(image_capture2)
#         #lumo_viewer(fart,0,0,0,False,False)
#         print("final output", fart.shape)
#         cv2.imwrite(f"/home/scambilight/0{weewee}.jpg", fart)
#         weewee = weewee + 1

# if __name__ == '__main__':
#     main()

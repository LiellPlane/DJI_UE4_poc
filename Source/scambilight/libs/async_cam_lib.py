
import os

# factory is imported from another directory by module load
from factory import (
    Camera_async,
    Camera_synchronous,
    ImageGenerator,
    Camera_async_buffer,
    Camera_synchronous_with_buffer)

import numpy as np
import time
import os
import cv2
from contextlib import contextmanager
from multiprocessing import Process, Queue
from libs.utils import (
    get_platform,
    _OS,
    time_it_sparse)
import random
from itertools import permutations
from itertools import islice

if get_platform() == _OS.RASPBERRY:
    # sorry not sorry
    from picamera2 import Picamera2




class FinishedProcess():
    finished = True



class RunScambisWithAsyncImage():
    """ Run scambiunits in a parallel Process

    scambiunits: list of scambiunits
    async_image_buf: shared memory object
    curr_img: ndarray image, used to get shape, dtype etc
    subsample_cutoff: parameter to set dynamic subsampling
    Scambi_unit_LED_only: dataclass, easier to pass in than import
        """
    def __init__(
            self,
            scambiunits,
            async_image_buf,
            curr_img,
            Scambi_unit_LED_only,
            subsample_cutoff: int) -> None:
        self.subsample_cutoff = subsample_cutoff
        self.scambiunits = scambiunits
        self.done_queue = Queue(maxsize=1)
        self.handshake_queue = Queue(maxsize=1)
        self.curr_img = curr_img
        self.Scambi_unit_LED_only = Scambi_unit_LED_only
        args = (
            self.done_queue,
            async_image_buf,
            self.handshake_queue)
    
        self._process = Process(
            target=self._run,
            args=args,
            daemon=True)

        self._process.start()

    def _run(
            self,
            done_q,
            async_img_buf,
            handshake_queue):

        prev: np.ndarray = np.ndarray(
            self.curr_img.shape,
            dtype=self.curr_img.dtype,
            buffer=async_img_buf.buf)

        while True:
            scambiunits_led_info = []
            for unit in self.scambiunits:
                unit.set_dom_colour_with_auto_subsample(prev, cut_off = self.subsample_cutoff)
                scambiunits_led_info.append(self.Scambi_unit_LED_only(
                    colour=unit.colour,
                        physical_led_pos=unit.physical_led_pos))
            done_q.put(scambiunits_led_info, block=True, timeout=None)
            _ = handshake_queue.get(block=True,timeout=None)


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
                unit, block=True, timeout=None)
        self.initialised_scambis_q.put(
            FinishedProcess(), block=True, timeout=None)
        return


class SynthImgGen(ImageGenerator):
    
    def __init__(self, res) -> None:
        self.blank_image = np.zeros(res, np.uint8)

    def _get_image(self):
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


class RandomColour(ImageGenerator):

    def __init__(self, res) -> None:
        self.blank_image = np.zeros(res, np.uint8)

    def _get_image(self):
        self.blank_image[:] = random.randint(20,255)
        return self.blank_image
    

class ImageLibrary(ImageGenerator):
    
    def __init__(self, res) -> None:
        self.blank_image = np.zeros(res, np.uint8)
        imgfoler = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.images = jpgs_in_folder(imgfoler)
        self.res = res
        if len(self.images) < 1:
            raise Exception("could not find images in folder")

        self.img_to_load = random.choice(self.images)
        self.latch = cv2.imread(self.img_to_load)
        self.latch = cv2.resize(self.latch, list(reversed(self.res[0:2])))

    def _get_image(self):
        
        latch = self.latch.copy()
        #random.randint(0,255)
        # latch[:, :, 0] = latch[:, :, 0] * random.random()
        # latch[:, :, 1] = latch[:, :, 1] * random.random()
        # latch[:, :, 2] = latch[:, :, 2] * random.random()
        colours = [[255,0,0],[0,255,0],[0,0,255]]

        #latch[height-1,width-1,:] 
        colors = [0, 125, 255]
        color_permutations = list(permutations(colors, 3))
        _, width, _ = latch.shape
        # lazy bastard
        #latch[:] =random.choice(color_permutations)
        #latch[:,0:width//2,:] =random.choice(color_permutations)
        latch[:,:,0] =random.choice(random.choice(color_permutations))
        latch[:,0:width//2,0] =random.choice(random.choice(color_permutations))
        #latch[:, :, 0] = 0
        #latch[:, :, 1] = random.randint(0,255)
        #latch[:, :, 2] = 0
        time.sleep(0.02)
        return latch
    def get_raw_image(self):
        """Get full uncropped raw image from source. Must be implemented by subclasses."""
        raise Exception("raw image not implemented!")

class ScambilightCamImageGen_fps_test(ImageGenerator):
    
    def __init__(self, res) -> None:
        from libcamera import controls
        self.cam_res = res
        self.picam2 = Picamera2()
        res_xy = tuple(reversed(res[0:2]))
        self.picam2.video_configuration.controls.FrameRate = 30.0
        self.picam2.video_configuration.controls.AnalogueGain = 10.0
        self.picam2.video_configuration.controls.AwbEnable = False
        self.picam2.video_configuration.controls.AeMeteringMode = controls.AeMeteringModeEnum.Spot
        self.picam2.video_configuration.size = res_xy
        self.picam2.video_configuration.format = "RGB888"
        self.picam2.start("video")
        time.sleep(0.2)

    def _get_image(self):
        output = self.picam2.capture_array("main")
        return output

    def get_raw_image(self):
        """Get full uncropped raw image from source. Must be implemented by subclasses."""
        raise Exception("raw image not implemented!")

class ScambilightCamImageGen(ImageGenerator):
    
    def __init__(self, res) -> None:
        from libcamera import controls
        self.cam_res = res
        self.picam2 = Picamera2()
        # have to reverse as quirk of ov5647 camera
        res_xy = tuple(reversed(res[0:2]))
        _config = self.picam2.create_video_configuration(
                    main={"size": res_xy, "format": "RGB888"},
                    controls={'FrameRate': 90, "FrameDurationLimits": (22222, 33333)}, # ex FrameDurationLimits:  24fps = 1000000/24 = 41667
                    buffer_count=1)#, controls={"FrameDurationLimits": (22222, 33333)})
        self.picam2.configure(_config)
        #  set_controls must come after config!!
        self.picam2.set_controls({"AwbEnable": 0})
        #self.picam2.set_controls({"AeEnable": 0})
        self.picam2.set_controls({"AeMeteringMode": controls.AeMeteringModeEnum.Spot})
        self.picam2.set_controls({"AnalogueGain": 6.0})
        #self.picam2.set_controls({"ExposureTime": 1000000}) # for blurring - but can get over exposed at night
        #self.picam2.set_controls({"FrameDurationLimits": (1000,1000)})
        #self.picam2.set_controls({"ExposureTime": 100000000, "AnalogueGain": 1.0})
        #self.picam2.video_configuration.controls.FrameRate = 90
        self.picam2.start()
        time.sleep(0.2)

    def _get_image(self):
        output = self.picam2.capture_array("main")
        return output
    def get_raw_image(self):
        """Get full uncropped raw image from source. Must be implemented by subclasses."""
        raise Exception("raw image not implemented!")

class Scamblight_Camera_Async(Camera_async):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ScambilightCamImageGen)


class Scamblight_Camera_Async_buffer(Camera_async_buffer):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ScambilightCamImageGen)


class Synth_Camera_Async_buffer(Camera_async_buffer):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ImageLibrary)

class Synth_Camera_Async(Camera_async):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ImageLibrary)

class Container_Camera(Camera_synchronous):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ImageLibrary)


class Synth_Camera_sync_buffer(Camera_synchronous_with_buffer):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ImageLibrary)

class Scambi_Camera_sync_buffer(Camera_synchronous_with_buffer):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ScambilightCamImageGen_fps_test)
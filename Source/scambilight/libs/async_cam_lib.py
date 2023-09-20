
import os

# factory is imported from another directory by module load
from factory import Camera_async, Camera_synchronous, ImageGenerator
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


if get_platform() == _OS.RASPBERRY:
    # sorry not sorry
    from picamera2 import Picamera2



class FinishedProcess():
    finished = True


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
        while True:
            raise Exception("this has to be updated for new scamiprocess code")
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
        imgfoler = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.images = jpgs_in_folder(imgfoler)
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
        #self.picam2.set_controls({"ExposureTime": 100000000, "AnalogueGain": 1.0})
        self.picam2.start()
        time.sleep(0.2)

    def get_image(self):
        output = self.picam2.capture_array("main")
        return output


class Scamblight_Camera_Async(Camera_async):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ScambilightCamImageGen)


class Synth_Camera_Async(Camera_async):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ImageLibrary)


class Synth_Camera_sync(Camera_synchronous):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, ImageLibrary)

from dataclasses import dataclass, asdict
from logging import exception
from multiprocessing import ProcessError
import subprocess
import os
from datetime import datetime
import time
import re
from subprocess import Popen, PIPE
from os import kill
from signal import SIGKILL
import enum
import json
from urllib.request import urlopen
import itertools
import socket
import cv2
import numpy as np
import enum
import RPi.GPIO as GPIO
import time
import decode_clothID_v1 as decode_clothID

#from tomlkit import datetime

class lumogun_state():
    def __init__(self) -> None:
        self.long_cam_vidmodes = [e.value for e in HQ_Cam_vidmodes]
        self.long_cam_mode = 0
        self.chn1_debounce = Trigger()
        self.chn2_debounce = Trigger()
        self.chn3_debounce = Trigger()
        self.relay_chn1 = 29
        self.relay_chn2 = 31
        self.relay_chn3 = 16
        self.trigger_rear = 15
        self.trigger_front = 13



    def next_long_vid_mode(self):
        self.long_cam_mode += 1
        if self.long_cam_mode > len(self.long_cam_vidmodes) -1:
            self.long_cam_mode = 0

    @property
    def long_vid_res(self):
        return self.long_cam_vidmodes[self.long_cam_mode][1]


class screensizes(enum.Enum):
    desktop_os_opencv = (740,480)

class libcam_commands(enum.Enum):
    basic_rotated_for_preview = "libcamera-hello -t 0 --vflip --hflip"
    basic_high_gain = "libcamera-hello -t 0 --shutter 20000 --gain 4.5 --vflip --hflip"
    basic_HD_vid = "libcamera-vid -t0  --width 1920 --height 1080 --vflip --hflip"
    basic_HD_vid_2 = "libcamera-vid -t0  --width 2028 --height 1520 --vflip --hflip" # doesnt work
    basic_HD_vid_crop = "libcamera-vid -t0  --width 1332 --height 990 --vflip --hflip"
    # for low latency RTMP streams use:
    # ffplay -f live_flv -fast -fflags nobuffer -flags low_delay -strict experimental rtmp://liell-VirtualBox.local/streams/cam
    stream_locally = "libcamera-vid -t 0 --width 1920 --height 1080 --framerate 30 --codec h264 --inline -o - | ffmpeg -i pipe:0 -an -c:v copy -f flv rtmp://liell-VirtualBox.local/streams/cam" #playback rtmp://liell-VirtualBox.local/streams/cam (this is bouncing off nginx with rtmp module), thjis works on phone but need way of turning off buffer using -fflags nobuffer, not possible with standard vlc app
    stream_locally_udp = "libcamera-vid -t 0 --width 1920 --height 1080 --framerate 30 --codec h264 --inline -o - | ffmpeg -i pipe:0 -an -c:v copy -f mpegs -f mpegts udp://LiellOMEN.local:1234" # playback ffplay -fflags nobuffer udp://127.0.0.1:1234
    stream_locally_mobile = "libcamera-vid -t 0 --width 640 --height 480 --framerate 30 --codec h264 --inline -o - | ffmpeg -i pipe:0 -an -c:v copy -f flv rtmp://liell-VirtualBox.local/streams/cam" #playback rtmp://liell-VirtualBox.local/streams/cam (this is bouncing off nginx with rtmp module), thjis works on phone but need way of turning off buffer using -fflags nobuffer, not possible with standard vlc app
    stream_locally_udp_low_latency = "libcamera-vid -t 0 --width 1920 --height 1080 --framerate 30 --codec h264 --inline -o - | ffmpeg -i pipe:0 -fflags nobuffer -an -c:v copy -f mpegs -f mpegts udp://LiellOMEN.local:1234" # playback ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental -probesize 32 -analyzeduration 0 udp://127.0.0.1:1234
    legacy_camera_1 = "raspividyuv --output - --timeout 0 --framerate 30 --luma --nopreview "
    #stream_locally_raw = " libcamera-vid -t 0 --width 640 --height 480 --framerate 30 --codec yuv420 --codec h264 --inline -o - | ffmpeg -i pipe:0 -an -c:v copy -f mpegs -f mpegts udp://LiellOMEN.local:1234"

class HQ_Cam_vidmodes(enum.Enum):
    _4 = ["640 × 480",(640, 480)] #0.3MP
    _2 = ["2028 × 1080p50,",(2020, 1080)] # 2.0MP  this is not losing res -  turn camera 90 degrees - probably want this one
    _3 = ["1332 × 990p120",(1332, 990)] 
    _1 = ["2028 × 1520p40",(2020, 1520)]


    #_4 = ["640 × 480",(640, 480)]

def ImageViewer_Quick_no_resize(inputimage,pausetime_Secs=0,presskey=False,destroyWindow=True):
    try:
        ###handy quick function to view images with keypress escape andmore options
        cv2.imshow("img", inputimage.copy()); 
        cv2.moveWindow("img", 0, 0)
        if presskey==True:
            cv2.waitKey(0); #any key
    
        if presskey==False:
            if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                    #for some reason
                    pass
                
        if pausetime_Secs>0:
            time.sleep(pausetime_Secs)
        if destroyWindow==True: cv2.destroyAllWindows()
    except Exception as e:
        print(e)

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

def start_subprocess(command):

    if command is None or command == "":
        pipeline = 'no command'
    else:
        pipeline=command
    p = subprocess.Popen(pipeline, shell=True)
    return p

def rectangle_animate_step(imgshape, version = "", stepsize = 10):
    _step = stepsize
    
    while True:
        if _step > 100:
            break
        empty_frame = np.ones(imgshape, dtype=np.uint8)
        empty_frame=empty_frame*255
        cv2.rectangle(
            empty_frame,
             (_step, _step),
              (imgshape[1]-_step,imgshape[0]-_step),
               (0, 0, 255),
               2
               )
        cv2.putText(empty_frame, f"LUMOTAG {version}" , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        _step += stepsize
        yield empty_frame.copy()

def exceptionwindow(exceptiontext, imgshape):
    today = datetime.now()
    dt_string = today.strftime("%d/%m/%Y %H:%M:%S")
    empty_frame = np.ones(imgshape, dtype=np.uint8)
    empty_frame=empty_frame*255
    _step = 10
    cv2.rectangle(
        empty_frame,
            (_step, _step),
            (imgshape[1]-_step,imgshape[0]-_step),
            (0, 0, 255),
            2
            )
    cv2.putText(empty_frame, f"INFO: {dt_string} {exceptiontext}" , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return empty_frame

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

class Trigger:
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

def clean_up_processes(cmds, rec_depth=0):
    rec_depth += 1
    if rec_depth > 10:
        raise RecursionError(
            "cannot clean up previous session streaming processes")
    process = Popen(['ps', '-eo', 'pid,args'], stdout=PIPE, stderr=PIPE)
    stdout, _ = process.communicate()
    for line in stdout.splitlines():
        match_list = re.findall(cmds, str(line))
        if len(match_list) > 0:
            print(f"PROCESS {str(line)}")
            pid = int(str(line).split()[1])
            print(f"PID {pid}")
            kill(pid, SIGKILL)
            time.sleep(1)
            clean_up_processes(cmds, rec_depth)
            break


#if False:
#    if False:
#	anim_rectangle = rectangle_animate_step(imgshape=screensizes.desktop_os_opencv.value,version="2")
#	[ImageViewer_Quick_no_resize(_,0.2,False,False) for _ in anim_rectangle]
#
#
#
#	# Initialize the camera
#	camera = PiCamera()
#	 
#	# Set the camera resolution
#	camera.resolution = (640, 480)
#	 
#	# Set the number of frames per second
#	camera.framerate = 32
#	 
#	# Generates a 3D RGB array and stores it in rawCapture
#	raw_capture = PiRGBArray(camera, size=(640, 480))
#	 
#	# Wait a certain number of seconds to allow the camera time to warmup
#	time.sleep(1)
#	 
#	# Capture frames continuously from the camera
#	for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
#	     
#	    # Grab the raw NumPy array representing the image
#	    image = frame.array
#	     
#	    # Display the frame using OpenCV
#	    cv2.imshow("Frame", image)
#	     
#	    # Wait for keyPress for 1 millisecond
#	    key = cv2.waitKey(1) & 0xFF
#	     
#	    # Clear the stream in preparation for the next frame
#	    raw_capture.truncate(0)
#	     
#	    # If the `q` key was pressed, break from the loop
#	    if key == ord("q"):
#		break





def use_preview_output_loop():
    start_subprocess(libcam_commands.stream_locally_udp.value)
    while True:
        test_inputs()

def text_on_image(inputimage, text):
    """draw text in a fixed format and position"""
    cv2.putText(
        inputimage,
        text,
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=2)
    return inputimage


def test_yuv(lumostate : lumogun_state):
    from picamera2 import Picamera2, Preview
    import time
    # Configure camera
    picam2 = Picamera2()
    ### Take IR image in YUV format
    capture_config = picam2.create_video_configuration(main = {"size": (2028, 1080), "format": "YUV420"}) #2028 × 1080
    #capture_config = picam2.create_still_configuration()
    picam2.configure(capture_config)
    picam2.start()
    while True:
        trigs = test_inputs(lumostate)
        output = picam2.capture_array("main")
        output = output[0:1080, 0:2028]
        output=cv2.resize(output,tuple((screensizes.desktop_os_opencv.value)))
        output = cv2.normalize(output, output,0, 255, cv2.NORM_MINMAX)
        output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
        ImageViewer_Quick_no_resize(output,0,False,False)

def use_process_loop(lumostate : lumogun_state):
    #anim_rectangle = rectangle_animate_step(imgshape=screensizes.desktop_os_opencv.value,version="2")
    #[ImageViewer_Quick_no_resize(_,0.2,False,False) for _ in anim_rectangle]
    #anim_rectangle = rectangle_animate_step(imgshape=screensizes.desktop_os_opencv.value,version=" fuk picam2")
    #[ImageViewer_Quick_no_resize(_,0.2,False,False) for _ in anim_rectangle]

    from picamera2 import Picamera2, Preview
    import time
    picam2 = Picamera2()

    #https://stackoverflow.com/questions/74075544/how-to-capture-raspberry-pi-hq-camera-data-in-yuv-format-using-picamera2
    while True:
        #2028 × 1080p50, 2028 × 1520p40 and 1332 × 990p120
        #camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
        #picam2.create_video_configuration()["controls"]{'NoiseReductionMode': <NoiseReductionMode.Fast: 1>, 'FrameDurationLimits': (33333, 33333)}
        config = picam2.create_video_configuration(main={"size": lumostate.long_vid_res})#, controls={"FrameDurationLimits": (233333, 233333)})
        picam2.configure(config)
        picam2.start()
        time.sleep(0.1)

        while True:
            trigs = test_inputs()
            times = []
            perf_strings = ""
            if trigs[2] is True:
                lumostate.next_long_vid_mode()
                break
            try:
                print("trying to get image")
                times.append((time.perf_counter(),"start"))
                output = picam2.capture_array("main")
                times.append((time.perf_counter()-times[-1][0],"get output"))
                array=cv2.resize(output,tuple(reversed(screensizes.desktop_os_opencv.value)))
                times.append((time.perf_counter()-times[-1][0],"resize once"))
                #array = text_on_image(array, str(trigs))
                array = text_on_image(array, f"{output.shape}")
                array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
                array = cv2.normalize(array, array,0, 255, cv2.NORM_MINMAX)
                array = cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
                array = cv2.resize(array,tuple(reversed(screensizes.desktop_os_opencv.value)))
                times.append((time.perf_counter()-times[-1][0],"various functions"))
                #array = cv2.cvtColor(array, cv2.COLORMAP_RAINBOW)
                ImageViewer_Quick_no_resize(array,0,False,False)
                times.append((time.perf_counter()-times[-1][0],"image viewer"))
                # for tm, funct in times:
                #     perf_strings = perf_strings + f"{funct}:{tm}"
                # cv2.putText(
                #             array,
                #             perf_strings,
                #             (200, 80),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=1,
                #             color=(0, 0, 255),
                #             thickness=1.5)
            except Exception as e:
                print(e)
                except_img = exceptionwindow(e, imgshape=tuple(reversed(screensizes.desktop_os_opencv.value)))
                ImageViewer_Quick_no_resize(except_img,0,False,False)

        picam2.stop()

def take_image(lumostate : lumogun_state):
        #anim_rectangle = rectangle_animate_step(imgshape=screensizes.desktop_os_opencv.value,version="2")
    #[ImageViewer_Quick_no_resize(_,0.2,False,False) for _ in anim_rectangle]
    #anim_rectangle = rectangle_animate_step(imgshape=screensizes.desktop_os_opencv.value,version=" fuk picam2")
    #[ImageViewer_Quick_no_resize(_,0.2,False,False) for _ in anim_rectangle]
    print("entering loop")
    from picamera2 import Picamera2, Preview
    import time
    picam2 = Picamera2()

    #https://stackoverflow.com/questions/74075544/how-to-capture-raspberry-pi-hq-camera-data-in-yuv-format-using-picamera2
    while True:
        #2028 × 1080p50, 2028 × 1520p40 and 1332 × 990p120
        #camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
        #picam2.create_video_configuration()["controls"]{'NoiseReductionMode': <NoiseReductionMode.Fast: 1>, 'FrameDurationLimits': (33333, 33333)}
        config = picam2.create_video_configuration(main={"size": lumostate.long_vid_res})#, controls={"FrameDurationLimits": (233333, 233333)})
        picam2.configure(config)
        picam2.start()
        time.sleep(0.1)
        output = None
        while True:
            trigs = test_inputs(lumostate)
            times = []
            perf_strings = ""
            if trigs[2] is True:
                if output is not None:
                    now_ns = time.time_ns()
                    cv2.imwrite(f"/home/lumotag/{now_ns}.jpg",output.copy())
            try:
                print("trying to get image")
                times.append((time.perf_counter(),"start"))
                output = picam2.capture_array("main")
                times.append((time.perf_counter()-times[-1][0],"get output"))
                array=cv2.resize(output,tuple(reversed(screensizes.desktop_os_opencv.value)))
                times.append((time.perf_counter()-times[-1][0],"resize once"))
                #array = text_on_image(array, str(trigs))
                array = text_on_image(array, perf_strings)#f"{output.shape}")
                array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
                array = cv2.normalize(array, array,0, 255, cv2.NORM_MINMAX)
                array = cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
                array = cv2.resize(array,tuple(reversed(screensizes.desktop_os_opencv.value)))
                times.append((time.perf_counter()-times[-1][0],"various functions"))
                #array = cv2.cvtColor(array, cv2.COLORMAP_RAINBOW)
                ImageViewer_Quick_no_resize(array,0,False,False)
                times.append((time.perf_counter()-times[-1][0],"image viewer"))
                # for tm, funct in times:
                #     perf_strings = perf_strings + f"{funct}:{tm}"
                # cv2.putText(
                #             array,
                #             perf_strings,
                #             (200, 80),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=1,
                #             color=(0, 0, 255),
                #             thickness=1.5)
            except Exception as e:
                print(e)
                except_img = exceptionwindow(e, imgshape=tuple(reversed(screensizes.desktop_os_opencv.value)))
                ImageViewer_Quick_no_resize(except_img,0,False,False)

        picam2.stop()
           
def decode_pattern(lumostate : lumogun_state):
         #anim_rectangle = rectangle_animate_step(imgshape=screensizes.desktop_os_opencv.value,version="2")
    #[ImageViewer_Quick_no_resize(_,0.2,False,False) for _ in anim_rectangle]
    #anim_rectangle = rectangle_animate_step(imgshape=screensizes.desktop_os_opencv.value,version=" fuk picam2")
    #[ImageViewer_Quick_no_resize(_,0.2,False,False) for _ in anim_rectangle]
    from picamera2.encoders import Encoder
    workingdata_decodetag =decode_clothID.WorkingData()
    workingdata_decodetag.debug= False
    no_res = np.zeros((50,50,3), np.uint8)
    no_res[:,:,2] = 255
    good_res = np.zeros((50,50,3), np.uint8)
    good_res[:,:,1] = 255
    from picamera2 import Picamera2, Preview
    picam2 = Picamera2()

    #https://stackoverflow.com/questions/74075544/how-to-capture-raspberry-pi-hq-camera-data-in-yuv-format-using-picamera2
    while True:
        ImageViewer_Quick_no_resize(exceptionwindow("start lumotag YO",screensizes.desktop_os_opencv.value),2,False,True)
        #2028 × 1080p50, 2028 × 1520p40 and 1332 × 990p120
        #camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
        #picam2.create_video_configuration()["controls"]{'NoiseReductionMode': <NoiseReductionMode.Fast: 1>, 'FrameDurationLimits': (33333, 33333)}
        config = picam2.create_video_configuration(main={"size": lumostate.long_vid_res,  "format": "YUV420"})#, controls={"FrameDurationLimits": (233333, 233333)})
    
        #config = picam2.create_video_configuration(raw={}, encode="raw")#
        picam2.set_controls({"ExposureTime": 10000}) # for blurring - but can get over exposed at night
        picam2.configure(config)
        picam2.start()
        time.sleep(0.1)
        output = None
        lumotags_found = None
        while True:
            with decode_clothID.time_it():
                trigs = test_inputs(lumostate)
                print("input test time")
            if trigs[2] is True:
                #do we want to take the image before or after?
                output = picam2.capture_array("main")
                
                (x, y) = lumostate.long_vid_res#  need to do this for YUV!
                output = output[0:y, 0:x]#  need to do this for YUV!
                if output is None:
                    continue
                #try:
                output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
                with decode_clothID.time_it():
                    lumotags_found, playerfound = decode_clothID.find_lumotag(output, workingdata_decodetag)
                    print("decode pattern time")
                if playerfound is True:
                    trigger_res = cv2.resize(good_res,tuple(reversed(screensizes.desktop_os_opencv.value)))
                    ImageViewer_Quick_no_resize(trigger_res,0,False,False)
                else:
                    trigger_res = cv2.resize(no_res,tuple(reversed(screensizes.desktop_os_opencv.value)))
                    ImageViewer_Quick_no_resize(trigger_res,0,False,False)

                #     lumotags_found = cv2.resize(lumotags_found,tuple(reversed(screensizes.desktop_os_opencv.value)))
                # except Exception as e:
                #     lumotags_found = None
                #     ImageViewer_Quick_no_resize(exceptionwindow(e,screensizes.desktop_os_opencv.value),2,False,True)
                # if lumotags_found is not None:
                #     ImageViewer_Quick_no_resize(lumotags_found,0,False,False)
                # else:
                #     ImageViewer_Quick_no_resize(output,0,False,False)

                #     continue
            #     continue
            #     if output is not None:
            #         now_ns = time.time_ns()
            #         try:
            #             cv2.imwrite(f"/home/lumotag/{now_ns}.jpg",lumotags_found)
            #             lumotags_found = decode_clothID.find_lumotag(output.copy(), workingdata_decodetag)
            #             if lumotags_found is not None:
            #                 lumotags_found = cv2.rotate(lumotags_found, cv2.ROTATE_90_CLOCKWISE)
            #                 lumotags_found = cv2.resize(lumotags_found,tuple(reversed(screensizes.desktop_os_opencv.value)))
            #                 cv2.imwrite(f"/home/lumotag/{now_ns}.jpg",lumotags_found)
            #                 ImageViewer_Quick_no_resize(lumotags_found,0.3,False,False)
            #             else:
            #                 out = np.zeros_like(cv2.cvtColor(output, cv2.COLOR_GRAY2RGB))
            #                 out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
            #                 out = cv2.resize(out,tuple(reversed(screensizes.desktop_os_opencv.value)))
            #                 ImageViewer_Quick_no_resize(out,0.3,False,False)
                            
            #         except Exception as e:
            #             ImageViewer_Quick_no_resize(exceptionwindow(str(e),screensizes.desktop_os_opencv.value),2,False,True)
            #             pass #BAD

            
            #try:
                #print("trying to get image")
            with decode_clothID.time_it():
                output = picam2.capture_array("main") # 
                
                (x, y) = lumostate.long_vid_res#  Need to do this for YUV!
                output = output[0:y, 0:x]#  Need to do this for YUV!
                print("image capture time")
            with decode_clothID.time_it():
                
                #  have to chop out luminance part of YUV format image
                
                output=cv2.resize(output,tuple((screensizes.desktop_os_opencv.value)))
                output = cv2.normalize(output, output,0, 255, cv2.NORM_MINMAX)
                output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
                output = cv2.cvtColor(output,cv2.COLOR_GRAY2BGR)
                print("image prepare time time") # 17 ms max res

                if lumotags_found is not None:
                    mini_latch = cv2.resize(lumotags_found,(300,300))
                    output[0:300,0:300] = mini_latch
            with decode_clothID.time_it():
                ImageViewer_Quick_no_resize(output,0,False,False)
                print("image display time")
                

            # except Exception as e:
            #     print(e)
            #     except_img = exceptionwindow(e, imgshape=tuple(reversed(screensizes.desktop_os_opencv.value)))
            #     except_img = cv2.rotate(except_img, cv2.ROTATE_90_CLOCKWISE)
            #     ImageViewer_Quick_no_resize(except_img,0,False,False)
def decode_pattern_speedup(lumostate : lumogun_state):
         #anim_rectangle = rectangle_animate_step(imgshape=screensizes.desktop_os_opencv.value,version="2")
    #[ImageViewer_Quick_no_resize(_,0.2,False,False) for _ in anim_rectangle]
    #anim_rectangle = rectangle_animate_step(imgshape=screensizes.desktop_os_opencv.value,version=" fuk picam2")
    #[ImageViewer_Quick_no_resize(_,0.2,False,False) for _ in anim_rectangle]
    from picamera2.encoders import Encoder
    workingdata_decodetag =decode_clothID.WorkingData()
    workingdata_decodetag.debug= False
    no_res = np.zeros((50,50,3), np.uint8)
    no_res[:,:,2] = 255
    good_res = np.zeros((50,50,3), np.uint8)
    good_res[:,:,1] = 255
    from picamera2 import Picamera2, Preview
    picam2 = Picamera2()

    #https://stackoverflow.com/questions/74075544/how-to-capture-raspberry-pi-hq-camera-data-in-yuv-format-using-picamera2
    while True:
        ImageViewer_Quick_no_resize(exceptionwindow("start lumotag YO",screensizes.desktop_os_opencv.value),2,False,True)
        #2028 × 1080p50, 2028 × 1520p40 and 1332 × 990p120
        #camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
        #picam2.create_video_configuration()["controls"]{'NoiseReductionMode': <NoiseReductionMode.Fast: 1>, 'FrameDurationLimits': (33333, 33333)}
        #config = picam2.create_video_configuration(main={"size": lumostate.long_vid_res}, raw={"format": "SRGGB12_CSI2P"}, display=None)#, controls={"FrameDurationLimits": (233333, 233333)})
        config = picam2.create_video_configuration(main={"size": lumostate.long_vid_res},display=None)#, controls={"FrameDurationLimits": (233333, 233333)})
          
        #config = picam2.create_video_configuration(raw={}, encode="raw")#
        picam2.set_controls({"ExposureTime": 10000})#,"size": (4056, 3040)
        picam2.set_controls({"ScalerCrop": [200,200,200,200]})
        picam2.configure(config)
        picam2.start()
        time.sleep(0.1)
        output = None
        lumotags_found = None
        while True:
            with decode_clothID.time_it():
                trigs = test_inputs(lumostate)
                print("input test time")
            if trigs[2] is True:
                lumostate.next_long_vid_mode()
                picam2.stop()
                break
                #do we want to take the image before or after?
                output = picam2.capture_array("main")
                if output is None:
                    continue
                #try:
                output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
                with decode_clothID.time_it():
                    lumotags_found, playerfound = decode_clothID.find_lumotag(output, workingdata_decodetag)
                    print("decode pattern time")
                if playerfound is True:
                    trigger_res = cv2.resize(good_res,tuple(reversed(screensizes.desktop_os_opencv.value)))
                    ImageViewer_Quick_no_resize(trigger_res,0,False,False)
                else:
                    trigger_res = cv2.resize(no_res,tuple(reversed(screensizes.desktop_os_opencv.value)))
                    ImageViewer_Quick_no_resize(trigger_res,0,False,False)

                #     lumotags_found = cv2.resize(lumotags_found,tuple(reversed(screensizes.desktop_os_opencv.value)))
                # except Exception as e:
                #     lumotags_found = None
                #     ImageViewer_Quick_no_resize(exceptionwindow(e,screensizes.desktop_os_opencv.value),2,False,True)
                # if lumotags_found is not None:
                #     ImageViewer_Quick_no_resize(lumotags_found,0,False,False)
                # else:
                #     ImageViewer_Quick_no_resize(output,0,False,False)

                #     continue
            #     continue
            #     if output is not None:
            #         now_ns = time.time_ns()
            #         try:
            #             cv2.imwrite(f"/home/lumotag/{now_ns}.jpg",lumotags_found)
            #             lumotags_found = decode_clothID.find_lumotag(output.copy(), workingdata_decodetag)
            #             if lumotags_found is not None:
            #                 lumotags_found = cv2.rotate(lumotags_found, cv2.ROTATE_90_CLOCKWISE)
            #                 lumotags_found = cv2.resize(lumotags_found,tuple(reversed(screensizes.desktop_os_opencv.value)))
            #                 cv2.imwrite(f"/home/lumotag/{now_ns}.jpg",lumotags_found)
            #                 ImageViewer_Quick_no_resize(lumotags_found,0.3,False,False)
            #             else:
            #                 out = np.zeros_like(cv2.cvtColor(output, cv2.COLOR_GRAY2RGB))
            #                 out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
            #                 out = cv2.resize(out,tuple(reversed(screensizes.desktop_os_opencv.value)))
            #                 ImageViewer_Quick_no_resize(out,0.3,False,False)
                            
            #         except Exception as e:
            #             ImageViewer_Quick_no_resize(exceptionwindow(str(e),screensizes.desktop_os_opencv.value),2,False,True)
            #             pass #BAD

            
            try:
                #print("trying to get image")
                # with decode_clothID.time_it():
                #     request = picam2.capture_request()
                #     #request.save("main", "image.jpg")
                #     output = request.make_array("main")
                #     request.release()
                #     print("borrow buffer time")
                with decode_clothID.time_it():
                    output = picam2.capture_array("main") # 90 ms on pi max res!
                    print("image capture time")
                crop_in = 200
                output = output[crop_in:output.shape[1]-crop_in, crop_in:output.shape[0]-crop_in,:]
                with decode_clothID.time_it():
                    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                    output=cv2.resize(output,tuple((screensizes.desktop_os_opencv.value)))
                    output = cv2.normalize(output, output,0, 255, cv2.NORM_MINMAX)
                    output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
                    print("image prepare time time") # 17 ms max res

                    if lumotags_found is not None:
                        mini_latch = cv2.resize(lumotags_found,(300,300))
                        output[0:300,0:300] = cv2.cvtColor(mini_latch,cv2.COLOR_BGR2GRAY)
                with decode_clothID.time_it():
                    ImageViewer_Quick_no_resize(output,0,False,False)
                    print("image display time")
                    

            except Exception as e:
                print(e)
                except_img = exceptionwindow(e, imgshape=tuple(reversed(screensizes.desktop_os_opencv.value)))
                except_img = cv2.rotate(except_img, cv2.ROTATE_90_CLOCKWISE)
                ImageViewer_Quick_no_resize(except_img,0,False,False)

def unknown_loop():
    picam2.start_preview(Preview.QTGL)

    while True:
        time.sleep(1)
    # time.sleep(99999999)
    picam2 = Picamera2()
    picam2.start()
    #picam2.configure(picam2.create_preview_configuration())
    #picam2.set_controls({"ExposureTime": 10000, "AnalogueGain": 10.0})#,"size": (4056, 3040)
    time.sleep(1)

    while True:
        array = picam2.capture_array("main")
        #cv2.putText(array, str(array.shape), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        #array = cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
        ImageViewer_Quick_no_resize(array,0,False,False)

def test_inputs(lumostate : lumogun_state):
    
    outputs = [None,False,False]

    if GPIO.input(lumostate.trigger_rear) == GPIO.LOW:

        #chn1_debounce.trigger(GPIO.output, relay_chn1, GPIO.HIGH)

        #trig = chn2_debounce.trigger(GPIO.output, relay_chn2, GPIO.HIGH)

        #trig1 = chn3_debounce.trigger(GPIO.output, relay_chn3, GPIO.HIGH)

        res1 = lumostate.chn1_debounce.trigger(GPIO.output, lumostate.relay_chn1, GPIO.HIGH)

        res2 = lumostate.chn3_debounce.trigger(GPIO.output, lumostate.relay_chn3, GPIO.HIGH)

        outputs[1] = all([res1, res2])

    else:

        #chn1_debounce.trigger(GPIO.output, relay_chn1, GPIO.LOW)

        lumostate.chn1_debounce.trigger(GPIO.output, lumostate.relay_chn1, GPIO.LOW)

        lumostate.chn3_debounce.trigger(GPIO.output, lumostate.relay_chn3, GPIO.LOW)

        #chn3_debounce.trigger(GPIO.output, relay_chn3, GPIO.LOW)
    if GPIO.input(lumostate.trigger_front) == GPIO.LOW:

       outputs[2] = lumostate.chn2_debounce.trigger(GPIO.output, lumostate.relay_chn2, GPIO.HIGH)

    else:

       lumostate.chn2_debounce.trigger(GPIO.output, lumostate.relay_chn2, GPIO.LOW)
    return outputs

#use_process_loop(lumostate)
def startlumoing():
        
    clean_up_processes(cmds=r'ffmpeg|libcamera')

    lumostate = lumogun_state()

    # Seems that all 3 on at same time exceeds
    # some kind of current draw limit and the
    # relay channels don't activate properly
    GPIO.setmode(GPIO.BOARD) # pin out corresponds to board id, not BCM id
    GPIO.setup(lumostate.relay_chn1, GPIO.OUT)
    GPIO.setup(lumostate.relay_chn2, GPIO.OUT)
    GPIO.setup(lumostate.relay_chn3, GPIO.OUT)
    GPIO.setup(lumostate.trigger_rear, GPIO.IN, pull_up_down=GPIO.PUD_UP)    # Set BtnPin's mode is input, and pull up to high level(3.3V)
    GPIO.setup(lumostate.trigger_front, GPIO.IN, pull_up_down=GPIO.PUD_UP)    # Set BtnPin's mode is input, and pull up to high level(3.3V)


    for n in range(0,2):
        GPIO.output(lumostate.relay_chn1, GPIO.HIGH)
        GPIO.output(lumostate.relay_chn2, GPIO.LOW)
        GPIO.output(lumostate.relay_chn3, GPIO.HIGH)
        time.sleep(0.05)
        GPIO.output(lumostate.relay_chn1, GPIO.LOW)
        GPIO.output(lumostate.relay_chn2, GPIO.HIGH)
        GPIO.output(lumostate.relay_chn3, GPIO.LOW)
        time.sleep(0.05)
        GPIO.output(lumostate.relay_chn1, GPIO.HIGH)
        GPIO.output(lumostate.relay_chn2, GPIO.LOW)
        GPIO.output(lumostate.relay_chn3, GPIO.HIGH)
        time.sleep(0.05)
        GPIO.output(lumostate.relay_chn1, GPIO.LOW)
        GPIO.output(lumostate.relay_chn2, GPIO.LOW)
        GPIO.output(lumostate.relay_chn3, GPIO.LOW)
        time.sleep(0.2)

    def rapidfire():
        for n in range(0,5):
            GPIO.output(lumostate.relay_chn2, GPIO.HIGH)
            GPIO.output(lumostate.relay_chn3, GPIO.LOW)
            time.sleep(0.01)
            GPIO.output(lumostate.relay_chn2, GPIO.LOW)
            GPIO.output(lumostate.relay_chn3, GPIO.HIGH)
            time.sleep(0.01)
            GPIO.output(lumostate.relay_chn2, GPIO.LOW)
            GPIO.output(lumostate.relay_chn3, GPIO.LOW)
            time.sleep(0.01)




    #decode_pattern(lumostate)
    decode_pattern(lumostate)
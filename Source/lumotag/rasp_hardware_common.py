import time
import re
from subprocess import Popen, PIPE
import os
from signal import SIGKILL
import cv2
import numpy as np
import enum
import functools
# for finding pinout, type pinout in terminal
import pickle
import time
import factory
# import rabbit_mq
from picamera2 import Picamera2
from libcamera import controls
#accelerometer
# adafruit board library forces BCM mode!!
import board
import digitalio
import busio
import adafruit_lis3dh
import json
import img_processing
import utils
import video_recorder
from configs import HQ_Cam_vidmodes, HQ_GS_Cam_vidmodes, RPICAMv2Noir_Cam_vidmodes
#import imutils


class filesystem(factory.filesystem):
    

    def __init__(self) -> None:
        # have to do it here otherwise permission error 
        # because linux is a retarded cunt vvvvvv
        self.images_folder = "/home/lumotag"
        if not os.path.isdir(self.images_folder):
            os.mkdir(self.images_folder)

    def save_image(self, img, message=""):
        ts = utils.get_epoch_timestamp()
        filename = f"{self.images_folder}/{message}{ts}.jpg"
        cv2.imwrite(
            filename,
            img)
        print(f"saved image {filename}")

    def save_barcodepair(self, result:list, message = ""):
        ts = utils.get_epoch_timestamp()
        filename = f"{self.images_folder}/{message}{ts}.pck"
        all_results = []
        for res in result:
            all_results.append(res._2d_samples)
        with open(filename, 'wb') as file:
            pickle.dump(all_results, file)
        with open(filename, 'rb') as file:
            check_data= pickle.load(file)

    
def lumo_viewer(
        inputimage,
        move_windowx,
        move_windowy,
        pausetime_Secs=0,
        presskey=False,
        destroyWindow=True):
    try:


        window_name = "img"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(window_name, inputimage)
        cv2.moveWindow(window_name, move_windowx, move_windowy)
        if presskey==True:
            cv2.waitKey(0); #any key
    
        if presskey==False:
            if cv2.waitKey(1) & 0xFF == 27:
                    pass
        if pausetime_Secs>0:
            time.sleep(pausetime_Secs)
        if destroyWindow==True: cv2.destroyAllWindows()

    except Exception as e:
        print(e)


class Accelerometer(factory.Accelerometer):
    def __init__(self) -> None:
        super().__init__()
        self._disp_val_lim_max = 9
        self._disp_val_lim_min = -9
        #using l2c not spi!!
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.int1 = digitalio.DigitalInOut(board.D24)
        self.lis3dh = adafruit_lis3dh.LIS3DH_I2C(
            self.i2c,
            int1=self.int1)

    def update_vel(self):
        x, y, z = self.lis3dh.acceleration
        self._last_xyz = (x, y, z)
        # reverse polarity is to match with
        # LT display - not good place to have it
        self.update_fifo()
        return (
            self.round(x*-1),
            self.round(y*-1),
            self.round(z*-1))
        

    def get_visual(self):
        return super().get_visual()


class display(factory.display):

    def display_method(self, image):

        try:
            # Display the image
            lumo_viewer(
                inputimage=image,
                move_windowx=self.opencv_win_pos[0],
                move_windowy=self.opencv_win_pos[1],
                pausetime_Secs=0,
                presskey=False,
                destroyWindow=False)
        except Exception as e:
            # when SSHing
            print(f"Error in display_method: {e}")

    # def display_output_with_implant(self, main_img, img_to_implant):
    #         """avoid performing higher workload by resizing images to
    #         display size before any rotation or copying """
    #         if self.display_rotate == 0:

    #             img, scale_factor = img_processing.resize_centre_img(
    #                 main_img,
    #                 self.screen_size)
    #             imp_size_x = int(img_to_implant.shape[0] * scale_factor)
    #             imp_size_y = int(img_to_implant.shape[1] * scale_factor)
    #             img_to_implant = cv2.resize(img_to_implant, dsize=(imp_size_x, imp_size_y))
    #             output = img_processing.implant_internal_section(img, img_to_implant)
    #             output = img_processing.add_cross_hair(output, adapt=True)

    #         elif self.display_rotate == -90 or self.display_rotate == 270:
    
    #             img, scale_factor = img_processing.resize_centre_img(
    #                 main_img,
    #                 tuple(reversed(self.screen_size)))
    #             img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #             img_to_implant = cv2.rotate(img_to_implant, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #             imp_size_x = int(img_to_implant.shape[0] * scale_factor)
    #             imp_size_y = int(img_to_implant.shape[1] * scale_factor)
    #             img_to_implant = cv2.resize(img_to_implant, dsize=(imp_size_x, imp_size_y))
    #             output = img_processing.implant_internal_section(img, img_to_implant)
    #             output = img_processing.add_cross_hair(output, adapt=True)
    #             #TODO this is rough - we know this rotation is stryker which is connors
    #             # unit - so for now do connor -specific stuff here although it should
    #             # be in the gun config
    #             font = cv2.FONT_HERSHEY_SIMPLEX
    #             cv2.putText(
    #                 img,
    #                 f'CONNOR',
    #                 (50, 50),
    #                 font,
    #                 1.0,
    #                 (0, 0, 200),
    #                 3,
    #                 cv2.LINE_AA
    #                 )
    #         else:
    #             raise Exception("unhandled screen rotation", self.display_rotate)

    
    #         lumo_viewer(
    #             inputimage=output,
    #             move_windowx=self.opencv_win_pos[0],
    #             move_windowy=self.opencv_win_pos[1],
    #             pausetime_Secs=0,
    #             presskey=False,
    #             destroyWindow=False)


class CsiCameraImageGen_GS(factory.ImageGenerator):
    
    def __init__(self, res) -> None:
        self.cam_res = tuple(reversed(res))
        self.picam2 = Picamera2()
        _config = self.picam2.create_video_configuration(
                    main={"size": res,  "format": "YUV420"}, controls={'FrameRate': 40})# , controls={'FrameRate': 40}, controls={"FrameDurationLimits": (233333, 233333)})
                #self.picam2.set_controls({"ExposureTime": 1000}) # for blurring - but can get over exposed at night
        self.picam2.configure(_config)
        #  set_controls must come after config!!
        self.picam2.set_controls(
            {
                "AwbEnable": 0,
                "AeMeteringMode": controls.AeMeteringModeEnum.Spot,
                "AnalogueGain": 5.0
             }
             )
        self.picam2.start()
        time.sleep(0.2)

    def get_image(self):
        #output = 
        x = self.cam_res[0]
        y = self.cam_res[1]
        #output = output[0: y, 0: x]#  Need to do this for YUV!
        #print("get_image", output.shape, output.dtype)
        yuv_image = self.picam2.capture_array("main")
        
        # Save raw YUV420 as monochrome image for debugging
        # import random
        # if random.randint(0,10) < 2:
        #     timestamp = int(time.time())
        #     cv2.imwrite(f"yuv420_closerange_cnt2104cnt{timestamp}.jpg", yuv_image)
            
        return yuv_image[0: x, 0: y]


class CsiCameraImageGen_GS_test(factory.ImageGenerator):
    
    def __init__(self, res) -> None:
        self.cam_res = tuple(reversed(res))
        # Calculate lores resolution (approximately 1/4 of main resolution)
        lores_width = int(res[0] // 2)
        lores_height = int(res[1] // 2)
        lores_res = (lores_width, lores_height)
        
        self.picam2 = Picamera2()
        _config = self.picam2.create_video_configuration(
                    main={"size": res, "format": "YUV420"}, 
                    lores={"size": lores_res, "format": "YUV420"},
                    controls={'FrameRate': 50})
        # self.picam2.align_configuration(_config) # this helps fix bad alignment for our lowres image
        # self.picam2.configure(_config)

        #  set_controls must come after config!!
        self.picam2.set_controls(
            {
                "AwbEnable": 0,
                "AeMeteringMode": controls.AeMeteringModeEnum.Spot,
                "AnalogueGain": 5.0
             }
             )
        self.picam2.start()
        time.sleep(0.2)

    def get_image(self):
        #output = 
        x = self.cam_res[0]
        y = self.cam_res[1]
        #output = output[0: y, 0: x]#  Need to do this for YUV!
        #print("get_image", output.shape, output.dtype)
        yuv_image = self.picam2.capture_array("main")
        # test_img = self.picam2.capture_array("lores")
        
        # Write debug info to file instead of raising exception
        # debug_info = f"test_img debug info - Type: {type(test_img)}"
        # if hasattr(test_img, 'shape') and hasattr(test_img, 'dtype'):
        #     debug_info += f", Shape: {test_img.shape}, Dtype: {test_img.dtype}"
        # else:
        #     debug_info += " - Not a numpy array"
        
        # # Write to /tmp which should be writable on Raspberry Pi
        # with open('/tmp/test_img_debug.txt', 'w') as f:
        #     f.write(debug_info)
        
        return yuv_image[0: x, 0: y]
    
    def get_lores_image(self):
        """Get the low resolution image stream"""
        lores_x = self.lores_res[1]  # height for YUV420 format
        lores_y = self.lores_res[0]  # width for YUV420 format
        yuv_lores_image = self.picam2.capture_array("lores")
        
        return yuv_lores_image[0: lores_x, 0: lores_y]


class CsiCameraImageGenRCAMv2NOIR(factory.ImageGenerator):
    
    def __init__(self, res) -> None:
        self.cam_res = tuple(reversed(res))
        self.picam2 = Picamera2(1)
        _config = self.picam2.create_video_configuration(
                    main={"size": res,  "format": "YUV420"}, controls={'FrameRate': 40})#, controls={"FrameDurationLimits": (233333, 233333)})
                #self.picam2.set_controls({"ExposureTime": 1000}) # for blurring - but can get over exposed at night
        self.picam2.configure(_config)
        #  set_controls must come after config!!
        #self.picam2.set_controls({"AwbEnable": 0})
        self.picam2.set_controls({
            "AwbEnable": 0, 
            "AeMeteringMode": controls.AeMeteringModeEnum.Spot,
            "AeExposureMode": controls.AeExposureModeEnum.Short,
            "ExposureValue": -2
            })
        #self.picam2.set_controls({"AnalogueGain": 5.0})
        self.picam2.start()
        time.sleep(0.2)

    def get_image(self):
        x = self.cam_res[0]
        y = self.cam_res[1]

        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #  THIS IS CORRECT WAY AROUND!! SIGNED LIELL 4TH OCTOBER!!
        # return self.picam2.capture_array("main")[0: x, 0: y] VVVV MUST BE THE SAME!!
        # IF YOU CHANGE THIS YOUR MOTHER WILL DIE IN HER SLEEP
        # Save raw YUV420 as monochrome image for debugging
        yuv_img = self.picam2.capture_array("main")
        # timestamp = int(time.time())
        # import random
        # if random.randint(0,10) < 2:
        #     cv2.imwrite(f"yuv420_longrange_cnt2104cnt{timestamp}.jpg", yuv_img)
        return yuv_img[0: x, 0: y] # DO not change!!
        # IF YOU CHANGE THIS YOUR MOTHER WILL DIE IN HER SLEEP
        # comes in at shape = (1080, 2020)
        # return self.picam2.capture_array("main")[0: x, 0: y] ^^^ MUST BE THE SAME
        #  THIS IS CORRECT WAY AROUND!! SIGNED LIELL 4TH OCTOBER!!
        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #   DO NOT MODIFY


class CsiCameraImageGen_HQ(factory.ImageGenerator):
    
    def __init__(self, res) -> None:
        self.cam_res = tuple(reversed(res))
        self.picam2 = Picamera2(0)
        _config = self.picam2.create_video_configuration(
                    main={"size": res,  "format": "YUV420"}, controls={'FrameRate': 90})#, controls={"FrameDurationLimits": (233333, 233333)})
                #self.picam2.set_controls({"ExposureTime": 1000}) # for blurring - but can get over exposed at night
        self.picam2.configure(_config)
        #  set_controls must come after config!!
        self.picam2.set_controls({"AwbEnable": 0})
        self.picam2.set_controls({"AnalogueGain": 5.0})
        self.picam2.start()
        time.sleep(0.2)

    def get_image(self):
        x = self.cam_res[0]
        y = self.cam_res[1]

        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #  THIS IS CORRECT WAY AROUND!! SIGNED LIELL 4TH OCTOBER!!
        # return self.picam2.capture_array("main")[0: x, 0: y] VVVV MUST BE THE SAME!!
        # IF YOU CHANGE THIS YOUR MOTHER WILL DIE IN HER SLEEP
        return self.picam2.capture_array("main")[0: x, 0: y] # DO not change!!
        # IF YOU CHANGE THIS YOUR MOTHER WILL DIE IN HER SLEEP
        # comes in at shape = (1080, 2020)
        # return self.picam2.capture_array("main")[0: x, 0: y] ^^^ MUST BE THE SAME
        #  THIS IS CORRECT WAY AROUND!! SIGNED LIELL 4TH OCTOBER!!
        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #   DO NOT MODIFY
        #   DO NOT MODIFY



class CSI_Camera_Async(factory.Camera_async):
    
    def __init__(self, video_modes) -> None:
        if video_modes == HQ_Cam_vidmodes:
            super().__init__(video_modes, factory.ImageLibrary)#CsiCameraImageGen_HQ)
        elif video_modes == HQ_GS_Cam_vidmodes:
            super().__init__(video_modes, factory.ImageLibrary)#CsiCameraImageGen_GS)
        elif video_modes == RPICAMv2Noir_Cam_vidmodes:
            super().__init__(video_modes, CsiCameraImageGenRCAMv2NOIR)
        else:
            raise Exception("no match for video mode input")

class CSI_Camera_async_flipflop(factory.Camera_async_flipflop):
    
    def __init__(self, video_modes) -> None:
        if video_modes == HQ_Cam_vidmodes:
            super().__init__(video_modes, CsiCameraImageGen_HQ)
        elif video_modes == HQ_GS_Cam_vidmodes:
            super().__init__(video_modes, CsiCameraImageGen_GS)
        elif video_modes == RPICAMv2Noir_Cam_vidmodes:
            super().__init__(video_modes, CsiCameraImageGenRCAMv2NOIR)
        else:
            raise Exception("no match for video mode input")

class CSI_Camera_Synchro(factory.Camera_synchronous):

    def __init__(self, video_modes) -> None:
        if video_modes == HQ_Cam_vidmodes:
            super().__init__(video_modes, CsiCameraImageGen_HQ)
        elif video_modes == HQ_GS_Cam_vidmodes:
            super().__init__(video_modes, CsiCameraImageGen_GS)
        elif video_modes == RPICAMv2Noir_Cam_vidmodes:
            super().__init__(video_modes, CsiCameraImageGenRCAMv2NOIR)
        else:
            raise Exception("no match for video mode input")


class KillProcess(factory.KillProcess):
    def clean_up_processes(self, cmds, rec_depth=0):
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
                os.kill(pid, SIGKILL)
                time.sleep(1)
                self.clean_up_processes(cmds, rec_depth)
                break


# Messenger = rabbit_mq.Messenger


def get_my_info(file):
    with open(file, 'r') as file:
        data =  json.load(file)
        MY_ID = data["MY_ID"]

    return MY_ID

import time
import random
from typing import Optional
import cv2
import numpy as np
import time
import factory
import math
# import rabbit_mq
import json
import img_processing
import os
import tempfile
import pickle
from configs import Fake_Cam_vidmodes_longrangeFILES, Fake_Cam_vidmodes_closerangeFILES
import video_recorder
import utils

def lumo_viewer(
        inputimage,
        move_windowx,
        move_windowy,
        pausetime_Secs=0,
        presskey=False,
        destroyWindow=True):
    try:
        cv2.imshow("img", inputimage)
        # cv2.moveWindow("img", move_windowx, move_windowy)
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


class filesystem(factory.FileSystemABC):
    
    def __init__(self) -> None:
        # Use system temp directory - works on Windows, Mac, Linux
        self.temp_folder = tempfile.gettempdir()
        print(f"Fake hardware filesystem using temp folder: {self.temp_folder}")
    
    def save_image(self,img,message=None):
        if img is None:
            raise Exception("save debug; img is None")
        if not isinstance(img, np.ndarray):
            raise Exception("save debug; img is not a numpy array")
        if img.ndim not in (2, 3):
            raise Exception("save debug; img must be a 2D or 3D numpy array")
        
        ts = utils.get_epoch_timestamp()
        filename = os.path.join(self.temp_folder, f"{message}{ts}.jpg")
        cv2.imwrite(filename, img)
        return filename
    def save_barcodepair(self, barcodepair:list, message=None):
        pass

    def save_numberstatus_cache(self, cache_data: dict[str, np.ndarray]) -> bool:
        """Save the numberstatus cache to temporary storage cross-platform."""
        try:
            cache_path = os.path.join(self.temp_folder, "fake_numberstatus_cache.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Fake hardware: saved numberstatus cache to {cache_path}")
            return True
        except Exception as e:
            print(f"Fake hardware: error saving numberstatus cache: {e}")
            return False

    def load_numberstatus_cache(self) -> dict[str, np.ndarray] | None:
        """Load the numberstatus cache from temporary storage cross-platform."""
        try:
            cache_path = os.path.join(self.temp_folder, "fake_numberstatus_cache.pkl")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"Fake hardware: loaded numberstatus cache from {cache_path}")
            return cache_data
        except Exception as e:
            print(f"Fake hardware: error loading numberstatus cache: {e}")
            return None

    def save_shieldstatus_cache(self, cache_data: list[np.ndarray]) -> bool:
        """Save the shieldstatus cache to temporary storage cross-platform."""
        try:
            cache_path = os.path.join(self.temp_folder, "fake_shieldstatus_cache.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Fake hardware: saved shieldstatus cache to {cache_path}")
            return True
        except Exception as e:
            print(f"Fake hardware: error saving shieldstatus cache: {e}")
            return False

    def load_shieldstatus_cache(self) -> list[np.ndarray] | None:
        """Load the shieldstatus cache from temporary storage cross-platform."""
        try:
            cache_path = os.path.join(self.temp_folder, "fake_shieldstatus_cache.pkl")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"Fake hardware: loaded shieldstatus cache from {cache_path}")
            return cache_data
        except Exception as e:
            print(f"Fake hardware: error loading shieldstatus cache: {e}")
            return None

    def _clean_images_folder(self):
        """Clean up all image files in the images folder on startup - fake hardware placeholder."""
        print("Fake hardware: _clean_images_folder called (no-op for fake hardware)")

class Triggers(factory.Triggers):
    def __init__(self, _gun_config) -> None:
        super().__init__(_gun_config)
        self.blink_timer = factory.TimeDiffObject()
        self.flipflop = False
    def test_states(self):
        if self.blink_timer.get_dt() > 2.2:
            self.flipflop = not self.flipflop
            self.blink_timer.reset()
        outputs = {pos:gpio for pos, gpio
                   in self.gun_config.TRIGGER_IO.items()}
        for _, (pos, _) in enumerate(
            self.gun_config.TRIGGER_IO.items()):
            outputs[pos] = self.flipflop
        return outputs

class FakeRelay():
    def __init__(self):
        self.state: bool = False
    
    def on(self):
        self.state = True
    
    def off(self):
        self.state = False

class Relay(factory.Relay):
    def __init__(self, _gun_config) -> None:
        self.fakerelays = {}
        super().__init__(_gun_config)
        

    def getOutputDevice(self, gpio):
        fake_relay = FakeRelay()
        self.fakerelays[gpio] = fake_relay
        return fake_relay


# class CSI_Camera(factory.Camera):

#     def __init__(self, video_modes) -> None:
#         self.cam_res = video_modes
#         # fake input needed for interchangeability
#         super().__init__()

#     def get_res(self):
#         return [e.value for e in self.cam_res][self.res_select][1]

#     def gen_image(self):
#         blank_image = np.zeros((self.get_res() + (3,)), np.uint8)
#         blank_image[:,:,:] = random.randint(0,255)
#         blank_image = cv2.circle(
#             blank_image,
#             (blank_image.shape[1]//2, blank_image.shape[0]//2),
#             blank_image.shape[0]//10,
#             50,
#             -1)

#         return blank_image



class SynthImgGen(factory.ImageGenerator):

    def __init__(self, res) -> None:
        self.blank_image = np.zeros(res, np.uint8)
        self.res = res

    def _get_image(self):
        if len(self.res) == 3:
            self.blank_image[:,:,:] = random.randint(0,255)
        else:
            self.blank_image[:,:] = random.randint(0,255)
        self.blank_image = cv2.circle(
            self.blank_image,
            (self.blank_image.shape[1]//2, self.blank_image.shape[0]//2),
            self.blank_image.shape[0]//10,
            50,
            -1)
        self.blank_image = cv2.circle(
            self.blank_image,
            (self.blank_image.shape[1]//2, self.blank_image.shape[0]//4),
            self.blank_image.shape[0]//30,
            50,
            -1)
        buffer = int(self.blank_image.shape[0]/100)
        self.blank_image = cv2.rectangle(
            self.blank_image,
            (buffer, buffer),
            tuple(np.asarray(list(reversed(self.blank_image.shape[0:2]))) - np.asarray([buffer, buffer])),
            255,
            min(int(buffer/2),2))
        #time.sleep(0.10 * random.random())
        return self.blank_image
    
    def get_raw_image(self):
        """Return a color (3-channel) synthetic image"""
        # Create color image (height, width, 3)
        if len(self.res) == 2:
            # res is (height, width), make it color
            color_res = (self.res[0], self.res[1], 3)
        else:
            # Already has 3 channels
            color_res = self.res
        
        # Create random color image
        color_img = np.zeros(color_res, np.uint8)
        color_img[:,:,0] = random.randint(0, 100)  # Blue channel
        color_img[:,:,1] = random.randint(0, 100)  # Green channel
        color_img[:,:,2] = random.randint(0, 100)  # Red channel
        
        # Draw colored shapes
        color_img = cv2.circle(
            color_img,
            (color_img.shape[1]//2, color_img.shape[0]//2),
            color_img.shape[0]//10,
            (255, 0, 0),  # Blue circle
            -1)
        color_img = cv2.circle(
            color_img,
            (color_img.shape[1]//2, color_img.shape[0]//4),
            color_img.shape[0]//30,
            (0, 255, 0),  # Green circle
            -1)
        buffer = int(color_img.shape[0]/100)
        color_img = cv2.rectangle(
            color_img,
            (buffer, buffer),
            tuple(np.asarray(list(reversed(color_img.shape[0:2]))) - np.asarray([buffer, buffer])),
            (0, 0, 255),  # Red rectangle
            min(int(buffer/2), 2))
        
        return color_img
    
    def set_controls(self, torch_on: bool, controls_override: Optional[dict] = None):
        pass  # no-op for fake hardware


class CSI_Camera_Async(factory.Camera_async):

    def __init__(self, video_modes) -> None:
        super().__init__(
            video_modes=video_modes,
            imagegen_cls=factory.ImageLibrary)


class CSI_Camera_Synchro(factory.Camera_synchronous):
    
    def __init__(self, video_modes) -> None:
        super().__init__(video_modes, factory.ImageLibrary)


class CSI_Camera_async_flipflop(factory.Camera_async_flipflop):
        
    def __init__(self, video_modes) -> None:
        if video_modes == Fake_Cam_vidmodes_longrangeFILES:
            super().__init__(video_modes, factory.ImageLibrary_longrange)
        elif video_modes == Fake_Cam_vidmodes_closerangeFILES:
            super().__init__(video_modes, factory.ImageLibrary_closerange)
        else:
            raise Exception("no match for video mode input")

class CSI_Camera_tribuffer(factory.RingBufferCamera):
        
    def __init__(self, video_modes) -> None:
        if video_modes == Fake_Cam_vidmodes_longrangeFILES:
            super().__init__(video_modes, factory.ImageLibrary_longrange)
        elif video_modes == Fake_Cam_vidmodes_closerangeFILES:
            super().__init__(video_modes, factory.ImageLibrary_closerange)
        else:
            raise Exception("no match for video mode input")


class display(factory.display):

    def display_method(self, image):

        lumo_viewer(
            inputimage=image,
            move_windowx=self.opencv_win_pos[0],
            move_windowy=self.opencv_win_pos[1],
            pausetime_Secs=0,
            presskey=False,
            destroyWindow=False)
    # def display_output(self, output):
    #     img, scale_factor = img_processing.resize_centre_img(output, self.screen_size)
    #     img = img_processing.add_cross_hair(img, adapt=True)
    #     lumo_viewer(output,self.opencv_win_pos[0], self.opencv_win_pos[1],False,False)

    # def display_output_with_implant(self, main_img, img_to_implant):
    #     """resize both images before implantating central graphic as
    #     this can be a significant contribution to update latency"""
    #     img, scale_factor = img_processing.resize_centre_img(
    #          main_img,
    #          self.screen_size)
    #     imp_size_x = int(img_to_implant.shape[0] * scale_factor)
    #     imp_size_y = int(img_to_implant.shape[1] * scale_factor)
    #     img_to_implant = cv2.resize(img_to_implant, dsize=(imp_size_x, imp_size_y))
    #     output = img_processing.implant_internal_section(img, img_to_implant)
    #     output = img_processing.add_cross_hair(output, adapt=True)
    #     lumo_viewer(
    #         inputimage=output,
    #         move_windowx=self.opencv_win_pos[0],
    #         move_windowy=self.opencv_win_pos[1],
    #         pausetime_Secs=0,
    #         presskey=False,
    #         destroyWindow=False)
        

class KillProcess(factory.KillProcess):
    def clean_up_processes(self, cmds, rec_depth=0):
        pass


class Accelerometer(factory.Accelerometer):
    def __init__(self) -> None:
        super().__init__()
        self._x = 1
        self._y = -1
        self._z = 0
        self._callcnt = 0
        self._last_xyz = (0, 0, 0)

    def update_vel(self):
        self._callcnt += 1
        self._x += 0.1
        self._y += 0.1
        self._z += 0.1
        if self._x > 9999999:
            self._x = 0
        if self._y > 9999999:
            self._y = 0
        if self._z > 9999999:
            self._z = 0
        real_accel_range = 30
        self._last_xyz = (
            self.round(math.sin(self._x)*real_accel_range),
            self.round(math.sin(self._y)*real_accel_range),
            self.round(math.sin(self._z)*real_accel_range))
        self.update_fifo()
        return (
            self.round(math.sin(self._x)*real_accel_range),
            self.round(math.sin(self._y)*real_accel_range),
            self.round(math.sin(self._z)*real_accel_range))

# Messenger = rabbit_mq.Messenger

# class Messenger(factory.Messenger):

#     def __init__(self, config) -> None:
#         super().__init__(config=config)
    
#     def _in_box_worker(self, in_box, config, scheduler):
#         cnt = 0
#         while True:
#             cnt += 1
#             time.sleep(4)
#             if in_box._qsize() >= in_box.maxsize:
#                 print("can't push on any more test messages")
#                 continue
#             in_box.put(
#                 msgs.create_test_msg(),
#                 block=False)

#     def _out_box_worker(self, out_box, config, scheduler):
#         while True:
#             message = out_box.get(block=True)
#             print("sending into void", message)

#     def _heartbeat(self, out_box, config):
#         while True:
#             time.sleep(config.msg_heartbeat_s)
#             hb = msgs.create_heartbeat_msg(config)
#             out_box.put(
#                 hb,
#                 block=True)

class GetID(factory.GetID):
    def get_persistant_device_id(self):
        """Get unique and persistant device id - create if does not exist"""
        return "abc12345"


def get_my_info(file):
    id_text_file = '{"MY_ID" : "SIMITZAR", "HQ" : "http://liell-VirtualBox.local/lumoscript.py"}'
    data =  json.loads(id_text_file)
    MY_ID = data["MY_ID"]
    return MY_ID
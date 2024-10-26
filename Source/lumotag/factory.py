from abc import ABC, abstractmethod
import numpy as np
import json
import time
from enum import Enum
from functools import lru_cache
from typing import Literal, Optional
import cv2
import os
import threading
import random
import pickle
#from queue import Queue
import queue
import uuid
from enum import Enum
from multiprocessing import Process, Queue, shared_memory
from functools import reduce
import img_processing
from math import floor
from functools import reduce
from my_collections import (
    ShapeItem,
    CropSlicing,
    UI_ready_element,
    UI_Element,
    SharedMem_ImgTicket
    )
import re
import itertools
from functools import reduce
try:
    pass
except Exception as e:
    # TODO
    print("this must be scambilight - bad solution please fix TODO")
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lumotag import get_perspectivewarp_dictkey, get_perspectivewarp_filename

RELAY_BOUNCE_S = 0.02

class RelayFunction(Enum):
    torch = 1
    unused_1 = 2
    unused_2 = 3

def create_id():
    return str(uuid.uuid4())


class gun_config(ABC):
    model = "NOT OVERRIDDEN!"
    DETAILS_FILE = '/home/lumotag/MY_INFO.txt'
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
        self.zoom_debounce = Debounce(
            debounce_sec=0.1)
        self.msg_heartbeat_s = 20

        self.torch_debounce = Debounce(
            debounce_sec=1.0)

        self._UI_overlay = None


    @property
    @abstractmethod
    def button_torch(self):
        ...
    @property
    @abstractmethod
    def button_trigger(self):
        ...
    @property
    @abstractmethod
    def button_rear(self):
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
    @property
    @abstractmethod
    def img_subsmple_factor(self):
        ...
    # UNIQUEFIRE T65 IR light has 3 modes
    # need to cycle through them each time
    @abstractmethod
    def light_strobe_cnt(self):
        ...
    @abstractmethod
    def internal_img_crop_lr(self):
        ...
    @abstractmethod
    def internal_img_crop_sr(self):
        ...
    @property
    @abstractmethod
    def video_modes(self):
        ...
    @property
    @abstractmethod
    def video_modes_closerange(self):
        ...
    @property
    @abstractmethod
    def ui_overlay(self) -> dict:
        ...

    def get_unrotated_UI_canvas(self):
        """For creating UI elements - we need a canvas with no rotation
        to make adding elements easier. So modify the canvas size using
        rotation so we can add UI elements and rotate later. For instance
        if we have a screen size of 1000 * 500, with 0 rotation this will be the same,
        but at 90 degree it will be 500 * 1000 and thats what we want to draw on before
        rotating to match viewer/LCD offset
        
        180 degrees we do not have to worry about as image ratio is the same"""
        if self.screen_rotation in [90, -90, 270, -270]:
            # flip dims
            return self.screen_size[::-1]
        return self.screen_size
    

class filesystem(ABC):
    @abstractmethod
    def save_image(self):
        pass

    @abstractmethod
    def save_barcodepair(self):
        pass

    @staticmethod
    def get_closerange_to_longrange_transform():
        script_path = os.path.abspath(__file__)
        parent_dir = os.path.dirname(script_path)
        pickle_file_path = os.path.join(parent_dir, get_perspectivewarp_filename())
        with open(pickle_file_path, 'rb') as f:
            perp_details = pickle.load(f)
        return perp_details[get_perspectivewarp_dictkey()]

class display(ABC):
    
    def __init__(self,  _gun_config: gun_config) -> None:
        self.display_rotate = _gun_config.screen_rotation
        self.screen_size = _gun_config.screen_size
        self.opencv_win_pos = _gun_config.opencv_window_pos
        self.emptyscreen = img_processing.get_empty_lumodisplay_img(_gun_config.screen_size)
        # np.zeros(
        #     ( _gun_config.screen_size + (3,)), np.uint8)
        #self.draw_test_rect()
        self._affine_transform = {}


    @abstractmethod
    def display_method(image, self):
        pass

    def TESTgenerate_output_affine2cam(self, cam_capture1, cam_2capture):
        """use affine transform to resize and rotate image in one calculation
        need 2 sets of 3 corresponding points to create calculation"""
        crop = 400
        cam_2capture= cam_2capture[
            crop:cam_2capture.shape[0]-crop,
            crop:cam_2capture.shape[1]-crop]
        concatted = img_processing.concat_image(cam_capture1, cam_2capture)

        if self._affine_transform is None:
            self._affine_transform = img_processing.get_fitted_affine_transform(
                cam_image_shape=concatted.shape,
                display_image_shape=self.emptyscreen.shape,
                rotation=self.display_rotate
            )

        row_cols = self.emptyscreen.shape[0:2][::-1]
        outptu_img = img_processing.do_affine(concatted, self._affine_transform, row_cols)
        outptu_img = cv2.cvtColor(outptu_img, cv2.COLOR_GRAY2BGR)
        #height, width = outptu_img.shape
        #three_channel_image = np.zeros((height, width, 3), dtype=outptu_img.dtype)
        #three_channel_image[:, :, 2] = outptu_img
        return outptu_img
    

    
    def generate_output_affine(self, cam_capture):
        """use affine transform to resize and rotate image in one calculation
        need 2 sets of 3 corresponding points to create calculation"""

        if cam_capture.shape[0:2] not in self._affine_transform:
            self._affine_transform[cam_capture.shape[0:2]] = img_processing.get_fitted_affine_transform(
                cam_image_shape=cam_capture.shape,
                display_image_shape=self.emptyscreen.shape,
                rotation=self.display_rotate
            )

        row_cols = self.emptyscreen.shape[0:2][::-1]
        outptu_img = img_processing.do_affine(cam_capture, self._affine_transform[cam_capture.shape[0:2]], row_cols)
        outptu_img = cv2.cvtColor(outptu_img, cv2.COLOR_GRAY2BGR)
        #height, width = outptu_img.shape
        #three_channel_image = np.zeros((height, width, 3), dtype=outptu_img.dtype)
        #three_channel_image[:, :, 2] = outptu_img
        return outptu_img

    def add_internal_section_region(self, source_image_shape, inputimg, _slice: CropSlicing, affinetransform: Optional):
        """Draw the white square which is the inner zone for detecting patterns
        either use the image shape which is used as a dictionary look-up for previously calculated affine transforms
        or provide the affine transform itself
        """

        if affinetransform is None:
            affine_m = self._affine_transform[source_image_shape[0:2]]
        else:
            affine_m = affinetransform

        left_top = tuple(
            np.matmul(affine_m, np.array([_slice.left,_slice.top,1])).astype(int))
        right_low = tuple(
            np.matmul(affine_m, np.array([_slice.right,_slice.lower,1])).astype(int))
        inputimg = cv2.rectangle(inputimg, left_top, right_low, (255,255,255), 2)
        #inputimg[int(left_top[1]):int(right_low[1]), int(right_low[1])] = 100


    def add_target_tags(self, output, graphics: dict[tuple[int,int], list[ShapeItem]]):
        """ we are using IMAGE SHAPE to find the camera source and corresponding transform"""
        for _shape, result_package in graphics.items():
            for result in result_package:
                result.transform_points(self._affine_transform[_shape])
                img_processing.draw_pattern_output(
                    image=output,
                    patterndetails=result)

    # def add_crosshair_and_analytics_graphics(self, source_image_shape, output, graphics: ShapeItem):
    #     img_processing.add_cross_hair(
    #         output,
    #         adapt=True)
    #     for c in graphics:
    #         c.transform_points(self._affine_transform[source_image_shape[0:2]])
    #         img_processing.draw_pattern_output(
    #             image=output,
    #             patterndetails=c)

    @staticmethod
    def get_norm_fade_val(player, analysis):
        if len(analysis) > 0:
            return player.elements_fadein()
        else:
            return player.elements_fadeout()

    def add_playerinfo_graphics(self, output, players: dict, analysis: ShapeItem):

        for player in players.values():
            fade_norm = self.get_norm_fade_val(player, analysis)
            for element in player.ui_elements:
                img_processing.add_ui_elements(
                    output,
                    element,
                    fade_norm
                    )


# class PlayerInfoBox:
#     def __init__(
#             self,
#             playername,
#             playergraphic,
#             _gun_config: gun_config
#             ) -> None:
#         """object to persist player name and graphic
        
#         params:
        
#         cam_img_res: resolution of the image capture device, to calculate affine transforms
#         playergraphic: PNG file (with alpha channel)"""

#         self.timer = TimeDiffObject()
#         self.playername = playername
#         self.playergraphic = playergraphic
#         self.gun_config: gun_config = _gun_config
#         self.output_display_shape = img_processing.get_empty_lumodisplay_img(
#             _gun_config.screen_size
#             ).shape
  
#         self.gray_image, self.alphamask = self.create_player_image_and_mask()
#         self.fade_ms = 250
#         self.current_fade_ms = 0
#         #self.fade_direction = 1

#         self.ui_elements = []

#         self.ui_elements.append(self.get_affine_transform(
#             self.gray_image,
#             element_name=UI_Element.PHOTO.value)
#             )

#         self.ui_elements.append(self.get_affine_transform(
#             self.create_player_text(),
#             element_name=UI_Element.USER_ID.value)
#             )

#     def elements_fadein(self):
#         return self.calculate_fade(direction=1)

#     def elements_fadeout(self):
#         return self.calculate_fade(direction=-1)

#     def calculate_fade(self, direction: Literal[-1, 1]):
#         if direction not in [-1, 1]:
#             raise Exception("bad input to calculate fade", direction)
#         time_diff_ms = self.timer.get_dt() * 1000
#         self.timer.reset()
#         self.current_fade_ms += (time_diff_ms * direction)
#         # limit working fade value
#         self.current_fade_ms = min(
#             max(self.current_fade_ms, 0),
#             self.fade_ms
#             )
#         # get normalised value
#         norm = self.current_fade_ms / self.fade_ms
#         return self.lerp(norm)

#     @staticmethod
#     def lerp(x):
#         lerpation = 1 - (1 - x) * (1 - x)
#         return max(0, min(lerpation, 1))

#     def create_player_text(self):
#         """we need to create the player name/ID/handle
#         but to a specific size so it looks OK, then
#         rotate it"""

#         id_img = img_processing.print_text_in_boundingbox(
#             self.playername,
#             grayscale=True
#             )

#         return id_img

#     def create_player_image_and_mask(self):
#         """get the transparent player custom graphic"""
#         img = img_processing.load_img_set_transparency()
#         gray_image = cv2.cvtColor(img[:,:,0:3], cv2.COLOR_BGR2GRAY)
#         alpha_mask = img[:,:,3]

#         #img_processing.test_viewer(gray_image, 0, True, True)
#         return gray_image, alpha_mask


#     def get_affine_transform(
#             self,
#             ui_element,
#             element_name: UI_Element):


#         ui_element = img_processing.rotate_img_orthogonal(
#             ui_element,
#             (360-self.gun_config.screen_rotation)
#             )

#         input_pts = img_processing.AffinePoints(
#             top_left_w_h=[0, 0],
#             top_right_w_h=[ui_element.shape[1], 0],
#             lower_right_w_h=[ui_element.shape[1], ui_element.shape[0]]
#         )

#         # get pixel positions for display output
#         pixel_pos = self.gun_config.ui_overlay[element_name].get_pixel_positions(
#             self.output_display_shape
#             )

#         output_pts = img_processing.AffinePoints(
#             top_left_w_h=[pixel_pos.left, pixel_pos.top],
#             top_right_w_h=[pixel_pos.right, pixel_pos.top],
#             lower_right_w_h=[pixel_pos.right, pixel_pos.lower]
#         )

#         transfrm = img_processing.get_affine_transform(
#             pts1=np.asarray(input_pts.as_array(), dtype="float32"),
#             pts2=np.asarray(output_pts.as_array(), dtype="float32"))

#         #row_cols = self.output_display_shape[0:2][::-1]
#         #outptu_img = img_processing.do_affine(ui_element, transfrm, row_cols)

#         #img_processing.test_viewer(outptu_img, 0, True, True)

#         resized_element = img_processing.resize_image(
#             ui_element,
#             abs(pixel_pos.left-pixel_pos.right),
#             abs(pixel_pos.top-pixel_pos.lower)
#             )
#         #img_processing.test_viewer(resized_element, 0, True, True)
#         #test_grab = outptu_img[pixel_pos.top:pixel_pos.lower, pixel_pos.left:pixel_pos.right]
#         #print("test grab")
#         #img_processing.test_viewer(test_grab, 0, True, True)
#         #img_processing.test_viewer(resized_element, 0, True, True)

#         # plops = UI_ready_element(
#         #     name=element_name,
#         #     position=pixel_pos,
#         #     image=resized_element,
#         #     transform=transfrm
#         # )

#         # testempty = img_processing.get_empty_lumodisplay_img(self.gun_config.screen_size)
#         # img_processing.add_ui_elements(testempty, plops)
#         # img_processing.test_viewer(testempty, 0, True, True)

#         return UI_ready_element(
#             name=element_name,
#             position=pixel_pos,
#             image=resized_element,
#             transform=transfrm
#         )


class PlayerInfoBoxv2:
    def __init__(
            self,
            playername,
            playergraphic,
            _gun_config: gun_config
            ) -> None:
        """object to persist player name and graphic
       
        params:

        cam_img_res: resolution of the image capture device, to calculate affine transforms
        playergraphic: PNG file (with alpha channel)"""

        self.timer = TimeDiffObject()
        self.playername = playername
        self.playergraphic = playergraphic
        self.gun_config: gun_config = _gun_config

        self.unrotated_display_canvas = img_processing.get_empty_lumodisplay_img(
            self.gun_config.screen_size
            )
        self.gray_image, self.alphamask = self.create_player_image_and_mask()
        self.fade_ms = 250
        self.current_fade_ms = 0
        #self.fade_direction = 1
        self.healthpoints = 100
        self.ui_elements = []

        # load up array with UI element object
        self.ui_elements.append(self.get_affine_transform(
            self.gray_image,
            element_name=UI_Element.PHOTO.value)
            )

        self.ui_elements.append(self.get_affine_transform(
            self.create_player_text(self.playername),
            element_name=UI_Element.USER_ID.value)
            )

        self.ui_elements.append(self.get_affine_transform(
            self.create_player_text(playername="doesn't matter"),
            element_name=UI_Element.USER_INFO.value)
            )
        

        self.static_canvas = self.create_static_canvas_elements()

    def sim_healthpoints(self):
        self.healthpoints =- 1
        if self.healthpoints < 0:
            self.healthpoints = 100

    def create_static_canvas_elements(self):
        """for the elements that are not going to change, such
        as avatar and player name"""
        temp = self.unrotated_display_canvas.copy()
        for elm in self.ui_elements:
            img_processing.add_ui_elements(
                temp,
                elm,
                fade_norm=1
            )
        temp = img_processing.rotate_img_orthogonal(temp, self.gun_config.screen_rotation)
        #img_processing.quick_image_viewer(temp)
        return temp

    def elements_fadein(self):
        return self.calculate_fade(direction=1,fade_ms= self.fade_ms)

    def elements_fadeout(self):
        return 1
        #return self.calculate_fade(direction=-1, fade_ms=self.fade_ms*1000)

    def calculate_fade(self, direction: Literal[-1, 1], fade_ms):
        if direction not in [-1, 1]:
            raise Exception("bad input to calculate fade", direction)
        time_diff_ms = self.timer.get_dt() * 1000
        self.timer.reset()
        self.current_fade_ms += (time_diff_ms * direction)
        # limit working fade value
        self.current_fade_ms = min(
            max(self.current_fade_ms, 0),
            fade_ms
            )
        # get normalised value
        norm = self.current_fade_ms / fade_ms
        return self.lerp(norm)

    @staticmethod
    def lerp(x):
        lerpation = 1 - (1 - x) * (1 - x)
        return max(0, min(lerpation, 1))

    @staticmethod
    def create_player_text(playername):
        """we need to create the player name/ID/handle
        but to a specific size so it looks OK, then
        rotate it"""

        id_img = img_processing.print_text_in_boundingbox(
            playername,
            grayscale=True
            )

        return id_img

    
    def create_player_image_and_mask(self):
        """get the transparent player custom graphic"""
        img = img_processing.load_img_set_transparency()
        gray_image = cv2.cvtColor(img[:,:,0:3], cv2.COLOR_BGR2GRAY)
        alpha_mask = img[:,:,3]

        #img_processing.test_viewer(gray_image, 0, True, True)
        return gray_image, alpha_mask


    def get_affine_transform(
            self,
            ui_element,
            element_name: UI_Element):


        # ui_element = img_processing.rotate_img_orthogonal(
        #     ui_element,
        #     (360-self.gun_config.screen_rotation)
        #     )

        # get points from original input material
        # do not need to modify these
        input_pts = img_processing.AffinePoints(
            top_left_w_h=[0, 0],
            top_right_w_h=[ui_element.shape[1], 0],
            lower_right_w_h=[ui_element.shape[1], ui_element.shape[0]]
        )

        # get pixel positions for display output from normalised positions
        # here we can modify the pixels to keep the image ratio
        pixel_pos = self.gun_config.ui_overlay[element_name].get_pixel_positions_with_ratio(
            self.unrotated_display_canvas.shape,
            ui_element.shape
            )

        # check_ratio = (np.array(ui_element.shape) / np.array((pixel_pos.lower-pixel_pos.top, pixel_pos.right-pixel_pos.left))).astype("float")
        # if abs(check_ratio[0] - check_ratio[1]) > 0.5:
        #     raise Exception(f"bad ratio!! something weird happening for {element_name}")

        output_pts = img_processing.AffinePoints(
            top_left_w_h=[pixel_pos.left, pixel_pos.top],
            top_right_w_h=[pixel_pos.right, pixel_pos.top],
            lower_right_w_h=[pixel_pos.right, pixel_pos.lower]
        )

        transfrm = img_processing.get_affine_transform(
            pts1=np.asarray(input_pts.as_array(), dtype="float32"),
            pts2=np.asarray(output_pts.as_array(), dtype="float32"))

        #row_cols = self.output_display_shape[0:2][::-1]
        #outptu_img = img_processing.do_affine(ui_element, transfrm, row_cols)

        #img_processing.test_viewer(outptu_img, 0, True, True)

        resized_element = img_processing.resize_image(
            ui_element,
            pixel_pos.right-pixel_pos.left,
            pixel_pos.lower-pixel_pos.top
            )

        # check_ratio = (np.array(ui_element.shape) / np.array(resized_element.shape)).astype("float")
        # if abs(check_ratio[0] - check_ratio[1]) > 0.01:
        #     raise Exception("bad ratio!! something weird happening")
        #img_processing.test_viewer(resized_element, 0, True, True)
        #test_grab = outptu_img[pixel_pos.top:pixel_pos.lower, pixel_pos.left:pixel_pos.right]
        #print("test grab")
        #img_processing.test_viewer(test_grab, 0, True, True)
        #img_processing.test_viewer(resized_element, 0, True, True)

        # plops = UI_ready_element(
        #     name=element_name,
        #     position=pixel_pos,
        #     image=resized_element,
        #     transform=transfrm
        # )

        # testempty = img_processing.get_empty_lumodisplay_img(self.gun_config.screen_size)
        # img_processing.add_ui_elements(testempty, plops)
        # img_processing.test_viewer(testempty, 0, True, True)

        return UI_ready_element(
            name=element_name,
            position=pixel_pos,
            image=resized_element,
            transform=transfrm
        )


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
        self._id = str(uuid.uuid4())[:8]

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


class Camera_synchronous_with_buffer(Camera):
    
    def __init__(self, video_modes, imagegen_cls) -> None:
        super().__init__(video_modes)
        self.imagegen_cls = imagegen_cls(self.get_res())
        self._store_res = None
        if not self.get_is_reversed():
            self._store_res = self.get_res()
        else:
            self._store_res = tuple(reversed(self.get_res()))
        self.shared_mem_handler = []
        self.configure_shared_memory()

    def gen_image(self):
        return self.imagegen_cls.get_image()

    def __next__(self):
        img = self.gen_image()
        memblock_0 = self.shared_mem_handler[0].mem_ids[self._id]
        shared_mem_1: np.ndarray = np.ndarray(
                img.shape,
                dtype=img.dtype,
                buffer=memblock_0.buf)
        shared_mem_1[:] = img[:]
        return img

    def configure_shared_memory(self):
        img_byte_size = reduce(
            lambda acc, curr: acc * curr, self.get_res())


        # we add more than 1 instance of shared memory
        self.shared_mem_handler.append(SharedMemory(
                            obj_bytesize=img_byte_size,
                            discrete_ids=[self._id]
                                        ))

    def get_mem_buffers(self) -> dict:
        return (
            {0: self.shared_mem_handler[0].mem_ids[self._id]})

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
        self._shared_id_index_name = self._id + "whatever" # oh god this is turning into a mess
        self._store_res = None
        self.safe_mem_details = None
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


    # don't make this a property
    def get_safe_mem_details(self):
        return self.safe_mem_details

    def get_mem_buffers(self) -> dict:
        return (
            {0: self.shared_mem_handler[0].mem_ids[self._id + "0"],
            1: self.shared_mem_handler[1].mem_ids[self._id + "1"]})

    def configure_shared_memory(self):
        # we need to get shape of image first to
        # create memory buffer
        # don't call this before everything else has been initialised!

        img_byte_size = reduce(
            lambda acc, curr: acc * curr, self.get_res())


        # we add more than 1 instance of shared memory
        self.shared_mem_handler.append(SharedMemory(
                            obj_bytesize=img_byte_size,
                            discrete_ids=[self._id + "0"]
                                        ))

        self.shared_mem_handler.append(SharedMemory(
                            obj_bytesize=img_byte_size,
                            discrete_ids=[self._id + "1"]
                                        ))
        self.shared_mem_index = SharedMemory(
                            obj_bytesize=1,
                            discrete_ids=[self._shared_id_index_name]
                                        )

        memblock_0 = self.shared_mem_handler[0].mem_ids[self._id + "0"]
        memblock_1 = self.shared_mem_handler[1].mem_ids[self._id + "1"]
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
            int(mem_details.index)].mem_ids[self._id + str(mem_details.index)].buf

        img_byte_size = reduce(
            lambda acc, curr: acc * curr, self._store_res)
        
        img_buff = np.frombuffer(
            strm_buff,
            dtype=('uint8')
                )[0:img_byte_size].reshape(self._store_res)

        self.last_img = img_buff

        self.safe_mem_details = SharedMem_ImgTicket(
            index=mem_details.index,
            res=mem_details.res,
            buf_size=mem_details.buf_size,
            id = mem_details.id)
        #print("FLIPFLOP saving record for analyis", self.safe_mem_details)
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
                            discrete_ids=[self._id]
                                        )

        memblock = self.shared_mem_handler.mem_ids[self._id]

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
        
        
        strm_buff = self.shared_mem_handler.mem_ids[self._id].buf

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
        return self.shared_mem_handler.mem_ids[self._id].buf

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
        difference_secs = self._stop_time-self._start_time
        return difference_secs

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

def cycle_files(file_list):
    while True:
        for file in file_list:
            yield file

# Function to extract the number between "cnt" and "cnt"
def extract_number(file_name):
    match = re.search(r'cnt(\d+)cnt', file_name)
    if match:
        return int(match.group(1))
    return float('inf')  # Return a large number if the pattern is not found


def get_images_for_cam_pair(
        cam_name: Literal["close", "long"],
        filters: list[str],
        image_extension: str = ".jpg",
        imgfolder= None
        ):
    """Make sure not loading in broken pairs"""
    if imgfolder is None:
        imgfolder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    all_images = images_in_folder(imgfolder, image_extension)
    filtered_images = all_images
    for filter in filters:
        filtered_images = [i for i in filtered_images if filter in i]
    this_cam_images = [i for i in filtered_images if cam_name in i]
    if cam_name == "close":
        pair_name = "long"
    else:
        pair_name = "close"
    paired_cam_images = [i for i in filtered_images if pair_name in i]
    # get dictionary of image ID with key = filepath
    this_cam_with_id = {extract_number(i): i for i in this_cam_images}
    paired_cam_with_id = {extract_number(i): i for i in paired_cam_images}
    common_ids = list(set(list(this_cam_with_id.keys())).intersection(set(list(paired_cam_with_id.keys()))))
    this_cam_with_id = [(extract_number(i), i) for i in this_cam_images if extract_number(i) in common_ids]
    paired_cam_with_id = [(extract_number(i), i) for i in paired_cam_images if extract_number(i) in common_ids]

    valid_this_cam =  sorted([y for x, y in this_cam_with_id], key=extract_number)
    valid_other_cam =  sorted([y for x, y in paired_cam_with_id], key=extract_number)
    assert len(valid_this_cam) == len(valid_other_cam)
    return valid_this_cam


# class ImageLibrary_longrange(ImageGenerator):
#     def __init__(self, res) -> None:
#         self.blank_image = np.zeros(tuple(reversed(res)), np.uint8)
#         sorted_files = get_images_for_cam_pair(
#             cam_name="long",
#             filters=["unique"]#filters=["quad", "14877"]
#             )#mgfolder=r"D:\lumotag_training_data\_player_1"
#         # create duplicates
#         sorted_files = reduce(lambda acc, s: acc + [s] * 1, sorted_files, [])
#         self.cycled_files_generator = iter(sorted_files)#cycle_files(sorted_files)
#         self.res = res
#         # if len(self.images) < 1:
#         #     raise Exception("could not find images in folder")


#     def get_image(self):
#         img_to_load = next(self.cycled_files_generator)
#         time.sleep(0.03)
#         img = cv2.imread(img_to_load)
#         #print(f"img {img_to_load}")
#         if len(img.shape) == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         #img = cv2.resize(img, tuple(self.res[0:2]))
#         self.blank_image[:] = img
#         return self.blank_image


# class ImageLibrary_closerange(ImageGenerator):
#     def __init__(self, res) -> None:
#         self.blank_image = np.zeros(tuple(reversed(res)), np.uint8)
#         sorted_files = get_images_for_cam_pair(
#             cam_name="close",
#             filters=["unique"]#filters=["quad", "14877"]
#             )
#             #imgfolder=r"D:\lumotag_training_data\_player_1")
#         # create duplicates
#         sorted_files = reduce(lambda acc, s: acc + [s] * 1, sorted_files, [])
#         self.cycled_files_generator = iter(sorted_files)#cycle_files(sorted_files)
#         self.res = res
#         # if len(self.images) < 1:
#         #     raise Exception("could not find images in folder")


#     def get_image(self):
#         img_to_load = next(self.cycled_files_generator)
        
#         img = cv2.imread(img_to_load)
#         #print(f"img {img_to_load}")
#         if len(img.shape) == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         #img = cv2.resize(img, tuple(self.res[0:2]))
#         self.blank_image[:] = img
#         time.sleep(0.03)
#         return self.blank_image


class ImageLibraryMeta(type(ImageGenerator)):
    """Experiment with metaclasses - can we pass the metaclass to a downstream process
    and it succesfully instances it depending on flavour? would be useful
    if we have lots of similar classes that have to be passed around"""
    def __new__(cls, name, bases, attrs):
        if 'cam_name' not in attrs:
            raise TypeError(f"Class {name} must define 'cam_name'")
        
        def init(self, res):
            self.blank_image = np.zeros(tuple(reversed(res)), np.uint8)
            sorted_files = get_images_for_cam_pair(
                cam_name=self.cam_name,
                filters=["quadrocode"]#quadrocode_corners
            )
            sorted_files = reduce(lambda acc, s: acc + [s] * 1, sorted_files, [])
            self.cycled_files_generator = iter(sorted_files)
            self.res = res

        def get_image(self):
            img_to_load = next(self.cycled_files_generator)
            img = cv2.imread(img_to_load)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.blank_image[:] = img
            time.sleep(0.03)
            return self.blank_image

        attrs['__init__'] = init
        attrs['get_image'] = get_image
        
        return super().__new__(cls, name, bases, attrs)


class ImageLibrary_longrange(ImageGenerator, metaclass=ImageLibraryMeta):
    cam_name = "long"


class ImageLibrary_closerange(ImageGenerator, metaclass=ImageLibraryMeta):
    cam_name = "close"


class ImageLibrary(ImageGenerator):
    
    def __init__(self, res) -> None:
        self.blank_image = np.zeros(tuple(reversed(res)), np.uint8)
        imgfoler = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        self.images = images_in_folder(imgfoler, [".jpg"])

        self.images = [i for i in self.images if "quadrocode_corners" in i]

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


class test_ui_elements(ImageGenerator):
   
    def __init__(self, res) -> None:
        self.blank_image = np.zeros(tuple(reversed(res)), np.uint8)
        imgfoler = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.images = images_in_folder(imgfoler, [".jpg"])
        self.images = [i for i in self.images if "unique" in i]
        self.image_freq = 30
        self.res = res
        if len(self.images) < 1:
            raise Exception("could not find images in folder")


    def get_image(self):
        self.image_freq -= 1
        print(self.image_freq)
        if self.image_freq == 0:
            self.image_freq = 30
        img_to_load = random.choice(self.images)

        img = cv2.imread(img_to_load)

        print(f"img {img_to_load}")
    
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.blank_image[:] = img
        #if random.randint(0,100) > 5:
        if self.image_freq < 15 or random.randint(0, 20) == 1:
            self.blank_image[:] = 0
        if random.randint(0, 6) == 1:
            self.blank_image[:] = img
        return self.blank_image



    
def images_in_folder(directory, imgtypes: list[str]):
    allFiles = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name[-4:len(name)] in imgtypes:
                allFiles.append(os.path.join(root, name))
    return allFiles


class VoiceBase(ABC):

    def __init__(self) -> None:
        """Class to provide synthetic
        voice prompts or alerts"""
        self.in_box = Queue(maxsize=10)
        self.t = threading.Thread(
            target=self.speaker,
            args=(self.in_box,))
        self.t.start()

    def wait_for_speak(self):
        while not self.in_box.empty():
            time.sleep(0.1)
        # must be a better way to do this

    def speak(
            self,
            message: str):
        # use  in_box._qsize() to prevent
        # blowing it up
        try:
            if self.in_box.qsize() >= self.in_box._maxsize - 1:

                try:
                    for _ in range (0, self.in_box._maxsize):
                        self.in_box.get(block=False)
                except Exception: # queue here refers to the module, not a class
                    # don't trust the queue.empty exception 100% here
                    print('cleared overflowing voice queue')

                self.in_box.put(
                    "Voice buffer overflow",
                    block=False)
            else:
                self.in_box.put(
                    message,
                    block=False)
        except Exception as e:
            print(f"speak: something nasty happening {e}")

    def speaker(self, in_box):
        pass

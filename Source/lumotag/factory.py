# honestly
from abc import ABC, abstractmethod
import numpy as np
import json
import time
from enum import Enum
from functools import lru_cache
from typing import Literal, Optional, Union
import cv2
import os
import math
from contextlib import contextmanager
from collections import deque
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
from math import floor, ceil
from functools import reduce
from my_collections import (
    ShapeItem,
    CropSlicing,
    UI_ready_element,
    SharedMem_ImgTicket,
    ScreenPixelPositions,
    UI_Behaviour_static,
    UI_Behaviour_dynamic,
    ScreenNormalisedPositions,
    HeightWidth
    )
import re
import itertools
from functools import reduce
from dataclasses import dataclass
import video_recorder
try:
    pass
except Exception as e:
    # TODO
    print("this must be scambilight - bad solution please fix TODO")
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lumotag import get_perspectivewarp_dictkey, get_perspectivewarp_filename

RELAY_BOUNCE_S = 0.02

def create_image_id(alphanumeric_chars=(
    # put these here so we can cache it without using a global
    list(range(48, 58)) +    # 0-9
    list(range(65, 91)) +    # A-Z
    list(range(97, 123))     # a-z
)):
    """Create strictly alphanumeric ID with LID markers"""
    # Cache process ID to avoid repeated system calls
    if not hasattr(create_image_id, '_cached_pid'):
        create_image_id._cached_pid = os.getpid() % 10000
    
    # Convert PID to string and pad to 4 digits
    pid_str = f"{create_image_id._cached_pid:04d}"
    pid_chars = np.array([ord(c) for c in pid_str], dtype=np.uint8)
    
    # Randomly select 6 more alphanumeric characters
    random_chars = np.array(np.random.choice(alphanumeric_chars, 6), dtype=np.uint8)
    
    # Combine: 4 PID digits + 6 random chars = 10 total
    combined_id = np.concatenate([pid_chars, random_chars])
    
    lid_array = np.array([76, 73, 68], dtype=np.uint8) # LID in ascii!
    stacked_id = np.concatenate([lid_array, combined_id, lid_array])
    return stacked_id

def decode_image_id(image: np.ndarray) -> str:
    # Extract the full ID row (now 16 elements: LID + 10 random + LID)
    id_row = image[0, 0:16]
    decoded = id_row.tobytes().decode('utf-8')
    # see create_image_id how ID is created
    if decoded[0:3] != "LID" or decoded[-3:] != "LID":
        raise ValueError(f"image embedded ID not correctly formed: {decoded}")
    return decoded


class RelayFunction(Enum):
    torch = 1
    unused_1 = 2
    unused_2 = 3

def create_id():
    return str(uuid.uuid4())

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

        # self.my_id = create_id()

        self.trigger_debounce = Debounce(
            debounce_sec=0.1)
        self.zoom_debounce = Debounce(
            debounce_sec=0.1)
        self.msg_heartbeat_s = 20

        self.torch_debounce = Debounce(
            debounce_sec=1.0)

        # self._UI_overlay = None


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
    # @property
    # @abstractmethod
    # def ui_overlay(self) -> dict:
    #     ...

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
    

class GetID(ABC):
    @abstractmethod
    def get_persistant_device_id(self):
        """Get unique and persistant device id - create if does not exist"""
        ...

class FileSystemABC(ABC):
    @abstractmethod
    def save_image(self):
        pass

    @abstractmethod
    def save_barcodepair(self):
        pass

    @abstractmethod
    def save_numberstatus_cache(self, cache_data: dict[str, np.ndarray]) -> bool:
        """Save the _numberstatus_cache to temporary storage. Returns True if successful, False otherwise."""
        pass

    @abstractmethod
    def load_numberstatus_cache(self) -> Optional[dict[str, np.ndarray]]:
        """Load the _numberstatus_cache from temporary storage. Returns None if file doesn't exist or on error."""
        pass

    @abstractmethod
    def save_shieldstatus_cache(self, cache_data: list[np.ndarray]) -> bool:
        """Save the _shieldstatus_cache to temporary storage. Returns True if successful, False otherwise."""
        pass

    @abstractmethod
    def load_shieldstatus_cache(self) -> Optional[list[np.ndarray]]:
        """Load the _shieldstatus_cache from temporary storage. Returns None if file doesn't exist or on error."""
        pass

    @abstractmethod
    def _clean_images_folder(self):
        """Clean up all image files in the images folder on startup."""
        pass

    @staticmethod
    def get_closerange_to_longrange_transform():
        script_path = os.path.abspath(__file__)
        parent_dir = os.path.dirname(script_path)
        pickle_file_path = os.path.join(parent_dir, get_perspectivewarp_filename())
        print(f"Opening transform file {pickle_file_path}")
        try:
            with open(pickle_file_path, 'rb') as f:
                perp_details = pickle.load(f)
            return perp_details[get_perspectivewarp_dictkey()]
        except Exception as e:
            raise Exception(f"Error loading transform file get_closerange_to_longrange_transform: {e} : if this is a numpy error it may be a version match issue between creater and consumer")

class display(ABC):
    
    def __init__(self,  _gun_config: gun_config, recorder_screen: bool = False) -> None:
        self.video_recorder = None
        self.dim_check = {}
        self.recorder_screen = recorder_screen
        self.display_rotate = _gun_config.screen_rotation
        self.screen_size = _gun_config.screen_size
        self.opencv_win_pos = _gun_config.opencv_window_pos
        self.emptyscreen = img_processing.get_empty_lumodisplay_img(_gun_config.screen_size)
        # np.zeros(
        #     ( _gun_config.screen_size + (3,)), np.uint8)
        #self.draw_test_rect()
        self._affine_transform = {}
        # this is lazy - if we find we rotate a lot then do this properly
        if self.display_rotate == 270:
            self.cardio_gram_display =CardioGramDisplay(
                pos_x=0,
                pos_y=567,
                width=60,
                height=138,
                value_range=(0, 50),
                flow_direction=90
                )
        if self.display_rotate == 0:
            self.cardio_gram_display =CardioGramDisplay(
                pos_x=10,
                pos_y=self.screen_size[0]-80,
                width=self.screen_size[0]//2,
                height=60,
                value_range=(0, 100),
                flow_direction=0
                )
        if self.display_rotate in [180, 90]:
            print("Metric bars not configured yet!!!!!!!!!")
            self.cardio_gram_display =CardioGramDisplay(
                pos_x=10,
                pos_y=self.screen_size[0]-80,
                width=self.screen_size[0]//2,
                height=60,
                value_range=(0, 137),
                flow_direction=0
                )

    def display(self, image):
        if self.recorder_screen is False:
            self.display_method(image)
        else:
            self.display_with_recording(image)


    def display_with_recording(self, image):
        if image.shape[0:2] not in self.dim_check:
            self.dim_check[image.shape[0:2]] = True
        if len(self.dim_check) > 1:
            raise Exception(f"display_with_recording: dimensions changed: {self.dim_check}")
        if self.video_recorder is None:
            height, width = image.shape[:2]
            # Make dimensions even (required for some codecs)
            width = width - 1 if width % 2 == 1 else width
            height = height - 1 if height % 2 == 1 else height
            
            print(f"Initializing VideoRecorder with frame dimensions: {height}x{width}")
            self.video_recorder = video_recorder.VideoRecorder(
                width=width,
                height=height
            )
            self.video_recorder.start_recording()
        self.video_recorder.write_frame(
            image[0:self.video_recorder.height,
                  0:self.video_recorder.width,
                  :
                  ])
        self.display_method(image)

    @abstractmethod
    def display_method(image, self):
        pass

    def debug_add_imgpro_wait(self, time_ms, image):
        normed_to_100ms = int(image.shape[0] / 100)
        total_metrics = len(time_ms)
        offset = 20
        for cnt, metric in enumerate(time_ms):
            # Calculate step based on total number of metrics
            step = 255 // max(total_metrics, 1)
            
            # Generate colors using a simple formula
            red = (255 - cnt * step) % 256
            green = (cnt * step) % 256
            blue = (128 + cnt * step) % 256  # Offset blue for better visibility
            
            start_pos = (4 * cnt) + offset
            end_pos = (4 * (cnt + 1)) + offset
            
            # Draw the bar with the calculated color
            height = normed_to_100ms * int(metric)
            image[:height, start_pos:end_pos, 0] = red    # Red channel
            image[:height, start_pos:end_pos, 1] = green  # Green channel
            image[:height, start_pos:end_pos, 2] = blue   # Blue channel

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

    def add_internal_section_region(self, source_image_shape, inputimg, _slice: CropSlicing, affinetransform: Optional[np.ndarray]):
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


    def add_target_tags(self, output, graphics: dict[tuple[int,int], list[ShapeItem]], tags__Vs__healthpoints: Optional[dict[str, int]] = None):
        """ we are using IMAGE SHAPE to find the camera source and corresponding transform"""
        dedupe = set()
        for _shape, result_package in graphics.items():
            for result in result_package:
                if result.instance_id in dedupe:
                    continue
                dedupe.add(result.instance_id)
                transformed = result.transform_points(self._affine_transform[_shape])

                # get colour we want tag to show as depending on the players healthpoint (player corresponding to tag id )
                if tags__Vs__healthpoints and (hp := tags__Vs__healthpoints.get(str(transformed.decoded_id))) is not None:
                    colour = img_processing.health_to_color(hp)
                else:
                    colour = img_processing.GRAY  # default color

                img_processing.draw_pattern_output(
                    image=output,
                    patterndetails=transformed,
                    color=colour)


    @staticmethod
    def get_norm_fade_val(player, analysis):
        if len(analysis) > 0:
            return player.elements_fadein()
        else:
            return player.elements_fadeout()

@dataclass
class EnemyInfo:
    health: str
    displayname: str
    
    def __post_init__(self):
        self.health = f"H: {self.health}"
        self.displayname = f"N: {self.displayname}"
    
    @property
    def turn_red(self) -> bool:
        health_num = int(self.health.split(": ")[1])
        return health_num <= 0


class PlayerInfoBoxv2:
    # Shared fade state across all instances
    _shared_timer = TimeDiffObject()
    _shared_fade_ms = 250
    _shared_current_fade_ms = 0
    
    def __init__(
            self,
            playername,
            avatar_canvas: HeightWidth = None,
            info_box: HeightWidth = None
            ) -> None:
        """object to persist player name and graphic
       
        params:

        cam_img_res: resolution of the image capture device, to calculate affine transforms
        """

        self.playername = playername
        self.avatar_canvas = avatar_canvas
        self.info_box = info_box
        self.display_targetted_avatar = None
        self.enemyinfo = EnemyInfo(health="UNKNOWN", displayname="UNKNOWN")

        self.max_healthpoints = 100
        self.min_healthpoints = 0
        self.healthpoints = 1
        self.avatars_by_id: dict[str, np.ndarray] = {}
        # Pain system - performant for 40fps checks
        self.pain_duration_seconds = 0.3
        self._pain_expires_at = None  # None = not in pain

        # Create 500x500 black/white static image
        static = np.random.randint(0, 256, (500, 500), dtype=np.uint8)
        static_rgb = cv2.cvtColor(static, cv2.COLOR_GRAY2BGR)

    def add_player_avatar(self, display_name: str, img: np.ndarray):
        # col_image, alphamask = self.create_player_image_and_mask()
        if self.avatar_canvas is not None:
            # for local player we are not doing anything yet
            col_image = img_processing.get_resized_equalaspect(
                img,
                (self.avatar_canvas.height, self.avatar_canvas.width)
                )
            # alphamask = img_processing.get_resized_equalaspect(
            #     alphamask,
            #     (self.avatar_canvas.height, self.avatar_canvas.width)
            #     )
            self.avatars_by_id[display_name] = col_image

    def set_player_info(self, info: EnemyInfo):
        self.enemyinfo = info

    def get_player_avatar(self, display_name:str):
        return self.avatars_by_id.get(display_name, None)

    def set_targetted_avatar(self, display_name:str):
        self.display_targetted_avatar = self.avatars_by_id.get(display_name, None)

    def get_healthpoints(self):
        
        if self.healthpoints is None:
            return None
        if self.healthpoints < 1:
            self.healthpoints = self.min_healthpoints
        return self.healthpoints

    def update_healthpoints(self, diff: int):
        if self.healthpoints is not None:
            self.healthpoints = self.healthpoints + diff

    def set_healthpoints(self, new_hps: Optional[int]):
        self.healthpoints = new_hps

    def get_max_min_healthpoints(self)->tuple[int, int]:
        return self.max_healthpoints, self.min_healthpoints

    def is_in_pain(self) -> bool:
        """Check if player is currently in pain. Performant for 40fps checks."""
        if self._pain_expires_at is None:
            return False
        current_time = time.perf_counter()
        if current_time >= self._pain_expires_at:
            self._pain_expires_at = None  # Auto-expire
            return False
        return True

    def set_pain(self):
        """Trigger pain state for configured duration."""
        self._pain_expires_at = time.perf_counter() + self.pain_duration_seconds

    def elements_fadein(self):
        return self.calculate_fade(direction=1, fade_ms=self._shared_fade_ms)

    def elements_fadeout(self):
        return self.calculate_fade(direction=-1, fade_ms=self._shared_fade_ms, multiplier=(1/50))

    def calculate_fade(self, direction: Literal[-1, 1], fade_ms, multiplier=1):
        if direction not in [-1, 1]:
            raise Exception("bad input to calculate fade", direction)
        time_diff_ms = self._shared_timer.get_dt() * (1000 * multiplier)
        self._shared_timer.reset()
        self._shared_current_fade_ms += (time_diff_ms * direction)
        # limit working fade value
        self._shared_current_fade_ms = min(
            max(self._shared_current_fade_ms, 0),
            fade_ms
            )
        # get normalised value
        norm = self._shared_current_fade_ms / fade_ms
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
        col_image = img[:,:,0:3]
        alpha_mask = img[:,:,3]
        #img_processing.test_viewer(gray_image, 0, True, True)
        return col_image, alpha_mask



class LocalPlayerCard(PlayerInfoBoxv2):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.torchenergy = self.get_min_max_torchenergy()[1]
        self.ammo = self.get_min_max_ammo()[1]
        self.timer_torch_deplete = TimeDiffObject()
        self.timer_energy_recover = TimeDiffObject()
        self.torchstate = False

    def get_min_max_torchenergy(self):
        return(0, 100)

    def get_min_max_ammo(self):
        return(0, 300)
    
    def update_ammo(self, ammo:int):
        self.ammo += ammo
        min_, max_ = self.get_min_max_ammo()
        self.ammo = max(min(max_, self.ammo), min_)

    def torch_energy_update(self, deplete:bool):
        """update when energy used and recovery"""
        if deplete is False and self.torchstate is False:
            time_ms = self.timer_energy_recover.get_dt() * 1000
            self.timer_energy_recover.reset()
            self._update_torch_energy(time_ms/20)
            return
        elif deplete is True and self.torchstate is False:
            self.torchstate = True
            self.timer_torch_deplete.reset()
            return
        elif deplete is False and self.torchstate is True:
            self.torchstate = False
            self.timer_energy_recover.reset()
            return
        elif deplete is True and self.torchstate is True:
            time_ms = self.timer_torch_deplete.get_dt() * 1000
            self.timer_torch_deplete.reset()
            self._update_torch_energy(-time_ms/30)
            return

    def get_torch_energy(self):
        return self.torchenergy

    def get_normalised_torchenergy(self):
        """Get torch energy as a normalized value between 0 and 1"""
        min_energy, max_energy = self.get_min_max_torchenergy()
        current_energy = self.get_torch_energy()
        return round((current_energy - min_energy) / (max_energy - min_energy), 2)

    def _update_torch_energy(self, diff: float):
        min, max = self.get_min_max_torchenergy()
        self.torchenergy = self.torchenergy + diff
        if self.torchenergy < min:
            self.torchenergy = min
        if self.torchenergy > max:
            self.torchenergy = max

    def calculate_fade(self, direction: Literal[-1, 1], fade_ms, multiplier=1):
        return 1


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
    # this is the class which ACQUIRES the image from hardware/images etc 
    @abstractmethod
    def _get_image(self):
        pass
    
    @abstractmethod
    def get_raw_image(self):
        """Get full uncropped raw image from source. Must be implemented by subclasses."""
        pass
    
    def get_image(self):
        img = self._get_image()
        img_id = create_image_id()
        if len(img.shape) == 2:
            img[0, 0:img_id.shape[0]] = img_id
        else:
            img[0, 0:img_id.shape[0], 0] = img_id
        return img

class Camera(ABC):
    # this is the class which negotiates with an image acquisition class, and handles 
    # shared memory etc if we want async behaviour 
    def __init__(self, video_modes) -> None:
        self.res_select = 0 # This is how video modes are selected - but we always select the first one - I know its horrible
        self.last_img = None
        self.cam_res = video_modes
        self._is_reversed = None
        self._res = None
        self._id = str(uuid.uuid4())[:8]

    @abstractmethod
    def gen_image(self):
        pass

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
    
    def __init__(self, video_modes, imagegen_cls:ImageGenerator) -> None:
        super().__init__(video_modes)
        self.imagegen_cls = imagegen_cls(self.get_res())

    def gen_image(self):
        return self.imagegen_cls.get_image()

    def __next__(self):
        img = self.gen_image()
        self.last_img = img
        return img


class Camera_synchronous_with_buffer(Camera):
    
    def __init__(self, video_modes, imagegen_cls:ImageGenerator) -> None:
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
    def __init__(self, video_modes, imagegen_cls:ImageGenerator) -> None:
        super().__init__(video_modes)
        self.res_select = 0
        self.last_img = None
        self.handshake_queue = Queue(maxsize=1)
        self.handshake_queue2 = Queue(maxsize=1)
        self.raw_frame_request_queue = Queue(maxsize=1)  # Main → Subprocess: request signal
        self.raw_frame_response_queue = Queue(maxsize=1)  # Subprocess → Main: raw frame data
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
    
    def get_raw_image_sync(self) -> np.ndarray:
        """Synchronously request and retrieve full uncropped raw image from subprocess.
        
        This sends a request token to the subprocess via queue, which triggers
        the ImageGenerator to capture a raw frame instead of cropped.
        May cause slight lag while waiting for subprocess response.
        """
        # Empty response queue first in case there's a stale frame
        while not self.raw_frame_response_queue.empty():
            try:
                self.raw_frame_response_queue.get(block=False)
            except:
                break
        
        # Send request token (Main → Subprocess)
        try:
            self.raw_frame_request_queue.put("REQUEST_RAW", block=True, timeout=1.0)
        except queue.Full:
            print("possible race hazard in  raw_frame = self.raw_frame_response_queue.get - put token for request raw")
            return False

        # unblock the camera queue
        self.handshake_queue2.put("please rename this", block=True, timeout=None)
        
        # Consume the metadata that the subprocess will send (we don't need it for raw frames)
        _ = self.handshake_queue.get(block=True, timeout=2.0)
        
        # Wait for raw frame (Subprocess → Main)
        try:
            raw_frame = self.raw_frame_response_queue.get(block=True, timeout=2.0)
        except queue.Empty:
            print("possible race hazard in  raw_frame = self.raw_frame_response_queue.get")
            return False
        
        if isinstance(raw_frame, str):
            raise Exception(f"Expected raw frame, got: {raw_frame}")
        
        return raw_frame

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
            self.imagegen_cls,
            self.raw_frame_request_queue,
            self.raw_frame_response_queue)

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
        img_gen: ImageGenerator,
        raw_frame_request_queue: Queue,
        raw_frame_response_queue: Queue):

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
            
            # Check if raw frame is requested BEFORE handshake (non-blocking, no exception overhead)
            if not raw_frame_request_queue.empty():
                try:
                    _ = raw_frame_request_queue.get(block=False)  # Consume request (any value means "capture raw")
                    raw_img = _img_gen.get_raw_image()
                    raw_frame_response_queue.put(raw_img.copy(), block=True, timeout=1.0)
                except:
                    pass  # Queue became empty between check and get (rare race condition)
            
            #blocking put until consumer handshakes
            #print("FLIPFLOP waiting to send ASYNC outgoing:", output)
            _ = handshake_queue.get(block=True, timeout=None)
            myqueue.put(output, block=True, timeout=None)
            #print("FLIPFLOP sent!! ASYNC outgoing:", output)
            
            # Check if raw frame is requested AFTER handshake (non-blocking, no exception overhead)
            if not raw_frame_request_queue.empty():
                try:
                    _ = raw_frame_request_queue.get(block=False)  # Consume request (any value means "capture raw")
                    raw_img = _img_gen.get_raw_image()
                    raw_frame_response_queue.put(raw_img.copy(), block=True, timeout=1.0)
                except:
                    pass  # Queue became empty between check and get (rare race condition)
            


class FrameGrabber(Camera):
    """
    Triple-buffered async camera. Producer never blocks.
    
    Architecture:
        [Camera Subprocess]              [Main Process]           [Analysis Processes]
              |                               |                          |
         capture()                            |                          |
              |                               |                          |
         write to buf[write_idx]              |                          |
              |                               |                          |
         atomic: ready_idx = write_idx        |                          |
         write_idx = next free buffer         |                          |
              |                          next() called                    |
              |                               |                          |
              |                          read ready_idx                   |
              |                          copy from buf[ready_idx]         |
              |                          return frame                     |
              |                               |                   trigger_analysis()
              |                               |                          |
              |                               |                   read ready_idx
              |                               |                   copy from buf[ready_idx]
              
    Why triple buffer:
        - Producer writes to buffer A
        - Marks A as ready, starts writing to B  
        - Consumer copies from A
        - Producer finishes B, marks ready, writes to C
        - Even if consumer is slow, producer has C to write to
        - Producer never waits, consumer always gets complete frame
    """
    
    NUM_BUFFERS = 3
    
    def __init__(self, video_modes, imagegen_cls: ImageGenerator) -> None:
        super().__init__(video_modes)
        
        # Resolve resolution
        if self.get_is_reversed():
            self._store_res = tuple(reversed(self.get_res()))
        else:
            self._store_res = self.get_res()
        
        self.imagegen_cls = imagegen_cls
        self.frame_size = self._store_res[0] * self._store_res[1]
        
        # Create triple buffer in shared memory
        self._buf_names = ["{}_buf{}".format(self._id, i) for i in range(self.NUM_BUFFERS)]
        self._cleanup_stale_shm()
        self.buffers = [
            shared_memory.SharedMemory(create=True, size=self.frame_size, name=name)
            for name in self._buf_names
        ]
        
        # Shared atomic index: which buffer has the latest complete frame
        self._ready_idx_name = "{}_ready".format(self._id)
        self._unlink_shm_if_exists(self._ready_idx_name)
        self._ready_idx_shm = shared_memory.SharedMemory(
            create=True, size=4, name=self._ready_idx_name
        )
        # Initialize to -1 (no frame ready yet)
        self._ready_idx_view = np.ndarray((1,), dtype=np.int32, buffer=self._ready_idx_shm.buf)
        self._ready_idx_view[0] = -1
        
        # Frame counter for unique IDs
        self._frame_counter_name = "{}_fcnt".format(self._id)
        self._unlink_shm_if_exists(self._frame_counter_name)
        self._frame_counter_shm = shared_memory.SharedMemory(
            create=True, size=4, name=self._frame_counter_name
        )
        self._frame_counter_view = np.ndarray((1,), dtype=np.int32, buffer=self._frame_counter_shm.buf)
        self._frame_counter_view[0] = 0
        
        # Actual image shape (set by producer after first capture)
        self._actual_res_name = "{}_res".format(self._id)
        self._unlink_shm_if_exists(self._actual_res_name)
        self._actual_res_shm = shared_memory.SharedMemory(
            create=True, size=8, name=self._actual_res_name  # 2 x int32
        )
        self._actual_res_view = np.ndarray((2,), dtype=np.int32, buffer=self._actual_res_shm.buf)
        self._actual_res_view[:] = [0, 0]  # Will be set by producer
        
        # Track what we gave out last (for analysis processes)
        self._current_ticket = None  # type: Optional[SharedMem_ImgTicket]
        
        self._start_producer()
    
    @staticmethod
    def _unlink_shm_if_exists(name):
        """Remove shared memory by name if it exists (cleanup from previous runs)"""
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
    
    def _cleanup_stale_shm(self):
        for name in self._buf_names:
            self._unlink_shm_if_exists(name)
    
    def _start_producer(self):
        Process(
            target=self._producer_loop,
            args=(
                self.imagegen_cls,
                self.get_res(),  # Pass original res, not _store_res - let image generator decide shape
                self._buf_names,
                self._ready_idx_name,
                self._frame_counter_name,
                self._actual_res_name,
            ),
            daemon=True
        ).start()
    
    @staticmethod
    def _producer_loop(img_gen_cls, resolution, buf_names, ready_idx_name, frame_counter_name, actual_res_name):
        """Subprocess: captures continuously, never blocks"""
        img_gen = img_gen_cls(resolution)
        num_buffers = len(buf_names)
        
        # Attach to shared memory (buffer views created after first capture)
        shm_objects = [shared_memory.SharedMemory(name=name) for name in buf_names]
        buffers = None  # Will be initialized with actual image shape
        
        ready_shm = shared_memory.SharedMemory(name=ready_idx_name)
        ready_idx = np.ndarray((1,), dtype=np.int32, buffer=ready_shm.buf)
        
        counter_shm = shared_memory.SharedMemory(name=frame_counter_name)
        frame_counter = np.ndarray((1,), dtype=np.int32, buffer=counter_shm.buf)
        
        actual_res_shm = shared_memory.SharedMemory(name=actual_res_name)
        actual_res = np.ndarray((2,), dtype=np.int32, buffer=actual_res_shm.buf)
        
        # Triple buffer indices: write, ready, free
        write_idx = 0
        free_idx = 1  # Will become write target after first frame
        
        while True:
            # Capture frame
            img = img_gen.get_image()
            
            # First frame: create buffer views and store actual shape
            if buffers is None:
                buffers = [
                    np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
                    for shm in shm_objects
                ]
                actual_res[:] = img.shape  # Tell main process the actual shape
            
            # Write to current buffer
            buffers[write_idx][:] = img
            frame_counter[0] += 1
            
            # Swap: written buffer becomes ready, old ready becomes free, free becomes write
            old_ready = ready_idx[0]
            ready_idx[0] = write_idx
            write_idx = free_idx
            free_idx = old_ready if old_ready >= 0 else (write_idx + 1) % num_buffers
    
    def gen_image(self) -> np.ndarray:
        """Get latest complete frame. Returns VIEW into shared memory (zero-copy)."""
        # Spin until we have at least one frame (producer sets shape after first capture)
        while self._ready_idx_view[0] < 0 or self._actual_res_view[0] == 0:
            time.sleep(0.001)
        
        idx = int(self._ready_idx_view[0])
        frame_id = int(self._frame_counter_view[0])
        actual_shape = tuple(self._actual_res_view)
        
        # Return view - safe because producer won't touch "ready" buffer
        img = np.ndarray(actual_shape, dtype=np.uint8, buffer=self.buffers[idx].buf)
        
        # Store ticket for analysis processes
        self._current_ticket = SharedMem_ImgTicket(
            index=idx,
            res=actual_shape,
            buf_size=self.frame_size,
            id=frame_id
        )
        
        self.last_img = img
        return img
    
    def __next__(self) -> np.ndarray:
        return self.gen_image()
    
    # === Interface for analysis processes ===
    
    def get_mem_buffers(self) -> dict:
        """Returns {index: SharedMemory} for all buffers"""
        return {i: self.buffers[i] for i in range(self.NUM_BUFFERS)}
    
    def get_safe_mem_details(self) -> SharedMem_ImgTicket:
        """Returns ticket for the frame last returned by next().
        
        Must call next() at least once before calling this.
        """
        if self._current_ticket is None:
            self.gen_image()
        return self._current_ticket


class RingBufferCamera(Camera):
    """
    Ring buffer async camera with configurable buffer count.
    
    Simpler than FrameGrabber - producer just cycles through buffers sequentially.
    More buffers = more time before overwrite = safer for slow consumers.
    
    Buffer lifecycle (example with 4 buffers):
        Producer writes: 0 → 1 → 2 → 3 → 0 → 1 → ...
        Consumer reads from latest complete buffer.
        Buffer N is safe until producer completes N-1 more captures.
        
    Default 4 buffers = 3 capture cycles before overwrite (~30-100ms at typical FPS).
    """
    
    def __init__(self, video_modes, imagegen_cls: ImageGenerator, num_buffers: int = 4) -> None:
        super().__init__(video_modes)
        
        self.num_buffers = num_buffers
        
        # Resolve resolution
        if self.get_is_reversed():
            self._store_res = tuple(reversed(self.get_res()))
        else:
            self._store_res = self.get_res()
        
        self.imagegen_cls = imagegen_cls
        self.frame_size = self._store_res[0] * self._store_res[1]
        
        # Create ring buffer in shared memory
        self._buf_names = ["{}_ring{}".format(self._id, i) for i in range(self.num_buffers)]
        self._cleanup_stale_shm()
        self.buffers = [
            shared_memory.SharedMemory(create=True, size=self.frame_size, name=name)
            for name in self._buf_names
        ]
        
        # Shared atomic index: which buffer has the latest complete frame
        self._ready_idx_name = "{}_rready".format(self._id)
        self._unlink_shm_if_exists(self._ready_idx_name)
        self._ready_idx_shm = shared_memory.SharedMemory(
            create=True, size=4, name=self._ready_idx_name
        )
        self._ready_idx_view = np.ndarray((1,), dtype=np.int32, buffer=self._ready_idx_shm.buf)
        self._ready_idx_view[0] = -1
        
        # Frame counter for unique IDs
        self._frame_counter_name = "{}_rfcnt".format(self._id)
        self._unlink_shm_if_exists(self._frame_counter_name)
        self._frame_counter_shm = shared_memory.SharedMemory(
            create=True, size=4, name=self._frame_counter_name
        )
        self._frame_counter_view = np.ndarray((1,), dtype=np.int32, buffer=self._frame_counter_shm.buf)
        self._frame_counter_view[0] = 0
        
        # Actual image shape (set by producer after first capture)
        self._actual_res_name = "{}_rres".format(self._id)
        self._unlink_shm_if_exists(self._actual_res_name)
        self._actual_res_shm = shared_memory.SharedMemory(
            create=True, size=8, name=self._actual_res_name
        )
        self._actual_res_view = np.ndarray((2,), dtype=np.int32, buffer=self._actual_res_shm.buf)
        self._actual_res_view[:] = [0, 0]
        
        self._current_ticket = None  # type: Optional[SharedMem_ImgTicket]
        
        # Error queue for subprocess crash propagation
        self._error_queue = Queue(maxsize=1)
        
        # Queues for synchronous raw/color image capture
        self._raw_request_queue = Queue(maxsize=1)
        self._raw_response_queue = Queue(maxsize=1)
        
        self._start_producer()
    
    @staticmethod
    def _unlink_shm_if_exists(name):
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
    
    def _cleanup_stale_shm(self):
        for name in self._buf_names:
            self._unlink_shm_if_exists(name)
    
    def _start_producer(self):
        Process(
            target=self._producer_loop,
            args=(
                self.imagegen_cls,
                self.get_res(),
                self._buf_names,
                self._ready_idx_name,
                self._frame_counter_name,
                self._actual_res_name,
                self.num_buffers,
                self._error_queue,
                self._raw_request_queue,
                self._raw_response_queue,
            ),
            daemon=True
        ).start()
    
    @staticmethod
    def _producer_loop(img_gen_cls, resolution, buf_names, ready_idx_name, frame_counter_name, actual_res_name, num_buffers, error_queue, raw_request_queue, raw_response_queue):
        """Subprocess: captures continuously, cycles through ring buffer"""
        try:
            img_gen = img_gen_cls(resolution)
            
            shm_objects = [shared_memory.SharedMemory(name=name) for name in buf_names]
            buffers = None
            
            ready_shm = shared_memory.SharedMemory(name=ready_idx_name)
            ready_idx = np.ndarray((1,), dtype=np.int32, buffer=ready_shm.buf)
            
            counter_shm = shared_memory.SharedMemory(name=frame_counter_name)
            frame_counter = np.ndarray((1,), dtype=np.int32, buffer=counter_shm.buf)
            
            actual_res_shm = shared_memory.SharedMemory(name=actual_res_name)
            actual_res = np.ndarray((2,), dtype=np.int32, buffer=actual_res_shm.buf)
            
            write_idx = 0
            
            while True:
                # Check for raw image request (non-blocking)
                if not raw_request_queue.empty():
                    try:
                        raw_request_queue.get_nowait()
                    except:
                        pass  # race condition - request already consumed
                    else:
                        raw_img = img_gen.get_raw_image()
                        raw_response_queue.put(raw_img)
                
                img = img_gen.get_image()
                
                if buffers is None:
                    buffers = [
                        np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
                        for shm in shm_objects
                    ]
                    actual_res[:] = img.shape
                
                # Write to current buffer
                buffers[write_idx][:] = img
                frame_counter[0] += 1
                
                # Mark as ready, move to next buffer (simple ring)
                ready_idx[0] = write_idx
                write_idx = (write_idx + 1) % num_buffers
        except Exception as e:
            error_queue.put(str(e))
            raise
    
    def gen_image(self) -> np.ndarray:
        """Get latest complete frame. Returns VIEW into shared memory."""
        # Check for subprocess crash
        if not self._error_queue.empty():
            raise RuntimeError(f"Camera subprocess crashed: {self._error_queue.get_nowait()}")
        
        # if not a valid index - need to wait for the producer to initialise 
        while self._ready_idx_view[0] < 0 or self._actual_res_view[0] == 0:
            time.sleep(0.001)
        
        idx = int(self._ready_idx_view[0])
        frame_id = int(self._frame_counter_view[0])
        actual_shape = tuple(self._actual_res_view)
        
        img = np.ndarray(actual_shape, dtype=np.uint8, buffer=self.buffers[idx].buf)
        
        self._current_ticket = SharedMem_ImgTicket(
            index=idx,
            res=actual_shape,
            buf_size=self.frame_size,
            id=frame_id
        )
        
        self.last_img = img
        return img
    
    def __next__(self) -> np.ndarray:
        return self.gen_image()
    
    def get_mem_buffers(self) -> dict:
        return {i: self.buffers[i] for i in range(self.num_buffers)}
    
    def get_safe_mem_details(self) -> SharedMem_ImgTicket:
        if self._current_ticket is None:
            self.gen_image()
        return self._current_ticket
    
    def get_raw_image_sync(self, timeout: float = 2.0) -> np.ndarray:
        """Synchronously request and retrieve full color BGR image from subprocess.
        
        Blocks until subprocess responds with the image.
        """
        # Clear any stale response
        while not self._raw_response_queue.empty():
            self._raw_response_queue.get_nowait()
        
        # Send request
        self._raw_request_queue.put(True)
        
        # Poll for response, checking error queue each iteration
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            # Check for subprocess crash
            if not self._error_queue.empty():
                raise RuntimeError(f"Camera subprocess crashed: {self._error_queue.get_nowait()}")
            try:
                return self._raw_response_queue.get(block=True, timeout=0.05)
            except:
                pass
        raise TimeoutError(f"Raw image capture timed out after {timeout}s")


class Camera_async(Camera):
    
    def __init__(self, video_modes, imagegen_cls:ImageGenerator) -> None:
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

    def __init__(self, video_modes, imagegen_cls:ImageGenerator) -> None:
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
        self.relays = {}
        for relay, gpio in self.gun_config.RELAY_IO.items():
            self.relays[relay] = self.getOutputDevice(gpio)
            self.debouncers[relay] = Debounce()
            self.debouncers_1shot[relay] = Debounce()
            print(f"GPIO {gpio} set for relay {relay}")

    def set_relay(
            self,
            relaypos: int,
            state: bool):

        debouncer = self.debouncers[relaypos]

        if state:
            # function is called by the debouncer class if it thinks its ok to go
            return debouncer.trigger(lambda: self.relays[relaypos].on()) 
        else:
            return debouncer.trigger(lambda: self.relays[relaypos].off())

    def force_set_relay(
        self,
        relaypos: int,
        state: bool
    ):
        if state is True:
            self.relays[relaypos].on()
        else:
            self.relays[relaypos].off()
        return  None

    @abstractmethod
    def getOutputDevice():
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
    """
    image_id_to_use if you want to specify a specific image for testing, otherwise set to empty string or none
    
    Experiment with metaclasses - can we pass the metaclass to a downstream process
    and it succesfully instances it depending on flavour? would be useful
    if we have lots of similar classes that have to be passed around"""
    def __new__(cls, name, bases, attrs):
        if 'cam_name' not in attrs:
            raise TypeError(f"Class {name} must define 'cam_name'")
        if 'image_id_to_use' not in attrs:
            raise TypeError(f"Class {name} must define 'image_id_to_use'")
        def init(self, res):
            self.blank_image = np.zeros(tuple(reversed(res)), np.uint8)
            sorted_files = get_images_for_cam_pair(
                cam_name=self.cam_name,
                filters=["75MM"]#quadrocode_corners
            )
 
            repeats = 1000
            if self.image_id_to_use is not None:
                if len(self.image_id_to_use)>0:
                    sorted_files = [i for i in sorted_files if self.image_id_to_use in i]
                    repeats = 5 # just to get some feedback when running
                    if len(sorted_files) == 0 :
                        raise Exception(f"could not find image id {self.image_id_to_use}")
            sorted_files = reduce(lambda acc, s: acc + [s] * repeats, sorted_files, [])
            random.shuffle(sorted_files)
            self.cycled_files_generator = itertools.chain(
                itertools.repeat(None, 5),
                iter(sorted_files)
            )
            self.res = res

        def _get_image(self):
            img_to_load = next(self.cycled_files_generator)
            # we are preloading the first 5 images as otherwise
            # during system initialise these are not analysed
            if img_to_load is None:
                return self.blank_image.copy()
            img = cv2.imread(img_to_load)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.blank_image[:] = img
            time.sleep(0.03)
            return self.blank_image
        
        def get_raw_image(self):
            """Return color (BGR) image from library"""
            img_to_load = next(self.cycled_files_generator)
            if img_to_load is None:
                # Return a color version of blank image
                if len(self.blank_image.shape) == 2:
                    return cv2.cvtColor(self.blank_image.copy(), cv2.COLOR_GRAY2BGR)
                return self.blank_image.copy()
            
            img = cv2.imread(img_to_load)
            # Keep as color (BGR) - don't convert to grayscale
            if len(img.shape) == 2:
                # If somehow grayscale, convert to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            time.sleep(0.03)
            return img

        attrs['__init__'] = init
        attrs['_get_image'] = _get_image
        attrs['get_raw_image'] = get_raw_image
        
        return super().__new__(cls, name, bases, attrs)


class ImageLibrary_longrange(ImageGenerator, metaclass=ImageLibraryMeta):
    cam_name = "long"
    image_id_to_use = None#"3786"


class ImageLibrary_closerange(ImageGenerator, metaclass=ImageLibraryMeta):
    cam_name = "close"
    image_id_to_use = None#"3786"


class ImageLibrary(ImageGenerator):
    
    def __init__(self, res) -> None:
        self.blank_image = np.zeros(tuple(reversed(res)), np.uint8)
        imgfoler = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        self.images = images_in_folder(imgfoler, [".jpg"])

        self.images = [i for i in self.images if "quadrocode_corners" in i]

        self.res = res
        if len(self.images) < 1:
            raise Exception("could not find images in folder")

    def _get_image(self):
        img_to_load = random.choice(self.images)
        
        img = cv2.imread(img_to_load)
        print(f"img {img_to_load}")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.resize(img, tuple(self.res[0:2]))
        self.blank_image[:] = img
        return self.blank_image
    
    def get_raw_image(self):
        """Return color (BGR) image from library"""
        img_to_load = random.choice(self.images)
        img = cv2.imread(img_to_load)
        print(f"raw img {img_to_load}")
        # Keep as color (BGR) - don't convert to grayscale
        if len(img.shape) == 2:
            # If somehow grayscale, convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


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


    def _get_image(self):
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
    
    def get_raw_image(self):
        """Return color (BGR) image for test UI elements"""
        img_to_load = random.choice(self.images)
        img = cv2.imread(img_to_load)
        print(f"test_ui raw img {img_to_load}")
        # Keep as color (BGR) - don't convert to grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Apply test UI effects in color
        if self.image_freq < 15 or random.randint(0, 20) == 1:
            img[:] = 0
        return img



    
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
        # must be a bett
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

    @abstractmethod 
    def speak_blocking(self, message):
        pass

from collections import deque
from contextlib import contextmanager
import time


class Perfmonitor:
    def __init__(self) -> None:
        self.measurements = {}
        self.timers = {}
        self.start_time = {}
        self._active_sections = {}  # Track active section measurements
    
    def create_metric(self, metric_name):
        if metric_name not in self.measurements:
            self.measurements[metric_name] = deque(maxlen=10)
            self.timers[metric_name] = time.perf_counter()
            self.start_time[metric_name] = time.perf_counter()

    def get_time(self, metric_name, reset=False):
        self.create_metric(metric_name)
        elapsed = time.perf_counter() - self.start_time[metric_name]
        self.measurements[metric_name].append(elapsed * 1000)  # Convert to ms
        if reset:
            self.reset(metric_name)
        return elapsed * 1000  # Return ms

    def get_average(self, metric_name):
        self.create_metric(metric_name)
        if not self.measurements[metric_name]:
            return 0
        return sum(self.measurements[metric_name]) / len(self.measurements[metric_name])
    
    def reset(self, metric_name):
        self.start_time[metric_name] = time.perf_counter()
    
    def manual_measure(self, metric_name, time_ms):
        self.create_metric(metric_name)
        self.measurements[metric_name].append(time_ms)

    def start_section(self, section_name: str):
        """Start timing a specific section of code"""
        self._active_sections[section_name] = time.perf_counter()
    
    def end_section(self, section_name: str) -> float:
        """End timing a specific section and return elapsed time in ms"""
        if section_name not in self._active_sections:
            raise ValueError(f"Section '{section_name}' was never started")
        
        elapsed_ms = (time.perf_counter() - self._active_sections[section_name]) * 1000
        del self._active_sections[section_name]
        
        self.create_metric(section_name)
        self.measurements[section_name].append(elapsed_ms)
        return elapsed_ms

    @contextmanager
    def measure(self, metric_name: str):
        """Context manager for measuring execution time of a code block in milliseconds"""
        self.start_section(metric_name)
        try:
            yield
        finally:
            self.end_section(metric_name)


# class Perfmonitor:
#     def __init__(self) -> None:
#         self.measurements = {}
#         self.timers = {}
#         self.start_time = {}
    
#     def create_metric(self, metric_name):
#         if metric_name not in self.measurements:
#             self.measurements[metric_name] = deque(maxlen=10)
#             self.timers[metric_name] = time.perf_counter()
#             self.start_time[metric_name] = time.perf_counter()

#     def get_time(self, metric_name, reset=False):
#         self.create_metric(metric_name)
#         elapsed = time.perf_counter() - self.start_time[metric_name]
#         self.measurements[metric_name].append(elapsed * 1000)  # Convert to ms
#         if reset:
#             self.reset(metric_name)
#         return elapsed * 1000  # Return ms

#     def get_average(self, metric_name):
#         self.create_metric(metric_name)
#         if not self.measurements[metric_name]:
#             return 0
#         return sum(self.measurements[metric_name]) / len(self.measurements[metric_name])
    
#     def reset(self, metric_name):
#         self.start_time[metric_name] = time.perf_counter()
    
#     def manual_measure(self, metric_name, time_ms):
#         self.create_metric(metric_name)
#         self.measurements[metric_name].append(time_ms)

#     @contextmanager
#     def measure(self, metric_name: str):
#         """Context manager for measuring execution time of a code block in milliseconds"""
#         self.create_metric(metric_name)
#         self.reset(metric_name)
#         try:
#             yield
#         finally:
#             elapsed_ms = (time.perf_counter() - self.start_time[metric_name]) * 1000
#             self.measurements[metric_name].append(elapsed_ms)

class CardioGramDisplay:
    def __init__(self, pos_x, pos_y, width, height, value_range=(-1, 1), flow_direction=0):
        """
        pos_x, pos_y: Top-left position in the parent image.
        width, height: Size of the display region.
        value_range: Expected numeric range for metric values.
        flow_direction: Direction (in degrees) for history flow:
            0   -> New data at bottom (scrolls upward)
            180 -> New data at top (scrolls downward)
            90  -> New data at right (scrolls leftward)
            270 -> New data at left (scrolls rightward)
        """
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.width = width
        self.height = height
        self.value_range = value_range
        self.flow_direction = flow_direction % 360

        # Create an overlay with an alpha channel (BGRA)
        self.overlay = np.zeros((height, width, 4), dtype=np.uint8)
        self.metrics = {"test": self.Metric("test", 0)}
        
        # Pre-calculate the gradient matrix - OPTIMIZATION
        self._precalculate_gradient()
        
        # Create cache for colors - OPTIMIZATION
        self.color_cache = {}
        
        # Pre-allocate arrays for compositing - OPTIMIZATION
        self._temp_overlay = np.zeros_like(self.overlay)
        
    def _precalculate_gradient(self):
        """Pre-compute the gradient for fade effect - OPTIMIZATION"""
        if self.flow_direction in (0, 180):
            if self.flow_direction == 0:
                self.gradient = np.linspace(0, 1, self.height).reshape(self.height, 1)
            else:
                self.gradient = np.linspace(1, 0, self.height).reshape(self.height, 1)
        else:
            if self.flow_direction == 90:
                self.gradient = np.linspace(0, 1, self.width).reshape(1, self.width)
            else:
                self.gradient = np.linspace(1, 0, self.width).reshape(1, self.width)

    @dataclass
    class MetricUpdate:
        __slots__ = ['metric', 'color', 'slice_row', 'slice_col']
        metric: str
        color: tuple[int, int, int, int]
        slice_row: slice
        slice_col: slice

    @dataclass
    class Metric:
        __slots__ = ['metric', 'pos']
        metric: str
        pos: int

    def apply_image_actions(self, image, image_actions):
        """
        Applies a list of image actions to the given image.
        """
        for action in image_actions:
            image[action.slice_row, action.slice_col] = action.color

    def update_metrics(self, updates: dict[str, float]) -> list[MetricUpdate]:
        """
        OPTIMIZED with direct indexing and pre-allocated arrays
        """
        min_val, max_val = self.value_range
        offset_range = 60
        bar_thickness = 3
        
        # OPTIMIZATION: Use more efficient array shifting with direct slice assignment
        if self.flow_direction == 0:
            # Shift directly with NumPy's optimized memory handling
            self.overlay[:-1, :] = self.overlay[1:, :]
            self.overlay[-1:, :] = 0  # Clear the last row
            # Pre-calculate positions once
            edge_positions = np.arange(self.height - 1, self.height - offset_range, -1)
        elif self.flow_direction == 180:
            self.overlay[1:, :] = self.overlay[:-1, :]
            self.overlay[:1, :] = 0
            edge_positions = np.arange(0, offset_range)
        elif self.flow_direction == 90:
            self.overlay[:, :-1] = self.overlay[:, 1:]
            self.overlay[:, -1:] = 0
            edge_positions = np.arange(self.width - 1, self.width - offset_range, -1)
        elif self.flow_direction == 270:
            self.overlay[:, 1:] = self.overlay[:, :-1]
            self.overlay[:, :1] = 0
            edge_positions = np.arange(0, offset_range)
        
        # OPTIMIZATION: Pre-calculate 1/(max-min) for faster normalization
        norm_factor = 1.0 / (max_val - min_val)
        
        output_actions = []
        for cnt, (metric, value) in enumerate(updates.items()):
            if metric not in self.metrics:
                max_pos = max(self.metrics.values(), key=lambda m: m.pos).pos
                self.metrics[metric] = self.Metric(metric, max_pos + bar_thickness)
            pos = self.metrics[metric].pos
            
            # if metric not in self.color_cache:
            #     # Calculate colors (same as before)
            #     step = 255 // max(len(self.metrics), 1)
            #     red = (255 - cnt * step) % 256
            #     green = (cnt * step) % 256
            #     blue = (128 + cnt * step) % 256
            #     self.color_cache[metric] = (red, green, blue)
            
            # color = self.color_cache[metric]
            step = 255 // max(len(self.metrics), 1)
            red = (255 - cnt * step) % 256
            green = (cnt * step) % 256
            blue = (128 + cnt * step) % 256
            color = (red, green, blue)
            # OPTIMIZATION: Faster normalization with pre-calculated factor
            value = min(max(value, min_val), max_val)
            norm_val = (value - min_val) * norm_factor
            
            # OPTIMIZATION: Process flow directions with cleaner code
            if self.flow_direction == 0:
                new_x = int(norm_val * (self.width - 1))
                edge_pos = edge_positions[pos]
                # Draw 4 pixels instead of 1 for better visibility
                self.overlay[edge_pos, new_x] = (color[0], color[1], color[2], 255)
                if new_x > 0:
                    self.overlay[edge_pos, new_x-1] = (color[0], color[1], color[2], 255)
                if new_x < self.width-1:
                    self.overlay[edge_pos, new_x+1] = (color[0], color[1], color[2], 255)
                if edge_pos > 0:
                    self.overlay[edge_pos-1, new_x] = (color[0], color[1], color[2], 255)
                row_slice = slice(edge_pos-bar_thickness, edge_pos)
                col_slice = slice(0, new_x)
            elif self.flow_direction == 180:
                new_x = int(norm_val * (self.width - 1))
                edge_pos = edge_positions[pos]
                x_pos = self.width - new_x - 1
                # Draw 4 pixels instead of 1 for better visibility
                self.overlay[edge_pos, x_pos] = (color[0], color[1], color[2], 255)
                if x_pos > 0:
                    self.overlay[edge_pos, x_pos-1] = (color[0], color[1], color[2], 255)
                if x_pos < self.width-1:
                    self.overlay[edge_pos, x_pos+1] = (color[0], color[1], color[2], 255)
                if edge_pos < self.height-1:
                    self.overlay[edge_pos+1, x_pos] = (color[0], color[1], color[2], 255)
                row_slice = slice(edge_pos, edge_pos+bar_thickness)
                col_slice = slice(self.width - new_x - 1, self.width)
            elif self.flow_direction == 90:
                new_y = int(norm_val * (self.height - 1))
                edge_pos = edge_positions[pos]
                # Draw 4 pixels instead of 1 for better visibility
                self.overlay[new_y, edge_pos] = (color[0], color[1], color[2], 255)
                if new_y > 0:
                    self.overlay[new_y-1, edge_pos] = (color[0], color[1], color[2], 255)
                if new_y < self.height-1:
                    self.overlay[new_y+1, edge_pos] = (color[0], color[1], color[2], 255)
                if edge_pos > 0:
                    self.overlay[new_y, edge_pos-1] = (color[0], color[1], color[2], 255)
                row_slice = slice(0, new_y)
                col_slice = slice(edge_pos-bar_thickness, edge_pos)
            elif self.flow_direction == 270:
                new_y = int(norm_val * (self.height - 1))
                edge_pos = edge_positions[pos]
                y_pos = self.height - new_y - 1
                # Draw 4 pixels instead of 1 for better visibility
                self.overlay[y_pos, edge_pos] = (color[0], color[1], color[2], 255)
                if y_pos > 0:
                    self.overlay[y_pos-1, edge_pos] = (color[0], color[1], color[2], 255)
                if y_pos < self.height-1:
                    self.overlay[y_pos+1, edge_pos] = (color[0], color[1], color[2], 255)
                if edge_pos < self.width-1:
                    self.overlay[y_pos, edge_pos+1] = (color[0], color[1], color[2], 255)
                row_slice = slice(self.height - new_y, self.height)
                col_slice = slice(edge_pos, edge_pos+bar_thickness)
            
            output_actions.append(self.MetricUpdate(metric, (color[0], color[1], color[2], 255),
                                               slice_row=row_slice, slice_col=col_slice))
        return output_actions

    def get_overlay_with_gradient(self):
        """
        OPTIMIZED to minimize array operations and type conversions
        """
        # Use provided array reference if no modifications needed
        # OPTIMIZATION: Apply gradient with minimal memory operations
        np.multiply(self.overlay[:,:,3], self.gradient, out=self._temp_overlay[:,:,3], casting='unsafe')
        np.copyto(self._temp_overlay[:,:,0:3], self.overlay[:,:,0:3])
        
        return self._temp_overlay

    def composite_onto_inplace(self, background, image_actions):
        """
        FINAL OPTIMIZATION: Simplified blending with uint8 math instead of float conversions
        """
        # Get overlay with gradient and apply actions
        overlay = self.get_overlay_with_gradient()
        self.apply_image_actions(overlay, image_actions)
        
        # Extract region of interest
        h, w = self.height, self.width
        roi = background[self.pos_y:self.pos_y+h, self.pos_x:self.pos_x+w]
        
        # Apply fast box blur to ROI for smoother appearance
        # Use a 3x3 kernel for minimal performance impact
        # some random kernel for fun
        kernel = np.array([[1, 1, 1],
                          [9, 1, 1], 
                          [9, 9, 1]], dtype=np.float32) / 9.0
        
        # Apply blur only to the region we're about to modify
        blurred_roi = cv2.filter2D(roi, -1, kernel)
        
        # Find pixels with non-zero alpha
        alpha_mask = overlay[:,:,3] > 0
        
        # Only process if there are visible pixels
        if np.any(alpha_mask):
            # Fast integer-based alpha blending
            for c in range(3):
                # Use integer math (much faster on low-power CPUs like Raspberry Pi)
                # This avoids expensive float conversions
                roi[:,:,c][alpha_mask] = (
                    (overlay[:,:,c][alpha_mask].astype(np.uint16) * overlay[:,:,3][alpha_mask].astype(np.uint16) + 
                     blurred_roi[:,:,c][alpha_mask].astype(np.uint16) * (255 - overlay[:,:,3][alpha_mask].astype(np.uint16))) 
                    // 255
                ).astype(np.uint8)
        
        return background

class LumoUI:
    class pixel(int):
        pass
    @dataclass
    class EnemyArea:
        #measure manually from the non-orientated image 
        offset_y: int
        width: int
    @dataclass
    class StatusBarArea:
        name: str
        offset_y: int
        offset_x: int
        width: int
        height: int
        def implant_image(self, base_image: np.ndarray, image: np.ndarray):
            # use this to implant an image into a base image
            h, w = image.shape[:2]
            base_image[self.offset_y:self.offset_y+h, self.offset_x:self.offset_x+w] = image

    # class members
    shieldstatus_dims = HeightWidth(height=59,width=70)

    def __init__(self, filesystem_: Optional[FileSystemABC] = None) -> None:
        self.filesystem = filesystem_
        self._number_limit = 300
        self.statusbar_area_AMMO = self.StatusBarArea(
            name="AMMO",
            offset_y=0,
            offset_x=0,
            width=93,
            height=61
        )
        self.statusbar_area_HEALTH = self.StatusBarArea(
            name="HEALTH",
            offset_y=0,
            offset_x=93,
            width=114,
            height=61
        )
        self.enemy_avatar_area = self.EnemyArea(offset_y=285,width=70)
        self.enemy_info_area = self.EnemyArea(offset_y=357,width=108)
        self.statusbar_img = self.load_media_image("doom_statusbar_blank.jpg")
        self.numerics_img = self.load_media_image("doom_numerals_font.jpg")
        self.ammo_section_template = self.load_media_image("ammo_section_template.jpg")
        self._numberstatus_cache: dict[str, np.ndarray] = {}
        self._shieldstatus_cache: list[np.ndarray] = []


    def get_shield_status_img(self, normalised_health: float):
        if not(0 <= normalised_health <= 1):
            raise ValueError(f"normalised health out of range: {normalised_health}")

        if self._shieldstatus_cache is not None and len(self._shieldstatus_cache) > 0:
            # get normalised position in list
            normed = 1/(len(self._shieldstatus_cache)-1)
            index = int(normalised_health/normed)
            return self._shieldstatus_cache[index]

        # Only try persisted cache after first early-return check
        if self.filesystem is not None:
            loaded_shieldstatus = self.filesystem.load_shieldstatus_cache()
            if loaded_shieldstatus is not None and isinstance(loaded_shieldstatus, list) and len(loaded_shieldstatus) > 0:
                self._shieldstatus_cache = loaded_shieldstatus
                normed = 1/(len(self._shieldstatus_cache)-1)
                index = int(normalised_health/normed)
                return self._shieldstatus_cache[index]

        
        # arbitrary image generation steps
        for gen_shield_status in np.arange(0, 1.01, 0.05):
            
            shieldstatus_img = img_processing.create_health_bar(
                health_value=gen_shield_status,
                width=600,
                height=600,
                use_anti_aliasing=True,
                use_noise=True,
                high_health_color='green'
            )
            self._shieldstatus_cache.append(cv2.resize(
                shieldstatus_img,
                (self.shieldstatus_dims.width, self.shieldstatus_dims.height)
                ))
        # Persist the shield cache once generated
        if self.filesystem is not None and self._shieldstatus_cache:
            try:
                self.filesystem.save_shieldstatus_cache(self._shieldstatus_cache)
            except Exception as e:
                print(f"Warning: failed to save shieldstatus cache: {e}")
        
        return self.get_shield_status_img(normalised_health)

    def display_player_info(self, dims: HeightWidth, info: EnemyInfo, fade_norm: float = 1.0):
        """Generate and display player info text
        
        Draws text directly in display orientation. Green channel (or red if hurt/eliminated) for performance.
        Text layout: displayname on top, healthpoints below (vertically stacked)
        
        Args:
            dims: Display dimensions (height, width)
            fade_norm: Normalized fade value 0.0-1.0 (0=invisible, 1=full brightness)
            turn_red: If True, draw in red channel instead of green (for hurt/eliminated state)
        """
        # Skip drawing if essentially invisible
        if fade_norm < 0.01:
            return
        
        # Create black canvas directly at display dimensions
        canvas = np.zeros((dims.height, dims.width, 3), dtype=np.uint8)
        
        # Calculate font size based on canvas height
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = dims.height / 120.0  # Larger font for better visibility
        font_thickness = 2  # Thicker for better visibility
        
        # Convert healthpoints to string
        hp_text = str(info.health)
        
        # Calculate text sizes for vertical positioning
        (name_w, name_h), _ = cv2.getTextSize(info.displayname, font, font_scale, font_thickness)
        (hp_w, hp_h), _ = cv2.getTextSize(hp_text, font, font_scale, font_thickness)
        
        # Position texts vertically (stacked: displayname on top, healthpoints below)
        # Left-justified with small padding from left edge
        left_padding = 5
        
        # Top text (displayname) - left-justified, positioned in upper third
        name_x = left_padding
        name_y = dims.height // 3 + name_h // 2
        
        # Bottom text (healthpoints) - left-justified, positioned in lower third  
        hp_x = left_padding
        hp_y = 2 * dims.height // 3 + hp_h // 2
        
        # Choose color based on state (green=normal, red=hurt/eliminated)
        text_color = (0, 0, 255) if info.turn_red else (0, 255, 0)  # BGR format
        cv2.putText(canvas, info.displayname, (name_x, name_y), 
                   font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(canvas, hp_text, (hp_x, hp_y), 
                   font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        # Apply fade effect (same as display_player_image)
        faded_canvas = (canvas * fade_norm).astype(np.uint8)
        
        player_h, player_w = faded_canvas.shape[:2]
        
        # Clip to available space (statusbar_img height and enemy_info_area width)
        max_h = self.statusbar_img.shape[0] - 1  # -1 because we start at row 1
        max_w = self.enemy_info_area.width
        
        if player_h > max_h or player_w > max_w:
            raise ValueError(
                f"Generated player info image ({player_h}x{player_w}) doesn't fit in available space "
                f"({max_h}x{max_w}). Reduce dims (currently h={dims.height}, w={dims.width})."
            )
        
        # Place in enemy_info_area (centered)
        off = self.enemy_info_area.offset_y + int((self.enemy_info_area.width - player_w) / 2)
        self.statusbar_img[1:player_h+1, off:off+player_w] = faded_canvas[:]

    def display_player_image(self, playerimage: np.ndarray, normalised_fade: float, turn_red: bool = False):

        player_h, player_w = playerimage.shape[:2]
 
        faded_image = (playerimage * normalised_fade).astype(np.uint8)
        
        # If turn_red, combine all channels into red channel for bright red effect
        if turn_red:
            # Combine B+G+R channels into red channel only (very bright red)
            faded_image[:,:,2] = np.minimum(
                faded_image[:,:,0] + faded_image[:,:,1] + faded_image[:,:,2], 255
            )
            faded_image[:,:,0] = 0  # Clear blue
            faded_image[:,:,1] = 0  # Clear green
        
        try:
            # Place the faded image in the center of the status bar
            off = self.enemy_avatar_area.offset_y + int((self.enemy_avatar_area.width-player_w)/2)
            self.statusbar_img[
                1:player_h+1,
                off:player_w+off
                ] = faded_image[:]
        except Exception as e:
            raise Exception(f"check the sizes of the statusbar and the area the PlayerCard is permitted  {e}")

    def draw_status_bar(
            self,
            base_image: np.ndarray,
            ammo: int = None,
            normalised_shieldstatus: float  = None,
            health_pts: int = None):
        # Get dimensions of both images
        base_h, base_w = base_image.shape[:2]
        bar_h, bar_w = self.statusbar_img.shape[:2]
        

        # plop in the ammo section
        if health_pts is None:
            number_img = self.get_number_img("NONE", self.statusbar_area_HEALTH)
        else:
            number_img = self.get_number_img(health_pts, self.statusbar_area_HEALTH)
        self.statusbar_area_HEALTH.implant_image(self.statusbar_img, number_img)

        if ammo is not None:
            
            number_img = self.get_number_img(ammo, self.statusbar_area_AMMO)
            # h, w = number_img.shape[:2]
            self.statusbar_area_AMMO.implant_image(self.statusbar_img, number_img)
            # self.statusbar_img[0:h, 0:w] = number_img
        if normalised_shieldstatus is not None:
            status_img = self.get_shield_status_img(normalised_health=normalised_shieldstatus)
            h, w = status_img.shape[:2]
            offset_along_statusbar = 212
            offset_from_top = 2
            self.statusbar_img[offset_from_top:h+offset_from_top, offset_along_statusbar:w+offset_along_statusbar] = status_img

        # Calculate position to place the status bar (centered vertically)
        # After rotation, the dimensions will be swapped
        y_start = (base_h - bar_w) // 2  # Center vertically, using bar_w since it will be height after rotation
        x_start = 0
        
        # Create a view of the target region in base_image
        target_region = base_image[y_start:y_start + bar_w, x_start:x_start + bar_h]
        
        # Rotate the status bar first to get the correct dimensions
        rotated_bar = cv2.rotate(self.statusbar_img, cv2.ROTATE_90_CLOCKWISE)
        
        # Copy the rotated bar into the target region
        target_region[:] = rotated_bar

    @staticmethod
    def load_media_image(filename: str) -> np.ndarray:
        current_script_path = os.path.abspath(__file__)
        parent_dir = os.path.dirname(current_script_path)
        doom_statusbar_path = os.path.join(parent_dir,"media", filename)
        print(f"Opening media file {doom_statusbar_path}")
        try:
            img = cv2.imread(doom_statusbar_path)
        except Exception as e:
            raise Exception(f"could not load media file {filename}")
        if img is None:
            raise Exception(f"Img load fail {filename}")
        return img
        
    
    def get_number_img(self, number: int, area: StatusBarArea) -> np.ndarray:
        """Return an image for the incoming number
        create cache for range on first call
        provide an Area, this means the background of the number will be 
        harvested from the area in the base status bar image"""
        key_ = str(number) + area.name
        if key_ in self._numberstatus_cache:
            return self._numberstatus_cache[key_]
        # Only try persisted cache after first early-return check
        if self.filesystem is not None:
            loaded_numberstatus = self.filesystem.load_numberstatus_cache()
            if loaded_numberstatus is not None and isinstance(loaded_numberstatus, dict) and len(loaded_numberstatus) > 0:
                self._numberstatus_cache = loaded_numberstatus
                if key_ in self._numberstatus_cache:
                    return self._numberstatus_cache[key_]
        h, w = self.numerics_img.shape[:2]
        colcnt = 0
        spans = []
        for col in range(0, w):
            
            has_non_white = np.any(np.any(self.numerics_img[:, col, :] < 220, axis=1))
            if bool(has_non_white) is True:
                if colcnt == 0:
                    colcnt += 1
                    spans.append(col)
            elif bool(has_non_white) is False and colcnt > 0:
                spans.append(-col)
                colcnt = 0
        # get rid of empty space (assumed to be white)
        if spans[0] < 0:
            spans.pop(0)
        
        # now move in pairs and get start/end cols of each character
        if len(spans) % 2 != 0:
            spans.pop()
        known_chars = [1,2,3,4,5,6,7,8,9,0]
        char_img: dict[int, np.ndarray] = {}
        span_indexer = 0
        for index, char in enumerate(known_chars):
            start_col = abs(spans[span_indexer])
            end_col = abs(spans[span_indexer+1])
            char_img[char] = self.numerics_img[:, start_col:end_col, :]
            span_indexer += 2
            # cv2.imshow(f'Character Debug', char_img[char])
            # cv2.waitKey(0)
        # now create a dictionary of all the images
        for i in range(0, self._number_limit+1):
            num_as_str = str(i)
            temp_img = None
            for char in num_as_str:
                if temp_img is None:
                    temp_img = char_img[int(char)]
                else:
                    temp_img = np.concatenate((temp_img, char_img[int(char)]), axis=1)
            target_height = 32
            resize = target_height/temp_img.shape[0]
            isolated_number = cv2.resize(temp_img, None, fx=resize, fy=resize)
            if i < 20:
                # give warning glow when low, see what this looks like
                isolated_number = img_processing.apply_bloom_effect(isolated_number, threshold=80, blur_size=5, intensity=0.5)
            # Create mask for non-white pixels (number pixels vs background)
            mask = np.any(isolated_number < 200, axis=2)
            
            # Convert boolean mask to 3-channel uint8 image for visualization
            mask_visual = (mask.astype(np.uint8) * 255)
            # mask_visual_3ch = np.stack([mask_visual, mask_visual, mask_visual], axis=2)
            
            # Combine cached_img with background using mask as alpha layer
            # Use the StatusBarArea to extract background from status bar image
            background = self.statusbar_img[
                area.offset_y:area.offset_y + area.height,
                area.offset_x:area.offset_x + area.width
            ].copy()
            
            # # Debug display to show the extracted background
            # cv2.imshow(f'Background from {area.name} area', background)
            # cv2.waitKey(1)  # Show for 1ms, non-blocking
 
            h, w = isolated_number.shape[:2]
            
            # Put the text on the image
            composite = background.copy()
            h_comp, w_comp = composite.shape[:2]
            top_offset = 4
            width_offset = int((w_comp - w) / 2)
            composite[top_offset:h+top_offset, width_offset:w+ width_offset][mask] = isolated_number[mask]

            self._numberstatus_cache[str(i) + area.name] = composite

            # When we need a blank area
            if str("NONE" + area.name) not in self._numberstatus_cache:
                self._numberstatus_cache["NONE"+ area.name] = background.copy()
            # cv2.imshow(f'Character Debug', self._numberstatus_cache[str(i) + area.name] )
            # cv2.waitKey(0)
        print(f"total size for ammo image cache = {round(self.get_image_cache_size_mb(self._numberstatus_cache))} Mb")
        # Persist the number cache once generated
        if self.filesystem is not None and self._numberstatus_cache:
            try:
                self.filesystem.save_numberstatus_cache(self._numberstatus_cache)
            except Exception as e:
                print(f"Warning: failed to save numberstatus cache: {e}")
        if key_ not in self._numberstatus_cache:
            raise Exception(f"bad logic after generating and caching numbers:input = {number}")
        return self._numberstatus_cache[key_]


    @staticmethod
    def get_image_cache_size_mb(imagedict: dict[str, np.ndarray]) -> float:
        """
        Calculate the memory usage of the _numberstatus_cache dictionary in megabytes.
        """
        total_bytes = 0
        
        # Calculate size of numpy arrays
        for key, array in imagedict.items():
            if isinstance(array, np.ndarray):
                total_bytes += array.nbytes
            total_bytes += sys.getsizeof(key)
        
        # Add dictionary overhead
        total_bytes += sys.getsizeof(imagedict)
        
        # Convert to megabytes
        return total_bytes / (1024 * 1024)

    def save_caches_to_filesystem(self) -> bool:
        """Save both _numberstatus_cache and _shieldstatus_cache to filesystem if available.
        Returns True if both caches were saved successfully, False otherwise."""
        if self.filesystem is None:
            print("No filesystem available for cache saving")
            return False
        
        success = True
        
        # Save numberstatus cache
        if self._numberstatus_cache:
            if not self.filesystem.save_numberstatus_cache(self._numberstatus_cache):
                success = False
        
        # Save shieldstatus cache
        if self._shieldstatus_cache:
            if not self.filesystem.save_shieldstatus_cache(self._shieldstatus_cache):
                success = False
        
        return success

    def load_caches_from_filesystem(self) -> bool:
        """Load both _numberstatus_cache and _shieldstatus_cache from filesystem if available.
        Returns True if at least one cache was loaded successfully, False otherwise."""
        if self.filesystem is None:
            print("No filesystem available for cache loading")
            return False
        
        success = False
        
        # Load numberstatus cache
        loaded_numberstatus = self.filesystem.load_numberstatus_cache()
        if loaded_numberstatus is not None:
            self._numberstatus_cache = loaded_numberstatus
            print(f"Loaded numberstatus cache with {len(loaded_numberstatus)} entries")
            success = True
        
        # Load shieldstatus cache
        loaded_shieldstatus = self.filesystem.load_shieldstatus_cache()
        if loaded_shieldstatus is not None:
            self._shieldstatus_cache = loaded_shieldstatus
            print(f"Loaded shieldstatus cache with {len(loaded_shieldstatus)} entries")
            success = True
        
        return success


if __name__ == '__main__':

    test_ui = LumoUI()

    # run tests
    top = 0
    left = 0
    lower = 500
    right = 1000

    image_height = 500
    image_width = 1000
    rotated_points = img_processing.rotate_points_right_angle(
        ((top, left),(lower, right)),
        0,
        image_height,
        image_width
        )

    if rotated_points != [(0, 0), (500, 1000)]:
        raise Exception

    top = 0
    left = 0
    lower = 500
    right = 1000

    image_height = 500
    image_width = 1000
    rotated_points = img_processing.rotate_points_right_angle(
        ((top, left),(lower, right)),
        90,
        image_height,
        image_width
        )

    if rotated_points != [(1000, 0), (0, 500)]:
        raise Exception


    top = 0
    left = 0
    lower = 500
    right = 1000

    image_height = 500
    image_width = 1000
    rotated_points = img_processing.rotate_points_right_angle(
        ((top, left),(lower, right)),
        270,
        image_height,
        image_width
        )

    if rotated_points != [(0, 500), (1000, 0)]:
        raise Exception
    

    test_pos = ScreenNormalisedPositions(
        top=0,
        lower=1,
        left=0,
        right=1
    )

    res = test_pos.get_pixel_positions_with_ratio(img_shape=(500,1000), element_shape=(500,1000))
    assert res == ScreenPixelPositions(top=0, lower=500, left=0, right=1000)

    res = test_pos.get_pixel_positions_with_ratio(img_shape=(500,1000), element_shape=(100,100))
    assert res == ScreenPixelPositions(top=0, lower=500, left=0, right=500)

    res = test_pos.get_pixel_positions_with_ratio(img_shape=(500,1000), element_shape=(1000,100))
    assert res == ScreenPixelPositions(top=0, lower=500, left=0, right=50)

    res = test_pos.get_pixel_positions_with_ratio(img_shape=(1000,500), element_shape=(1000,100))
    assert res == ScreenPixelPositions(top=0, lower=1000, left=0, right=100)

    res = test_pos.get_pixel_positions_with_ratio(img_shape=(500, 1000), element_shape=(1000,2000))
    assert res == ScreenPixelPositions(top=0, lower=500, left=0, right=1000)

    res = test_pos.get_pixel_positions_with_ratio(img_shape=(500, 1000), element_shape=(1000,3000))
    assert res == ScreenPixelPositions(top=0, lower=333, left=0, right=1000)

    res = test_pos.get_pixel_positions_with_ratio(img_shape=(500, 1000), element_shape=(1000,300))
    assert res == ScreenPixelPositions(top=0, lower=500, left=0, right=150)

    

    img_shape=(480, 775, 3)
    test_pos = ScreenNormalisedPositions(top=0.9, lower=0.95, left=0.75, right=1)




    ui_element_shape = (32, 269)
    unrotated_canvas_shape = (775, 480, 3)
    plop = ScreenNormalisedPositions(top=0.61, lower=0.65, left=0.01, right=0.2)
    # will throw exception internally
    plop.get_pixel_positions_with_ratio(unrotated_canvas_shape, ui_element_shape)
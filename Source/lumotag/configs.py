from enum import Enum
from factory import gun_config
from my_collections import (
    ImagingMode,
    _OS,
    ScreenNormalisedPositions,
    UI_Element
)
from dataclasses import dataclass
import time





@dataclass
class base_find_lumotag_config():
    SAVE_IMAGES_DEBUG: bool
    SAVE_IMAGES_PATH: str
    PRINT_DEBUG: bool


def get_lumofind_config(platform):
    if platform == _OS.RASPBERRY:
        return base_find_lumotag_config(
            SAVE_IMAGES_DEBUG=False,
            SAVE_IMAGES_PATH=r"dunno",
            PRINT_DEBUG=False)
    elif platform == _OS.WINDOWS:
        return base_find_lumotag_config(
            SAVE_IMAGES_DEBUG=False,
            SAVE_IMAGES_PATH=r"D:/lumodebug/",
            PRINT_DEBUG=False)
    elif platform == _OS.MAC_OS:
        return base_find_lumotag_config(
            SAVE_IMAGES_DEBUG=True,
            SAVE_IMAGES_PATH=r"/Users/liell_p/lumodebug/",
            PRINT_DEBUG=True)
    else:
        raise Exception(f"Platform {platform} not supported")


class HQ_Cam_vidmodes(Enum):
    _3 = ImagingMode(
        camera_model="HQ_Cam",
        res_width_height=(2020, 1080),
        doc_description="2028 × 1080p50,",
        shared_mem_reversed=True,special_notes="")
    _2 = ImagingMode(
        camera_model="HQ_Cam",
        res_width_height=(1332, 990),
        doc_description="1332 × 990p120",
        shared_mem_reversed=True,special_notes="")
    _1 = ImagingMode(
        camera_model="HQ_Cam",
        res_width_height=(2020, 1520),
        doc_description="2028 × 1520p40",
        shared_mem_reversed=True,special_notes="")

# class HQ_Cam_vidmodes(Enum):
#     _2 = ["2028 × 1080p50,",(2020, 1080)] # 2.0MP  this is not losing res -  turn camera 90 degrees - probably want this one
#     _3 = ["1332 × 990p120",(1332, 990)] 
#     _1 = ["2028 × 1520p40",(2020, 1520)]

class HQ_GS_Cam_vidmodes(Enum):
    """global shutter model"""
    _1 = ImagingMode(
        camera_model="global shutter HQ camera",
        res_width_height=(1456, 1088),
        doc_description="1456 × 1088p50",
        shared_mem_reversed=True,special_notes="")

# class HQ_GS_Cam_vidmodes(Enum):
#     """global shutter model"""
#     _2 = ["1456 × 1088p50,",(1456, 1088)]
 
class Fake_Cam_vidmodes2(Enum):
    _1 = ImagingMode(
        camera_model="global shutter HQ camera",
        res_width_height=(1456, 1088),
        doc_description="2028 × 1080p50",
        shared_mem_reversed=False,special_notes="")

class Fake_Cam_vidmodes(Enum):
    _1 = ImagingMode(
        camera_model="test5005x500",
        res_width_height=(1456, 1088),
        doc_description="500 × 500",
        shared_mem_reversed=True,special_notes="")
    

class screensizes(Enum):
    title_bar_pxls = 25
    format = ("height", "width")
    tzar = (800 - title_bar_pxls, 480)
    windows_laptop = (500, 1200)
    stryker = (480, 620)


class stryker_config(gun_config):
    model = "STRYKER"
    def __init__(self) -> None:
        super().__init__()
        #for reference on rasperry pi 4
        self.RELAY_IO_BOARD = {1:29, 3:31, 2:16}
        self.RELAY_IO_BCM = {1:5, 3:6, 2:23}
        self.TRIGGER_IO_BOARD = {1:15, 2:13}
        self.TRIGGER_IO_BCM = {1:22, 2:27}

    @property
    def rly_torch(self):
        return 1

    @property
    def rly_triggerclick(self):
        return 2

    @property
    def RELAY_IO(self):
        return(self.RELAY_IO_BCM)
    
    @property
    def TRIGGER_IO(self):
        return (self.TRIGGER_IO_BCM)
    
    @property
    def screen_rotation(self):
        return(270)

    @property
    def screen_size(self):
        return(screensizes.stryker.value)

    def loop_wait(self):
        pass
    
    @property
    def light_strobe_cnt(self):
        return(0)
    
    @property
    def internal_img_crop(self):
        return((500,500))

    @property
    def img_subsmple_factor(self):
        return 2

    @property
    def opencv_window_pos(self):
        return(0, 0)

    @property
    def video_modes(self):
        return HQ_Cam_vidmodes

    @property
    def ui_overlay(self) -> dict:
        if self._UI_overlay is None:
            self._UI_overlay = {
                UI_Element.PHOTO.value:ScreenNormalisedPositions(top=0.4, lower=0.9, left=0.1, right=0.2),
                UI_Element.USER_ID.value:ScreenNormalisedPositions(top=0.1, lower=0.2, left=0.1, right=0.4),
                UI_Element.USER_INFO.value:ScreenNormalisedPositions(top=0.1, lower=0.9, left=0.7, right=0.9)
            }

        return self._UI_overlay


class TZAR_config(gun_config):
    model = "TZAR"
    def __init__(self) -> None:
        super().__init__()
        #for reference on rasperry pi 4
        self.RELAY_IO_BOARD = {1:29, 3:31, 2:16}
        self.RELAY_IO_BCM = {1:5, 3:6, 2:23}
        self.TRIGGER_IO_BOARD = {1:15, 2:13}
        self.TRIGGER_IO_BCM = {1:22, 2:27}

    @property
    def rly_torch(self):
        return 1

    @property
    def rly_triggerclick(self):
        return 2

    @property
    def RELAY_IO(self):
        return(self.RELAY_IO_BCM)

    @property
    def TRIGGER_IO(self):
        return (self.TRIGGER_IO_BCM)
    
    @property
    def screen_rotation(self):
        return(0)

    @property
    def screen_size(self):
        return(screensizes.tzar.value)

    def loop_wait(self):
        #time.sleep(0.1)
        return

    def cam_processing(self, inputimg):
        return inputimg

    @property
    def light_strobe_cnt(self):
        return(4)

    @property
    def internal_img_crop(self):
        return((500,500))

    @property
    def img_subsmple_factor(self):
        return 4

    @property
    def opencv_window_pos(self):
        return(640, 0)

    @property
    def video_modes(self):
        return HQ_GS_Cam_vidmodes

    @property
    def ui_overlay(self) -> dict:
        if self._UI_overlay is None:
            self._UI_overlay = {
                UI_Element.PHOTO.value:ScreenNormalisedPositions(top=0.4, lower=0.9, left=0.1, right=0.2),
                UI_Element.USER_ID.value:ScreenNormalisedPositions(top=0.1, lower=0.2, left=0.1, right=0.4),
                UI_Element.USER_INFO.value:ScreenNormalisedPositions(top=0.1, lower=0.9, left=0.7, right=0.9)
            }

        return self._UI_overlay

class simitzar_config(gun_config):
    model = "SIMITZAR"
    def __init__(self) -> None:
        super().__init__()
        #for reference on rasperry pi 4
        self.RELAY_IO_BOARD = {1:29, 3:31, 2:16}
        self.RELAY_IO_BCM = {1:5, 3:6, 2:23}
        self.TRIGGER_IO_BOARD = {1:15, 2:13}
        self.TRIGGER_IO_BCM = {1:22, 2:27}

    @property
    def rly_torch(self):
        return 1

    @property
    def rly_triggerclick(self):
        return 2
        
    @property
    def RELAY_IO(self):
        return(self.RELAY_IO_BCM)
    
    @property
    def TRIGGER_IO(self):
        return (self.TRIGGER_IO_BCM)
    
    @property
    def screen_rotation(self):
        return(0)

    @property
    def screen_size(self):
        return(screensizes.windows_laptop.value)

    def loop_wait(self):
        time.sleep(0.05)

    @property
    def light_strobe_cnt(self):
        return(0)

    @property
    def internal_img_crop(self):
        return((500,600))

    @property
    def img_subsmple_factor(self):
        return 4

    @property
    def opencv_window_pos(self):
        return(0, 0)

    @property
    def video_modes(self):
        return Fake_Cam_vidmodes

    @property
    def ui_overlay(self) -> dict:
        if self._UI_overlay is None:
            self._UI_overlay = {
                UI_Element.PHOTO.value:ScreenNormalisedPositions(top=0.1, lower=0.4, left=0.1, right=0.3),
                UI_Element.USER_ID.value:ScreenNormalisedPositions(top=0.1, lower=0.2, left=0.1, right=0.4),
                UI_Element.USER_INFO.value:ScreenNormalisedPositions(top=0.1, lower=0.9, left=0.7, right=0.9)
            }

        return self._UI_overlay
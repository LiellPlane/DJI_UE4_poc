from enum import Enum
from factory import gun_config, custom_dynamic_UI_element_callback
from my_collections import (
    ImagingMode,
    _OS,
    ScreenNormalisedPositions,
    UI_Element,
    UI_Behaviour_static,
    UI_Behaviour_dynamic,
    CHANNEL_COLOUR
)
from dataclasses import dataclass
import time



@dataclass
class base_find_lumotag_config():
    SAVE_IMAGES_DEBUG: bool
    SAVE_IMAGES_PATH: str
    PRINT_DEBUG: bool
    SAVE_STREAM: bool


def get_lumofind_config(platform):
    if platform == _OS.RASPBERRY:
        return base_find_lumotag_config(
            SAVE_IMAGES_DEBUG=False,
            SAVE_IMAGES_PATH=r"dunno",
            PRINT_DEBUG=False,
            SAVE_STREAM=False
            )
    elif platform == _OS.WINDOWS:
        return base_find_lumotag_config(
            SAVE_IMAGES_DEBUG=False,
            SAVE_IMAGES_PATH=r"D:/lumodebug/",
            PRINT_DEBUG=False,
            SAVE_STREAM=True
            )
    elif platform == _OS.MAC_OS:
        return base_find_lumotag_config(
            SAVE_IMAGES_DEBUG=False,
            SAVE_IMAGES_PATH=r"/Users/liell_p/lumodebug/",
            PRINT_DEBUG=False,
            SAVE_STREAM=False
            )
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


class RPICAMv2Noir_Cam_vidmodes(Enum):
    """raspberry pi v2 model
1 : imx219 [3280x2464 10-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@80000/imx219@10)
    Modes: 'SRGGB10_CSI2P' : 640x480 [206.65 fps - (1000, 752)/1280x960 crop]
                             1640x1232 [41.85 fps - (0, 0)/3280x2464 crop]
                             1920x1080 [47.57 fps - (680, 692)/1920x1080 crop]
                             3280x2464 [21.19 fps - (0, 0)/3280x2464 crop]
           'SRGGB8' : 640x480 [206.65 fps - (1000, 752)/1280x960 crop]
                      1640x1232 [83.70 fps - (0, 0)/3280x2464 crop]
                      1920x1080 [47.57 fps - (680, 692)/1920x1080 crop]
                      3280x2464 [21.19 fps - (0, 0)/3280x2464 crop]
                      """
    _1 = ImagingMode(
        camera_model="raspberry pi v2 model",
        res_width_height=(1920, 1080),
        doc_description="1920x1080 47fps",
        shared_mem_reversed=True,special_notes="")
    _2 = ImagingMode(
        camera_model="raspberry pi v2 model",
        res_width_height=(1640, 1232),
        doc_description="1640x1232 83fps binned",
        shared_mem_reversed=True,special_notes="")


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

class Fake_Cam_vidmodes_longrangeFILES(Enum):
    """these have to res match the files we have captured"""
    _1 = ImagingMode(
        camera_model="test5005x500",
        res_width_height=(1456, 1088),
        doc_description="500 × 500",
        shared_mem_reversed=True,special_notes="")
    
class Fake_Cam_vidmodes_closerangeFILES(Enum):
    """these have to res match the files we have captured"""
    _1 = ImagingMode(
        camera_model="test1280*1920",
        res_width_height=(1920, 1080),
        doc_description="test1280*1920",
        shared_mem_reversed=True,special_notes="")
    

class screensizes(Enum):
    title_bar_pxls = 25
    format = ("height", "width")
    tzar = (800 - title_bar_pxls, 480)
    windows_laptop = (400, 800)
    stryker = (480, 620)


# class stryker_config(gun_config):
#     model = "STRYKER"
#     def __init__(self) -> None:
#         super().__init__()
#         #for reference on rasperry pi 4
#         self.RELAY_IO_BOARD = {1:29, 3:31, 2:16}
#         self.RELAY_IO_BCM = {1:5, 3:6, 2:23}
#         self.TRIGGER_IO_BOARD = {1:15, 2:13}
#         self.TRIGGER_IO_BCM = {1:22, 2:27}

#     @property
#     def button_torch(self):
#         return 1

#     @property
#     def button_trigger(self):
#         return 2

#     @property
#     def button_rear(self):
#         return 3

#     @property
#     def RELAY_IO(self):
#         return(self.RELAY_IO_BCM)
    
#     @property
#     def TRIGGER_IO(self):
#         return (self.TRIGGER_IO_BCM)
    
#     @property
#     def screen_rotation(self):
#         return(270)

#     @property
#     def screen_size(self):
#         return(screensizes.stryker.value)

#     def loop_wait(self):
#         pass
    
#     @property
#     def light_strobe_cnt(self):
#         return(0)
    
#     @property
#     def internal_img_crop_lr(self):
#         return((500,500))
#     @property
#     def internal_img_crop_sr(self):
#         return((500,500))
#     @property
#     def img_subsmple_factor(self):
#         return 2

#     @property
#     def opencv_window_pos(self):
#         return(0, 0)

#     @property
#     def video_modes(self):
#         return HQ_Cam_vidmodes
#     @property
#     def video_modes_closerange(self):
#         return RPICAMv2Noir_Cam_vidmodes
#     @property
#     def ui_overlay(self) -> dict:
#         if self._UI_overlay is None:
#             self._UI_overlay = {
#                 UI_Element.PHOTO.value:ScreenNormalisedPositions(top=0.4, lower=0.9, left=0.1, right=0.2),
#                 UI_Element.USER_ID.value:ScreenNormalisedPositions(top=0.1, lower=0.2, left=0.1, right=0.4),
#                 UI_Element.USER_INFO.value:ScreenNormalisedPositions(top=0.1, lower=0.9, left=0.7, right=0.9)
#             }

#         return self._UI_overlay


class TZAR_config(gun_config):
    model = "TZAR"
    def __init__(self) -> None:
        super().__init__()
        #for reference on rasperry pi 4
        self.RELAY_IO_BOARD = {1:29, 3:31, 2:16}
        self.RELAY_IO_BCM = {1:5, 3:6, 2:23}
        self.TRIGGER_IO_BOARD = {1:15, 2:13, 3:11}
        self.TRIGGER_IO_BCM = {1:22, 2:27, 3:17}

    @property
    def button_torch(self):
        return 3

    @property
    def button_trigger(self):
        return 2

    @property
    def button_rear(self):
        return 1

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
    def internal_img_crop_lr(self):
        return((500,500))
    @property
    def internal_img_crop_sr(self):
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
    def video_modes_closerange(self):
        return RPICAMv2Noir_Cam_vidmodes
    


    # @property
    # def ui_overlay(self) -> dict:
    #     if self._UI_overlay is None:
    #         self._UI_overlay = common_ui_overlay

    #     return self._UI_overlay


class simitzar_config(gun_config):
    model = "SIMITZAR"

    def __init__(self) -> None:
        super().__init__()
        #for reference on rasperry pi 4
        self.RELAY_IO_BOARD = {1:29, 3:31, 2:16}
        self.RELAY_IO_BCM = {1:5, 3:6, 2:23}
        self.TRIGGER_IO_BOARD = {1:15, 2:13, 3:11}
        self.TRIGGER_IO_BCM = {1:22, 2:27, 3:17}

    @property
    def button_torch(self):
        return 1

    @property
    def button_trigger(self):
        return 2

    @property
    def button_rear(self):
        return 3

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
        return(screensizes.tzar.value)

    def loop_wait(self):
        time.sleep(0.05)

    @property
    def light_strobe_cnt(self):
        return(0)

    @property
    def internal_img_crop_lr(self):
        return((500,500))
    @property
    def internal_img_crop_sr(self):
        return((600,600))

    @property
    def img_subsmple_factor(self):
        return 4

    @property
    def opencv_window_pos(self):
        return(0, 0)

    @property
    def video_modes(self):
        return Fake_Cam_vidmodes_longrangeFILES

    @property
    def video_modes_closerange(self):
        return Fake_Cam_vidmodes_closerangeFILES

    # @property
    # def ui_overlay(self) -> dict:
    #     if self._UI_overlay is None:
    #         self._UI_overlay = common_ui_overlay

    #     return self._UI_overlay


otherPlayers_ui_overlay = {
                UI_Element.PHOTO.value: UI_Behaviour_static(
                    screen_normed_pos=ScreenNormalisedPositions(
                        top=0.25,
                        lower=0.5,
                        left=0.01,
                        right=0.2
                        ),
                    channel=CHANNEL_COLOUR.BLUE_CHANNEL.value,
                    border=False
                ),
                UI_Element.USER_ID.value: UI_Behaviour_static(
                    screen_normed_pos=ScreenNormalisedPositions(
                        top=0.55,
                        lower=0.60,
                        left=0.01,
                        right=0.2
                        ),
                    channel=CHANNEL_COLOUR.BLUE_CHANNEL.value,
                    border=False
                ),
                UI_Element.USER_INFO.value: UI_Behaviour_static(
                    screen_normed_pos=ScreenNormalisedPositions(
                        top=0.61,
                        lower=0.65,
                        left=0.01,
                        right=0.2
                        ),
                    channel=CHANNEL_COLOUR.BLUE_CHANNEL.value,
                    border=False
                ),
                UI_Element.BARMETRIC_RL.value: UI_Behaviour_dynamic(
                    screen_normed_pos=ScreenNormalisedPositions(
                        top=0.51,
                        lower=0.54,
                        left=0.01,
                        right=0.2
                        ),
                    channel_A=CHANNEL_COLOUR.GREEN_CHANNEL.value,
                    channel_B=CHANNEL_COLOUR.RED_CHANNEL.value,
                    cut_off_value_norm=0.25,
                    border=False
                ),
            }


Player_ui_overlay = {
                UI_Element.USER_ID.value: UI_Behaviour_static(
                    screen_normed_pos=ScreenNormalisedPositions(
                        top=0.80,
                        lower=0.85,
                        left=0.5,
                        right=0.6
                        ),
                    channel=CHANNEL_COLOUR.BLUE_CHANNEL.value,
                    border=False
                ),
                UI_Element.USER_INFO.value: UI_Behaviour_static(
                    screen_normed_pos=ScreenNormalisedPositions(
                        top=0.85,
                        lower=0.9,
                        left=0.5,
                        right=0.6
                        ),
                    channel=CHANNEL_COLOUR.BLUE_CHANNEL.value,
                    border=False
                ),
                UI_Element.ENERGY_LR.value: UI_Behaviour_dynamic(
                    screen_normed_pos=ScreenNormalisedPositions(
                        top=0.7,
                        lower=0.73,
                        left=0.5,
                        right=0.6
                        ),
                    channel_A=CHANNEL_COLOUR.BLUE_CHANNEL.value,
                    channel_B=CHANNEL_COLOUR.RED_CHANNEL.value,
                    cut_off_value_norm=0.50,
                    border=False
                ),
                UI_Element.BARMETRIC_LR.value: UI_Behaviour_dynamic(
                    screen_normed_pos=ScreenNormalisedPositions(
                        top=0.73,
                        lower=0.76,
                        left=0.5,
                        right=0.6
                        ),
                    channel_A=CHANNEL_COLOUR.GREEN_CHANNEL.value,
                    channel_B=CHANNEL_COLOUR.RED_CHANNEL.value,
                    cut_off_value_norm=0.25,
                    border=False
                ),
            }
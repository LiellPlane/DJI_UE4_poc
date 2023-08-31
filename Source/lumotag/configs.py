from enum import Enum
from factory import gun_config
import time


class HQ_Cam_vidmodes(Enum):
    # always width/ height
    _2 = ["2028 × 1080p50,",(2020, 1080)] # 2.0MP  this is not losing res -  turn camera 90 degrees - probably want this one
    _3 = ["1332 × 990p120",(1332, 990)] 
    _1 = ["2028 × 1520p40",(2020, 1520)]


class HQ_GS_Cam_vidmodes(Enum):
    """global shutter model"""
    # always width/ height
    _2 = ["1456 × 1088p50,",(1456, 1088)]


class Fake_Cam_vidmodes(Enum):
    # always width/ height
    _2 = ["1456 × 500,",(1456, 500)]


class screensizes(Enum):
    title_bar_pxls = 25
    format = ("height", "width")
    tzar = (800 - title_bar_pxls, 480)
    windows_laptop = (800 - title_bar_pxls, 480)
    stryker = (480, 800 - title_bar_pxls)


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
    def opencv_window_pos(self):
        return(200, 200)

    @property
    def video_modes(self):
        return HQ_Cam_vidmodes


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
    def opencv_window_pos(self):
        return(640, 0)

    @property
    def video_modes(self):
        return HQ_GS_Cam_vidmodes


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
        time.sleep(0.3)

    @property
    def light_strobe_cnt(self):
        return(0)

    @property
    def internal_img_crop(self):
        return((500,500))

    @property
    def opencv_window_pos(self):
        return(0, 0)

    @property
    def video_modes(self):
        return Fake_Cam_vidmodes

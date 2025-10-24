import enum
from dataclasses import dataclass
from typing import Union
from numpy import ndarray
from typing import Any
from typing import Annotated

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class Edges(str, enum.Enum):
    TOP = "TOP"
    LOWER = "LOWER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class LensConfigs(str, enum.Enum):
    DAISYBANK_HQ = "DAISYBANK_HQ",
    DAISYBANK_LQ =  "DAISYBANK_LQ"


class LEDColours(tuple, enum.Enum):
    Black = (0,0,0)
    White = (255,255,255)
    Red = (0,0,255)
    Green = (0,255,0)
    Blue = (255,0,0)
    Yellow = (0,255,255)
    Magenta = (255,0,255)
    Cyan = (255,255,0)

def u8_validator(value: int) -> bool:
    return 0 <= value <= 255

def u16_validator(value: int) -> bool:
    return 0 <= value <= 65535

U8 = Annotated[int, u8_validator]
U16 = Annotated[int, u16_validator]
@dataclass
class Scambi_unit_LED_only():
    """cheat class so we don't have to pass the whole
    object to a class which expects these members
    
    not the best place for this but had to avoid circular imports"""
    colour:  tuple[U8, U8, U8]
    physical_led_pos: list[U16]
# struct ScambiUnitLedOnly {
#     colour: Vec<u8>,
#     physical_led_pos: Vec<u16>,
# }

@dataclass
class LedSpacing():
    positionxy: tuple[int, int]
    edge: Edges
    normed_pos_along_edge_mid: float
    normed_pos_along_edge_start: float
    normed_pos_along_edge_end: float


@dataclass
class lens_details():
    id: LensConfigs
    width: int
    height: int
    fish_eye_circle: int
    #corners: list[int]
    targets: list[int] = None
    def __post_init__(self):
        self.targets = [
        [0, 0],
        [self.width,0 ],
        [self.width, self.height],
        [0, self.height]]


@dataclass
class LedsLayout():
    """position facing viewer"""
    clockwise_start: int
    clockwise_end: int
    EDGE: str


@dataclass
class config_corner():
    flat_corner: list[int, int]
    real_corner: list[int, int]


@dataclass
class config_regions():
    no_leds_vert: int
    no_leds_horiz: int
    move_in_horiz: float
    move_in_vert: float
    sample_area_edge : int
    subsample_cut: int # min edge pxls, we can subsample areas of image to speed up, but we don't want to subsample small areas into nothing
    def __post_init__(self):
        self.no_leds_vert = clamp(self.no_leds_vert, 5, 100)
        self.no_leds_horiz = clamp(self.no_leds_horiz, 5, 100)
        self.move_in_horiz = clamp(self.move_in_horiz, 0, 1.5)
        self.move_in_vert = clamp(self.move_in_vert, 0, 1.5)
        self.sample_area_edge = clamp(self.sample_area_edge, 10, 1000)
        self.subsample_cut = clamp(self.subsample_cut, 10, 1000)

@dataclass
class clicked_xy():
    clickX: Union[int, str]
    clickY: Union[int, str]
    def __post_init__(self):
        self.clickX = int(self.clickX)
        self.clickY = int(self.clickY)


@dataclass
class External_Config():
    # user can click in any order
    fish_eye_clicked_corners: list[clicked_xy]


@dataclass
class PhysicalTV_details():
    edges: dict[str:LedsLayout]
    #receiver_hostname = 'LiellOMEN.broadband'
    receiver_hostname = 'scambilightled.broadband'
    #receiver_hostname = '127.0.0.1'
    port = 12345


@dataclass
class AllConfiguration():
    lens_details: lens_details
    clicked_corners: External_Config
    sample_regions: config_regions
    physical_tv_details: PhysicalTV_details

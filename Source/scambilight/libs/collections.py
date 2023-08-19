import enum
from dataclasses import dataclass

class Edges(str, enum.Enum):
    TOP = "TOP"
    LOWER = "LOWER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

class LensConfigs(str, enum.Enum):
    DAISYBANK_HQ = "DAISYBANK_HQ",
    DAISYBANK_LQ =  "DAISYBANK_LQ"

@dataclass
class LedSpacing():
    positionxy: tuple[int, int]
    edge: Edges
    normed_pos_along_edge_mid: float
    normed_pos_along_edge_start: float
    normed_pos_along_edge_end: float

@dataclass
class lens_details():
    id: str
    vid: str
    width: int
    height: int
    fish_eye_circle: int
    corners: list[int]
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

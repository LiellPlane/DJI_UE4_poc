import enum
from libs.collections import (
    Edges,
    lens_details,
    LedsLayout,
    config_regions,
    LensConfigs)

# good youtube videos to try:
#fluid sim
#the batmoan batmobile chase scene 2022
#4k background

SCAMILIGHT_API = "https://api.scambilight.com/hello"


class UploadImageTypes(enum.Enum):
    RAW = "image_raw"
    PERPWARPED = "perpwarp"
    OVERLAY = "image_overlay"


class ScambiLight_Cam_vidmodes(enum.Enum):
    """scambilight fisheye ov5647"""
    # dimensions are reversed (h, w) due to quirk of ov5647
    _1 = ["1296x972 [43.25 fps - (0, 0)/2592x1944 crop]",(972, 1296 , 3)]
    _2 = ["640x480 [58.92 fps - (16, 0)/2560x1920 crop]",(480, 640, 3)]
    _3 = ["1920x1080 [30.62 fps - (348, 434)/1928x1080 crop]",(1080, 1920 , 3)]
    _4 = ["2592x1944 [15.63 fps - (0, 0)/2592x1944 crop]",(1944, 2592, 3)]


class DaisybankLedSpacing():
    def __init__(self) -> None:
        self.edges = {
            Edges.LEFT: LedsLayout(
                clockwise_start=73, clockwise_end=110),
            Edges.TOP: LedsLayout(
                clockwise_start=114, clockwise_end=181),
            Edges.RIGHT: LedsLayout(
                clockwise_start=185, clockwise_end=223),
            Edges.LOWER: LedsLayout(
                clockwise_start=229, clockwise_end=296)}


def get_lens_details(lens: LensConfigs) -> lens_details:
    if lens == LensConfigs.DAISYBANK_HQ:
        return lens_details(
            id= lens,
            width= 1269,
            height= 972,
            fish_eye_circle= 1250,
            corners= [[81, 234],[1157, 346],[860, 600],[363, 572]]
        )
    if lens == LensConfigs.DAISYBANK_LQ:
        return lens_details(
            id= lens,
            width= 640,
            height= 480,
            fish_eye_circle= 600,
            corners= [[18, 114], [587, 173], [431, 301], [176, 285]]
            )

    raise Exception("lens config not matched", lens)
   
def get_sample_regions_config() -> config_regions:
    return config_regions(
        no_leds_vert= 11,
        no_leds_horiz= 20,
        move_in_horiz= 0.1,
        move_in_vert= 0.1,
        sample_area_edge= 100,
        subsample_cut=15
    )

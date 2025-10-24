import enum
from libs.collections import (
    Edges,
    lens_details,
    LedsLayout,
    config_regions,
    LensConfigs,
    PhysicalTV_details)
from dataclasses import dataclass
from my_collections import ImagingMode

# good youtube videos to try:
#fluid sim
#the batmoan batmobile chase scene 2022
#4k background

SCAMILIGHT_API = "https://api.scambilight.com/hello"
RPI_ROOTDIR = "/home/scambilight/"
CONFIG_FILENAME = "configfile.json"
SESSIONTOKEN_FILENAME = "session_token.json"


class UploadImageTypes(enum.Enum):
    """keywords for main lambda"""
    RAW = "image_raw"
    PERPWARPED = "perpwarp"
    OVERLAY = "image_overlay"


class ScambiLight_Cam_vidmodes(enum.Enum):
    """scambilight fisheye ov5647"""
    _2 = ImagingMode(
        camera_model="fisheye ov5647",
        res_width_height=(480, 640 , 3),
        doc_description="640x480 [58.92 fps - (16, 0)/2560x1920 crop]",
        shared_mem_reversed=False,special_notes="dimensions are reversed (h, w) due to quirk of ov5647")
    _1 = ImagingMode(
        camera_model="fisheye ov5647",
        res_width_height=(972, 1296 , 3),
        doc_description="1296x972 [43.25 fps - (0, 0)/2592x1944 crop]",
        shared_mem_reversed=False,special_notes="dimensions are reversed (h, w) due to quirk of ov5647")

    _3 = ImagingMode(
        camera_model="fisheye ov5647",
        res_width_height=(1080, 1920 , 3),
        doc_description="1920x1080 [30.62 fps - (348, 434)/1928x1080 crop]",
        shared_mem_reversed=False,special_notes="dimensions are reversed (h, w) due to quirk of ov5647")
    _4 = ImagingMode(
        camera_model="fisheye ov5647",
        res_width_height=(1944, 2592, 3),
        doc_description="2592x1944 [15.63 fps - (0, 0)/2592x1944 crop]",
        shared_mem_reversed=False,special_notes="dimensions are reversed (h, w) due to quirk of ov5647")


# class ScambiLight_Cam_vidmodes(enum.Enum):
#     """scambilight fisheye ov5647"""
#     # dimensions are reversed (h, w) due to quirk of ov5647
#     _1 = ["1296x972 [43.25 fps - (0, 0)/2592x1944 crop]",(972, 1296 , 3)]
#     _2 = ["640x480 [58.92 fps - (16, 0)/2560x1920 crop]",(480, 640, 3)]
#     _3 = ["1920x1080 [30.62 fps - (348, 434)/1928x1080 crop]",(1080, 1920 , 3)]
#     _4 = ["2592x1944 [15.63 fps - (0, 0)/2592x1944 crop]",(1944, 2592, 3)]




DaisybankLedSpacing = PhysicalTV_details(
        edges = {
            Edges.LEFT: LedsLayout(
                clockwise_start=73, clockwise_end=110, EDGE=Edges.LEFT.value),
            Edges.TOP: LedsLayout(
                clockwise_start=114, clockwise_end=184, EDGE=Edges.TOP.value),
            Edges.RIGHT: LedsLayout(
                clockwise_start=185, clockwise_end=225, EDGE=Edges.RIGHT.value),
            Edges.LOWER: LedsLayout(
                clockwise_start=229, clockwise_end=296, EDGE=Edges.LOWER.value)}
)

# class DaisybankLedSpacing():
#     def __init__(self) -> None:
#         self.edges = {
#             Edges.LEFT: LedsLayout(
#                 clockwise_start=73, clockwise_end=110, EDGE=Edges.LEFT.value),
#             Edges.TOP: LedsLayout(
#                 clockwise_start=114, clockwise_end=181, EDGE=Edges.TOP.value),
#             Edges.RIGHT: LedsLayout(
#                 clockwise_start=185, clockwise_end=223, EDGE=Edges.RIGHT.value),
#             Edges.LOWER: LedsLayout(
#                 clockwise_start=229, clockwise_end=296, EDGE=Edges.LOWER.value)}
#         self.receiver_hostname = 'scambilightled.broadband'
#         self.port = 12345

# class DaisybankLedSpacing():
#     def __init__(self) -> None:
#         self.edges = {
#             Edges.LEFT: LedsLayout(
#                 clockwise_start=73, clockwise_end=110),
#             Edges.TOP: LedsLayout(
#                 clockwise_start=114, clockwise_end=181),
#             Edges.RIGHT: LedsLayout(
#                 clockwise_start=185, clockwise_end=223),
#             Edges.LOWER: LedsLayout(
#                 clockwise_start=229, clockwise_end=296)}
#         self.receiver_hostname = 'scambilightled.broadband'
#         self.port = 12345

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

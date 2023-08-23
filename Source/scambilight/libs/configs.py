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

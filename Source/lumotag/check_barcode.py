import numpy as np
import os
import pickle
import cv2
import random
from dataclasses import dataclass
from typing import Union, List, Optional
from functools import reduce
from enum import Enum, auto
from math import floor

class CodeSegment(Enum):
    CORNER: int = 1
    MIDLINE: int = 2


@dataclass
class BadRead():
    reason: str


@dataclass
class WhiteBars():
    white_bar_positions: list [int]
    white_bar_widths: list[int]


@dataclass
class FilteredWhiteBars(WhiteBars):
    pass


def filter_white_bars(whitebars: WhiteBars, length_array: int) ->WhiteBars:

    # filter for bars touching edges - this can be valid
    valid_indices = [
        i for i, white_bar_transitions in enumerate(whitebars.white_bar_positions)
        if 0 not in white_bar_transitions and length_array not in white_bar_transitions
        and 1 not in white_bar_transitions and length_array-1 not in white_bar_transitions
    ]

    # Filter both white_bar_positions and white_bar_widths using the valid indices
    whitebars.white_bar_positions = [
        whitebars.white_bar_positions[i] for i in valid_indices
    ]
    whitebars.white_bar_widths = [
        whitebars.white_bar_widths[i] for i in valid_indices
    ]

    # filter for bad widths - this should invalidate the barcode if its full of noise
    # NB - do this after first filter
    if whitebars.white_bar_widths:
        if abs(max(whitebars.white_bar_widths) - min(whitebars.white_bar_widths)) > min(2, length_array//10):
            return BadRead(reason=f"bad widths {whitebars.white_bar_widths}")

    return FilteredWhiteBars(**whitebars.__dict__)


def check_pattern_valid(filtered_bars: Union[list[FilteredWhiteBars], BadRead], testlength: int):
    """Checking for Player1: one sample with 2 peaks and one with none"""

    SYM_ERR_PXLS = 3  # how much symmetrical error can we allow
    for bar in filtered_bars:
        if isinstance(bar, BadRead):
            return False
        if len(bar.white_bar_positions) not in [0, 2]:
            return False
    
    # dictionary will overwrite keys, should be left with 2 keys
    if len({len(i.white_bar_positions): None for i in filtered_bars}) != 2:
        return False
    
    # check they are symmtericall around centre point - this is for specific pattern
    for filtered_bar in filtered_bars:
        if len(filtered_bar.white_bar_positions) == 0 : continue
        midpositions_bars = [sum(i)/len(i) for i in filtered_bar.white_bar_positions]
        offsets = [(testlength/2)-i for i in midpositions_bars]
        if not all([
            max(offsets) > 0,
            min(offsets) < 0,
            reduce(lambda a, b: abs(a - b), [abs(testlength/2-i) for i in midpositions_bars]) < SYM_ERR_PXLS
        ]):
            return False
    return True


def decode_white_bars(data) -> WhiteBars:
    """
    Detects white bars in a barcode by identifying transitions from black to white to black.

    Parameters:
    - data: np.ndarray, 1D array of uint8 values representing the scanned barcode.
        NOT NORMALIZED!!!

    Returns:
    - white_bar_positions: List of (start_index, end_index) tuples for each white bar.
    - white_bar_widths: List of widths of the white bars.
    - binary_data: Binarized version of the input data.
    """
    MIN_VARIANCE = 15  # if the barcode has small intensity range, probably just noise. Set all to zero
    # Normalize data
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min < MIN_VARIANCE:
        normalized_data = np.zeros_like(data, dtype=float)
    else:
        normalized_data = (data - data_min) / (data_max - data_min)

    # Thresholding (fixed threshold at 0.5)
    binary_data = (normalized_data > 0.5).astype(np.int8)

    # Find transitions
    diff_data = np.diff(binary_data)

    # Find indices where transitions occur
    white_starts = np.where(diff_data == 1)[0] + 1
    white_ends = np.where(diff_data == -1)[0] + 1

    # Handle edge cases
    if binary_data[0] == 1:
        # The signal starts with a white bar
        white_starts = np.insert(white_starts, 0, 0)
    if binary_data[-1] == 1:
        # The signal ends with a white bar
        white_ends = np.append(white_ends, len(binary_data))

    # Pair up starts and ends to get the positions of white bars
    white_bar_positions = list(zip(white_starts, white_ends))

    # Calculate widths of the white bars
    white_bar_widths = white_ends - white_starts

    return (WhiteBars(white_bar_positions, white_bar_widths))


def decode_barcode_bw_transitions(data):
    """
    Decodes a barcode by detecting transitions from black to white.

    Parameters:
    - data: np.ndarray, 1D array of uint8 values representing the scanned barcode.

    Returns:
    - transitions_bw: List of indices where black-to-white transitions occur.
    - widths: List of widths between black-to-white transitions.
    - binary_data: Binarized version of the input data.
    """
    MINIMUM_VARIANCE = 5 # simple way to check if we have black or whiteout (8-bit intensity)
    # but we don't want to filter out low contast barcodes..
    # TODO think of a better way to filter this out, like finding variation using normal distribution

    # Normalize data
    data_min = data.min()
    data_max = data.max()
    if (data_max - data_min) < MINIMUM_VARIANCE:
        normalized_data = np.zeros_like(data, dtype=float)
    else:
        normalized_data = (data - data_min) / (data_max - data_min)

    # Thresholding (fixed threshold at 0.5)
    binary_data = (normalized_data > 0.5).astype(np.int8)

    # Finding Transitions from Black to White (0 to 1)
    diff_data = np.diff(binary_data)
    bw_transition_indices = np.where(diff_data == 1)[0] + 1  # Add 1 due to diff shift
    transitions_bw = bw_transition_indices.tolist()

    # Include start position if the barcode starts with white
    if binary_data[0] == 1:
        positions = [0] + transitions_bw
    else:
        positions = transitions_bw

    # Include end position if the barcode ends with black
    if binary_data[-1] == 0:
        positions += [len(data)]

    # Calculate widths between black-to-white transitions
    widths = np.diff(positions)

    return transitions_bw, widths.tolist(), binary_data


def visualise_1d_barcode(_1dbarcode, height, segmentise:Optional[int]=None):
    width = 50
    out_img1 = cv2.resize(np.asarray(_1dbarcode), (50, height), interpolation=cv2.INTER_NEAREST)
    out_img1 = cv2.cvtColor(out_img1, cv2.COLOR_GRAY2BGR)
    if segmentise:
        for segment_y in [i for i in range(0,height, height//segmentise)]:
            cv2.line(out_img1, (0, segment_y), (5, segment_y), (0, 0, 255), 3)


    return out_img1


def analyse_and_get_image(test_member):
    height = 500
    scale = height/len(test_member)
    whitebars = filter_white_bars(
        decode_white_bars(np.array(test_member)),
        length_array=len(test_member)
        )
    out_img1 = cv2.resize(np.asarray(test_member), (200, height), interpolation=cv2.INTER_NEAREST)
    out_img1 = cv2.cvtColor(out_img1, cv2.COLOR_GRAY2BGR)
    if isinstance(whitebars, FilteredWhiteBars):
        for white_bar in whitebars.white_bar_positions:

            bw = min(white_bar[0] * scale, height-1)
            wb = min(white_bar[1] * scale, height-1)
            out_img1[int(bw), :, :] = (0,0,255)
            out_img1[int(wb), :, :] = (0,255,0)

    return whitebars, out_img1


def create_debug_imagepair(barcodepair: List):
    whitebars1, out_img1 = analyse_and_get_image(barcodepair[0])
    whitebars2, out_img2 = analyse_and_get_image(barcodepair[1])
    midimg = np.zeros(out_img1.shape, np.uint8)
    if check_pattern_valid([whitebars1, whitebars2], len(barcodepair[0])):
        midimg[:, :, 1] = 255
    else:
        midimg[:, :, 2] = 255 

    stacked_img = np.hstack((
        out_img1,
        midimg,
        out_img2))
    
    return stacked_img


def is_valid_quadro_id(spoke_samples_corners: list[int]) -> True:
    """quadrocode with diagonal orientation and orthogonal ID
    Check that the diagonal elements are correct
    &%%#%%@&&&&&%%%%###%#%###&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&%%##%%%%%###%%&&&&&&&&&&&&&&&&&&&&&&&&&&&&%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%, .. #%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%* . ../,. .%%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%#.... ,%&&&&%...*%%&&&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&%%%%&&%/. ..#&&&&&&&&&&(...%%&&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&%%%%%%#......(&&&&&&&&&&&&&&&%....%%&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&%&&&%%%%%. ..../%&&&&&&&&&&&&&&&&&&&&%...(%%&%%%%&&&&&&&&&&&&&&&&&
    &&&&&&&&&%&&%%%%%%%.... .,%&&&&&&&&&&&&&&&&&&&&&&&&&&*.. %%%%%%&%&&&&&&&&&&&&&&&
    &&&&&&&&&%%%%%%......,%&&&&&&&&#&&&&&&&&&&&&&&&&&&&&&&%....%%%%%&&&&&&&&&&&&&&&&
    &&&&&&&&%%%, ... .#%&&&&&&&&&&&&&&&&&&&&&&&&&&&&#.,%&&&%#...(%%%%&&&&&&&&&&&&&&&
    &&&&&&#...   .(%&&&&&&&%%&&&&&&&&&&&&&&&&&&&&&%......%&&&%,.. %%%%%&%%%&&&&&&&&&
    &&&&&&&,,   .%&&%&&&&#.....&&&&&&&&&#&&%&&&&&&&%. ...,&&&&&%...*%%%%%%%%%&&&&&&&
    &&&&&&&&%#    (&&&&&%,.   .&&&&&&&&%......./(&&&&%%%%&&&&&&&&/.. #%%%%%%&&&&&&&&
    &&&&&&&&%&%*.   #&&&&%/...&&&&&&&&%/..,..  .*&&&&&&&&&%%%&&&&&%.  .%%%%%&&%%&&&&
    &&&&&&&&&%%&%.   .%&&&&&&&&&&&&&&&&%,.. ... %&&&&&&&%,....(&&&&&#. .,%%%%%%&&&&&
    &&&%%&&&%%%%%%%.  ./&&&&&&#..  #%&&%%%#/,/%&&&&&&&&&% ..  *&&&&&&%, ..,%%%&&&&&&
    &&&&&%&&&&&&%%&%#.   %&&&%(. ...*%&&&&&&&&&&&&&&&&&&&%*..#&&&&&&&&#... (%%%&&&&&
    %&&&&&&&&&&&&&%%&%,    %%#%%..  #%&&&&&&&&&&&&&&&&&&&&&&&&&&&%#. ... .#%%%%%%%&&
    &&%&&&&&&&&&&&&%%&%# . .*%&&&&&&&&&%%%&&&&&&&&&&&&&&&&&&&%%...    .%%%%%%%%%%%%%
    &&%%&&&&&&&&%%%%%%%%%(  . #%&&&&&#  .*  (&&&&&&&&&&&&&%/.   . .%%%%%%%%%%%%%%%%%
    &&&&&&&&&&&&%%%%%%%%%%%.  ..%%%%%#      .%&&&&&&&&%/ .     #%%%%%%%%%%%%%%%%%%%%
    &&&&&&&%%&&&&&&%%%%%%%%%(    *%%%%%%%#%%%&&&&&%#.   .  *#%%%%%%%%%%%%%%%%%%%%%%%
    &&&&&&&&%%%%%%%&&%%%%%%%%%.  . #%%%%&&&&&%%#, .    ,#%%%%%%%%%%%%%%%%%%%%%%%%&&&
    &&&&&&&&%%%%%%%%%%%%%%%%%%%#  . .%%%&%%%*      .#%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&&
    &&&&&&&&&&%%%%%%%%%%%%%%%%%%%/    *%(...   .#%%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&&&&&
    &&&&&&&&&&%%%%%%%%%%%%%%%%%%%%%.        (%%%%%&%%&&&&&%%%%%%%%%%%%%&&&&&&&&&&&&&
    %&&&&&&&&&&%%%%%%%%%%%%%%%%%%%%%(   ,%%%%%&%&%%%%%%%&&&&&&%%%%%%%&&&&&&&&&&&&&&&
    %%&&&&&&&&&&%%%%%%%%%%%%%%%%%%%%%%%%%%%%%&&&&%%%%%%%%&&&&&&&&&&&&&&&&&&&&&&&&&&&
    """
    # get all white bars - with no filtering for bars touching edges
    # this will normalise the data!
    white_bars: WhiteBars = decode_white_bars(spoke_samples_corners)
    # 7 and 8 here depends on if sample overlaps/touches boundary or not
    # see ascii art - we check all diagonal in a clockwise sequence, starting
    # each time from the centre. When these segments are connected together the
    # centrepoint and potential outerboundry are continuous (one bar), except for
    # the last sample which does not hit the outer boundary (bars = 7) or does
    # bars = 8

    expectedpeaks_insideboundary = 7
    expectedpeaks_ousideboundary = 8
    if len(white_bars.white_bar_positions) not in [
        expectedpeaks_insideboundary,
        expectedpeaks_ousideboundary
    ]:
        return False
    # now check it is in the format we expect
    # the quadroID should have white 1 peak in each segment (not touching edges),
    # except for one segment which has no peak. This is how we orientate and validate the ID
    if len(white_bars.white_bar_positions) > 9:
        plop=1
    # Filter out any bars touching the edges of each sample (each sample is 1/4 of the array)
    # calculate the terminus of each sample
    assert (len(spoke_samples_corners) % 4) == 0, "should be multiple of 4!! QuadroCode alignment is 4 diagonal segments" # make sure 4 samples or something weird going on
    segment_ends = [
        i for i
        in range(
            0, len(spoke_samples_corners), len(spoke_samples_corners)//4
            )
        ] + [len(spoke_samples_corners)] # 4 segments will have 5 edges

    # check first that each segment starts with a high signal - as 
    # pattern has the centre dot and each segment spans from centre outwards
    bar_positions_copy = white_bars.white_bar_positions.copy()
    segment_end_cnt = 0
    for segment_index in segment_ends[:-1]: # last segment may not be white due to sample lines not overlapping boundary
        while bar_positions_copy:
            test_start_is_white = bar_positions_copy[0]
            # check if white bar straddles the segment end/start
            if all([
                (test_start_is_white[0] <= segment_index),
                (test_start_is_white[1] >= segment_index)
            ]):
                segment_end_cnt += 1
                break
            bar_positions_copy.pop(0)

    # if we have detected 4 white bars on the start of each segment (due to starting from white dot in middle)
    # we can assume is a candidate for valid ID
    if segment_end_cnt > 4:
        raise RuntimeError("impossible to happen, logic error")
    if segment_end_cnt != 4:
        return False
    

    quad = {}
    for bar_pos in white_bars.white_bar_positions:
        while bar_pos[0] > segment_ends[0]:
            segment_ends.pop(0)
        # does white bar straddle or touch an edge?
        if (bar_pos[0] <= segment_ends[0]) and (bar_pos[1] >= segment_ends[0]):
            pass
        else:
            quad_key = floor(segment_ends[0] / (len(spoke_samples_corners)/4))
            if quad_key not in quad:
                quad[quad_key] = []
            quad[quad_key].append(bar_pos)     


    # should be 3 non-edge white bars for this ID (see example of diagonal sampling)
    # nb - continuous sample is in 4 segments, each segment start is categorised as an edge
    if not all([(len(whitebar_pos)==1) for _, whitebar_pos in quad.items()]):
        return False
  
    # now we check that these 3 barcodes are in the quadrants
    if len(quad) != 3:
        return False

    return True


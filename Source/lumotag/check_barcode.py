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
from functools import lru_cache

MSE_LIM_BAR_WIDTH = 2
MSE_LIM_BAR_DISTANCE = 1.5

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


class VerifyBarcodeResult:
    __slots__ = ['res', 'sqr_err', 'status', "orientation_offset", "decoded_id", "retry_reduce_blur"]
    def __init__(
            self,
            res: bool,
            sqr_err: float = -1,
            status:str="no status",
            orientation_offset: int=-1,
            decoded_id: int=-1,
            retry_reduce_blur: bool=False):
        self.res = res
        self.sqr_err = sqr_err
        self.status = status
        self.orientation_offset = orientation_offset
        self.decoded_id = decoded_id
        self.retry_reduce_blur = retry_reduce_blur


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


def decode_white_bars(data) -> tuple[WhiteBars, np.ndarray]:
    """
    Detects white bars in a barcode by identifying transitions from black to white to black.
    Uses statistical methods to handle hotspots and improve normalization.

    Parameters:
    - data: np.ndarray, 1D array of uint8 values representing the scanned barcode.
        NOT NORMALIZED!!!

    Returns:
    - white_bar_positions: List of (start_index, end_index) tuples for each white bar.
    - white_bar_widths: List of widths of the white bars.
    - binary_data: Binarized version of the input data.
    """
    MIN_VARIANCE = 15
    
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    
    # Define outlier threshold (e.g., 2 standard deviations)
    OUTLIER_THRESHOLD = 2
    
    # Create mask for non-outlier values
    valid_mask = np.abs(data - mean) <= (OUTLIER_THRESHOLD * std)
    
    if np.sum(valid_mask) < len(data) * 0.5:  # If we've masked too much data
        # Fall back to percentile method
        data_min = np.percentile(data, 5)
        data_max = np.percentile(data, 95)
    else:
        # Use statistics from non-outlier values
        valid_data = data[valid_mask]
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
    
    if data_max - data_min < MIN_VARIANCE:
        normalized_data = np.zeros_like(data, dtype=float)
    else:
        # Clip values to range before normalizing
        data_clipped = np.clip(data, data_min, data_max)
        normalized_data = (data_clipped - data_min) / (data_max - data_min)

    # Rest of the function remains the same
    binary_data = (normalized_data > 0.3).astype(np.int8)
    diff_data = np.diff(binary_data)
    white_starts = np.where(diff_data == 1)[0] + 1
    white_ends = np.where(diff_data == -1)[0] + 1

    if binary_data[0] == 1:
        white_starts = np.insert(white_starts, 0, 0)
    if binary_data[-1] == 1:
        white_ends = np.append(white_ends, len(binary_data))

    white_bar_positions = list(zip(white_starts, white_ends))
    white_bar_widths = white_ends - white_starts

    return (WhiteBars(white_bar_positions, white_bar_widths)), binary_data


def decode_white_bars_robust(data) -> tuple[WhiteBars, any]:
    """
    Robust barcode white bar detection optimized for small arrays (~40 elements).
    Uses Otsu's automatic thresholding with statistical fallback.
    
    Parameters:
    - data: np.ndarray, 1D array of uint8 values representing the scanned barcode.
        NOT NORMALIZED!!!
    
    Returns:
    - white_bar_positions: List of (start_index, end_index) tuples for each white bar.
    - white_bar_widths: List of widths of the white bars.
    - binary_data: Binarized version of the input data.
    """
    MIN_VARIANCE = 15
    
    # Quick variance check - if too low, return empty result
    if np.std(data) < MIN_VARIANCE:
        binary_data = np.zeros_like(data, dtype=np.int8)
        return WhiteBars([], []), binary_data
    
    # Method 1: Otsu's automatic threshold (optimal for bimodal distributions)
    # Very fast for small arrays like 40 elements
    try:
        # Otsu works on uint8 data directly
        data_uint8 = data.astype(np.uint8)
        otsu_threshold, _ = cv2.threshold(data_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply the threshold to get binary data
        binary_data = (data > otsu_threshold).astype(np.int8)
        
        # Quick validation: check if we have reasonable transitions
        transitions = np.sum(np.abs(np.diff(binary_data)))
        if transitions >= 2:  # At least one complete bar (2 transitions)
            return _extract_white_bars_fast(binary_data)
    
    except (cv2.error, ValueError):
        pass  # Fall through to backup method
    
    # Method 2: Fast robust statistical approach (fallback)
    # Use inter-quartile range for robust thresholding
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    
    if iqr < MIN_VARIANCE:
        binary_data = np.zeros_like(data, dtype=np.int8)
        return WhiteBars([], []), binary_data
    
    # Robust outlier bounds using IQR
    outlier_factor = 1.5
    data_min = max(data.min(), q1 - outlier_factor * iqr)
    data_max = min(data.max(), q3 + outlier_factor * iqr)
    
    # Clip and normalize
    data_clipped = np.clip(data, data_min, data_max)
    
    if data_max - data_min < MIN_VARIANCE:
        binary_data = np.zeros_like(data, dtype=np.int8)
        return WhiteBars([], []), binary_data
    
    normalized_data = (data_clipped - data_min) / (data_max - data_min)
    
    # Use median as threshold (more robust than mean for bimodal data)
    threshold = np.median(normalized_data)
    binary_data = (normalized_data > threshold).astype(np.int8)
    
    return _extract_white_bars_fast(binary_data)


def _extract_white_bars_fast(binary_data) -> tuple[WhiteBars, any]:
    """
    Fast extraction of white bar positions from binary data.
    Optimized for small arrays.
    """
    # Find transitions using diff
    diff_data = np.diff(binary_data)
    white_starts = np.where(diff_data == 1)[0] + 1
    white_ends = np.where(diff_data == -1)[0] + 1
    
    # Handle edge cases efficiently
    if len(white_starts) == 0 and len(white_ends) == 0:
        # Check if entire array is white
        if binary_data[0] == 1:
            white_bar_positions = [(0, len(binary_data))]
            white_bar_widths = np.array([len(binary_data)])
        else:
            white_bar_positions = []
            white_bar_widths = np.array([])
    else:
        # Standard case with transitions
        if binary_data[0] == 1:
            white_starts = np.insert(white_starts, 0, 0)
        if binary_data[-1] == 1:
            white_ends = np.append(white_ends, len(binary_data))
        
        # Ensure we have matching start/end pairs
        min_len = min(len(white_starts), len(white_ends))
        white_starts = white_starts[:min_len]
        white_ends = white_ends[:min_len]
        
        white_bar_positions = list(zip(white_starts, white_ends))
        white_bar_widths = white_ends - white_starts
    
    return WhiteBars(white_bar_positions, white_bar_widths), binary_data


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
    out_img1 = cv2.resize(np.asarray(_1dbarcode), (50, height), interpolation=cv2.INTER_NEAREST)
    out_img1 = cv2.cvtColor(out_img1, cv2.COLOR_GRAY2BGR)
    if segmentise:
        for segment_y in [i for i in range(0,height, height//segmentise)]:
            cv2.line(out_img1, (0, segment_y), (5, segment_y), (0, 0, 255), 3)


    return out_img1

def visualise_color_barcode(color_array, height, segmentise:Optional[int]=None):
    """
    Visualize an array of RGB values as a tall, thin barcode.
    
    Args:
        color_array: Array of RGB values, shape=(n,3)
        height: Height of the output image
        segmentise: Optional number of segments to mark with lines
        
    Returns:
        Tall, thin color barcode image
    """
    # Convert input to numpy array
    color_array = np.asarray(color_array)
    
    # Get number of color values
    n_values = len(color_array)
    
    # Create a narrow image with one row per color value
    color_image = np.zeros((n_values, 1, 3), dtype=np.uint8)
    
    # Fill in the colors
    for i, color in enumerate(color_array):
        color_image[i, 0] = color
        
    # Resize to the target height, keeping it narrow (50 pixels wide)
    out_img = cv2.resize(color_image, (10, height), interpolation=cv2.INTER_NEAREST)
    
    # Add segment lines if requested
    if segmentise:
        for segment_y in [i for i in range(0, height, height//segmentise)]:
            cv2.line(out_img, (0, segment_y), (5, segment_y), (0, 0, 255), 3)

    return out_img


def analyse_and_get_image(test_member):
    height = 500
    scale = height/len(test_member)
    whitebars, _ = filter_white_bars(
        decode_white_bars_robust(np.array(test_member)),
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

@lru_cache(maxsize=None)
def get_10000_mask(segment_length) -> np.ndarray:
    """for the quadrocode ID
    
    input should be length of the diagonal samples stacked horizontally
    4 diagonals expected, therefore divison by 4 and 8 in this function

    we want the start of every segment (sample/4) to be white - which
    represents the centre dot

    ex white bars sample:
    10101|10100|10000|10100|
    ex mask
    10000|10000|10000|10000|

    """
    segment_mask = np.zeros(segment_length//4, dtype=bool)
    segment_mask[0] = True
    return np.tile(segment_mask, 4)


@lru_cache(maxsize=None)
def get_1100011_mask(segment_length, percent_from_edge = 10) -> np.ndarray:
    """for the quadrocode ID
    
    input should be length of the diagonal samples stacked horizontally
    4 diagonals expected, therefore divison by 4 and 8 in this function

    for the quadrocode - one middle bar should be missing - this is the orientation bar

    ex white middle bars on sample:
    00100|00100|00000|00100|
    ex mask
    10001|10001|10001|10001|

    OR them together 
    """
    segment_mask = np.zeros(segment_length//4, dtype=bool)
    # how far in is the illegal zone for each segment
    pc_buffer = max(1, int(len(segment_mask) * (percent_from_edge/100)))
    segment_mask[0: pc_buffer] = True
    segment_mask[-pc_buffer:] = True
    return np.tile(segment_mask, 4)


@lru_cache(maxsize=None)
def get_00100_mask(segment_length) -> np.ndarray:
    """for the quadrocode ID
    
    input should be length of the diagonal samples stacked horizontally
    4 diagonals expected, therefore divison by 4 and 8 in this function
    
    ex white edge bars on sample:
    11001|10001|11001|10000|
    ex mask
    00100|00100|00100|00100|
    """

    segment_mask = np.zeros(segment_length//4, dtype=bool)
    length = len(segment_mask)
    if length % 2 == 1:
        middle_index = length // 2
        segment_mask[middle_index] = True
    else:
        middle_left_index = (length // 2) - 1
        middle_right_index = middle_left_index + 1
        segment_mask[middle_left_index] = True
        segment_mask[middle_right_index] = True
    return np.tile(segment_mask, 4)

def get_ID(
        spoke_samples_corners: list[int],
        spoke_samples_middle_edges: list[int]
        ) -> VerifyBarcodeResult:
    res = is_valid_quadro_id(
        spoke_samples_corners=spoke_samples_corners
        )
    if res.res is True:
        res = decode_id(
            spoke_samples_middle_edges=spoke_samples_middle_edges,
            verify_is_barcode_res=res
            )
    return res


def decode_id(
        spoke_samples_middle_edges: list[int],
        verify_is_barcode_res: VerifyBarcodeResult
        ) -> VerifyBarcodeResult:
    """quadrocode with diagonal orientation and orthogonal ID
    Check that the diagonal elements are correct, and get orientation
    Use orientation to shift orthogonal elements and decode ID
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
    
    same as diagonal quadrants/segments - we expected a high signal for the 
    start of each segment
    
    nb- each diagonal segment has origin at centre dot and radiates around like spokes"""


    # shift samples according to incoming orientation
    offset = (len(spoke_samples_middle_edges)//4) * verify_is_barcode_res.orientation_offset
    spoke_samples_middle_edges = np.roll(spoke_samples_middle_edges, shift=offset)

    # get all white bars - with no filtering for bars touching edges
    # this will normalise the data!
    white_bars, binary_bars = decode_white_bars_robust(spoke_samples_middle_edges)

    # now test that the start of each edge bar is white (corresponding to centre dot)
    segment_high_start_mask = get_10000_mask(len(spoke_samples_middle_edges))
    res = np.bitwise_and(segment_high_start_mask, binary_bars.astype(bool))
    res = res[::len(spoke_samples_middle_edges)//4] # get every Nth sample - here the start of each segment
    if not np.all(res):
        return VerifyBarcodeResult(
            res=False,
            sqr_err=verify_is_barcode_res.sqr_err,
            status="f1: quadrant not starting with high signal")
    

    non_edge_bars_per_quad, edge_bars_per_quad, _100001_bars, _00100_bars = breakout_edges_and_middle_bars(spoke_samples_middle_edges, white_bars)
    # each quadrant sans edge bars should have at most 1 bar
    if not all([(len(whitebar_pos) <= 1) for _, whitebar_pos in non_edge_bars_per_quad.items()]):
        return VerifyBarcodeResult(
            res=False,
            sqr_err=verify_is_barcode_res.sqr_err,
            status="f2: segment bar count invalid"
            )

    # for this specific barcode - we expect to have one bar at two positions
    pos1 = 1
    pos2 = 3
    if not all((
        pos1 in non_edge_bars_per_quad, # short circuit if not found
        pos2 in non_edge_bars_per_quad, # short circuit if not found
        pos1 in non_edge_bars_per_quad and len(non_edge_bars_per_quad[1]) == pos1,
        pos2 in non_edge_bars_per_quad and len(non_edge_bars_per_quad[3]) == pos1
    )):
        return VerifyBarcodeResult(
            res=False,
            sqr_err=verify_is_barcode_res.sqr_err,
            status=f"f3: did not find bars at position {pos1} and {pos2} for quadrocode"
            )

    _00100_mask = get_00100_mask(len(spoke_samples_middle_edges))
    if np.any(np.bitwise_and(_00100_mask, _100001_bars)):
        # check our corner edges don't touch the middle of each segment
        # ex white edge bars on sample:
        # 11001|10001|11001|10000|
        # ex mask
        # 00100|00100|00100|00100|
        # now OR them together - if any of the edge white bars touch the middle - fail
        return VerifyBarcodeResult(
            res=False,
            sqr_err=verify_is_barcode_res.sqr_err,
            status="f4: segment edge bar too large"
            )
    

    # get integer representation of 2 ID bits
    # should be left at position 1 and 3
    # don't know what endian they are, who cares

    # this sucks I am sorry
    barcode_id_int = 0
    # these should always be on for now 
    for index, bar_pos in enumerate([1,2,3,4]):
        if non_edge_bars_per_quad.get(bar_pos) is None:
            if bar_pos in [1,3]:
                raise Exception("During ID debugging - positions 1 and 3 are always ON!!!")
            continue
        barcode_id_int += 2**index


    # for index, bar_pos in enumerate([1,3]):
    #     if non_edge_bars_per_quad.get(bar_pos) is None:
    #         continue
    #     barcode_id_int += 2**index


    return VerifyBarcodeResult(
        res=True,
        sqr_err=verify_is_barcode_res.sqr_err,
        status="p ID CHECK: pass",
        decoded_id=barcode_id_int
        )


def breakout_edges_and_middle_bars(samples, white_bars):
    # we extract bars not touching edges of segments, and which straddles
    # the middle of each segment as expected for this barcode
    # samples in form:
    # 000000000000
    # segments in form (corresponding to radial samples outwards of centre)
    # |000|000|000|000|
    # we want the position of every bar
    segment_ends = [
        i for i
        in range(
            0, len(samples), len(samples)//4
            )
        ] + [len(samples)] # 4 segments will have 5 edges


    non_edge_bars_per_quad = {}
    edge_bars_per_quad = {}
    _100001_bars = np.zeros(len(samples), dtype=bool)
    _00100_bars = np.zeros(len(samples), dtype=bool)

    for bar_pos in white_bars.white_bar_positions:
        while bar_pos[0] > segment_ends[0]:
            segment_ends.pop(0)

        quad_key = floor(segment_ends[0] / (len(samples)/4))

        # does white bar straddle or touch an edge?
        if (bar_pos[0] <= segment_ends[0]) and (bar_pos[1] >= segment_ends[0]):
            if quad_key not in edge_bars_per_quad:
                edge_bars_per_quad[quad_key] = []
            edge_bars_per_quad[quad_key].append(bar_pos)
            # also load in the binary representation
            _100001_bars[bar_pos[0]:bar_pos[1]] = True
        else:
            # also load in the binary representation
            _00100_bars[bar_pos[0]:bar_pos[1]] = True
            if quad_key not in non_edge_bars_per_quad:
                non_edge_bars_per_quad[quad_key] = []
            non_edge_bars_per_quad[quad_key].append(bar_pos) 
    return non_edge_bars_per_quad, edge_bars_per_quad, _100001_bars, _00100_bars

def is_valid_quadro_id(spoke_samples_corners: list[int]) -> VerifyBarcodeResult:
    """quadrocode with diagonal orientation and orthogonal ID
    Check that the diagonal elements are correct, and get orientation
    Use orientation to shift orthogonal elements and decode ID
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
    white_bars, binary_bars = decode_white_bars_robust(spoke_samples_corners)
    # 7 and 8 here depends on if sample overlaps/touches boundary or not
    # see ascii art - we check all diagonal in a clockwise sequence, starting
    # each time from the centre. When these segments are connected together the
    # centrepoint and potential outerboundry are continuous (one bar), except for
    # the last sample which does not hit the outer boundary (bars = 7) or does
    # (bars = 8)

    expectedpeaks_insideboundary = 7
    expectedpeaks_ousideboundary = 8

    if len(white_bars.white_bar_positions) > expectedpeaks_ousideboundary:
        # if the boundary is motion blurred/out of focus sometimes a sample can fall outside the barcode and return a LOW signal
        # this can split a barcode into two parts and fail. Its worth retrying in this case
        return VerifyBarcodeResult(
            res=False,
            status="bad peak count, worth retrying",
            retry_reduce_blur=True
            )
    if len(white_bars.white_bar_positions) not in [
        expectedpeaks_insideboundary,
        expectedpeaks_ousideboundary
    ]:
        return VerifyBarcodeResult(
            res=False,
            status=f"bad peak total {len(white_bars.white_bar_positions)}")
    # now check it is in the format we expect
    # the quadroID should have white 1 peak in each segment (not touching edges),
    # except for one segment which has no peak. This is how we orientate and validate the ID
    # Filter out any bars touching the edges of each sample (each sample is 1/4 of the array)
    # calculate the terminus of each sample
    if (len(spoke_samples_corners) % 4) != 0:
        raise ValueError(
            "should be multiple of 4!! QuadroCode alignment is 4 diagonal segments"
         ) # make sure 4 samples or something weird going on

    # # samples in form:
    # # 000000000000
    # # segments in form (corresponding to radial samples outwards of centre)
    # # |000|000|000|000|
    # # we want the position of every bar
    # segment_ends = [
    #     i for i
    #     in range(
    #         0, len(spoke_samples_corners), len(spoke_samples_corners)//4
    #         )
    #     ] + [len(spoke_samples_corners)] # 4 segments will have 5 edges


    # now test that the start of each edge bar is white (corresponding to centre dot)
    segment_high_start_mask = get_10000_mask(len(spoke_samples_corners))
    res = np.bitwise_and(segment_high_start_mask, binary_bars.astype(bool))
    res = res[::len(spoke_samples_corners)//4] # get every Nth sample - here the start of each segment
    if not np.all(res):
        return VerifyBarcodeResult(
            res=False,
            status="quadrant not starting with high signal")
    
    # # we extract bars not touching edges of segments, and which straddles
    # # the middle of each segment as expected for this barcode
    # non_edge_bars_per_quad = {}
    # edge_bars_per_quad = {}
    # _100001_bars = np.zeros(len(spoke_samples_corners), dtype=bool)
    # _00100_bars = np.zeros(len(spoke_samples_corners), dtype=bool)

    # for bar_pos in white_bars.white_bar_positions:
    #     while bar_pos[0] > segment_ends[0]:
    #         segment_ends.pop(0)

    #     quad_key = floor(segment_ends[0] / (len(spoke_samples_corners)/4))

    #     # does white bar straddle or touch an edge?
    #     if (bar_pos[0] <= segment_ends[0]) and (bar_pos[1] >= segment_ends[0]):
    #         if quad_key not in edge_bars_per_quad:
    #             edge_bars_per_quad[quad_key] = []
    #         edge_bars_per_quad[quad_key].append(bar_pos)
    #         # also load in the binary representation
    #         _100001_bars[bar_pos[0]:bar_pos[1]] = True
    #     else:
    #         # also load in the binary representation
    #         _00100_bars[bar_pos[0]:bar_pos[1]] = True
    #         if quad_key not in non_edge_bars_per_quad:
    #             non_edge_bars_per_quad[quad_key] = []
    #         non_edge_bars_per_quad[quad_key].append(bar_pos)  

    non_edge_bars_per_quad,edge_bars_per_quad, _100001_bars, _00100_bars = breakout_edges_and_middle_bars(spoke_samples_corners, white_bars)
    # should be 3 non-edge white bars for this ID (see example of diagonal sampling)
    # nb - continuous sample is in 4 segments, each segment start is categorised as an edge
    if not all([(len(whitebar_pos) == 1) for _, whitebar_pos in non_edge_bars_per_quad.items()]):
        return VerifyBarcodeResult(
            res=False,
            status="segment bar count invalid"
            )
  
    # now we check that these 3 barcodes are in the quadrants
    if len(non_edge_bars_per_quad) != 3:
        return VerifyBarcodeResult(
            res=False,
            status="bad internal white bar count for all quadrant"
            )


    # now check middle bars aren't near the edges of each segment
    _1100011_mask = get_1100011_mask(len(spoke_samples_corners))
    if np.any(np.bitwise_and(_1100011_mask, _00100_bars)):
        # ex white middle bars on sample:
        # 00100|00100|00100|00100|
        # ex mask
        # 10001|10001|10001|10001|
       return VerifyBarcodeResult(
            res=False,
            status="segment internal bar touching edge of segment"
            )

    _00100_mask = get_00100_mask(len(spoke_samples_corners))
    if np.any(np.bitwise_and(_00100_mask, _100001_bars)):
        # check our corner edges don't touch the middle of each segment
        # ex white edge bars on sample:
        # 11001|10001|11001|10000|
        # ex mask
        # 00100|00100|00100|00100|
        # now OR them together - if any of the edge white bars touch the middle - fail
        return VerifyBarcodeResult(
            res=False,
            status="segment edge bar too large"
            )


    ###### ID element sometimes not straddling middle.... keep this incase we need it
    # # now check middle bars straddle the middle
    # _00100_mask = get_00100_mask(len(spoke_samples_corners))
    # res = np.bitwise_and(_00100_mask, _00100_bars)
    # if not np.all(_00100_mask == res):
    #     # ex white middle bars on sample:
    #     # 00100|00100|00100|00100|
    #     # ex mask
    #     # 00100|00100|00100|00100|
    #     return False
    

    # check the distance of each 0001000 bar from the centre point, make sure within error
    # try mean squared error?
    # get avergae position of bars first
    sqr_errors_dist = 0
    sqr_errors_width = 0
    avg_distance_from_seg_start = 0
    avg_width = 0
    for quad, edgebars in non_edge_bars_per_quad.items():
        start_sample_pos = (len(spoke_samples_corners)//4) * (quad-1)
        avg_distance_from_seg_start += ((edgebars[0][0] + edgebars[0][1])/2) - start_sample_pos

        avg_width += (edgebars[0][1] - edgebars[0][0])

    avg_width = round(avg_width/3, 3)
    avg_distance_from_seg_start = round(avg_distance_from_seg_start/3, 3)# Magic number 3 for only 3 mid bars (out of 4) for orientation
    
    # now calculate MSE from average mid position of bars
    for quad, edgebars in non_edge_bars_per_quad.items():
        start_sample_pos = (len(spoke_samples_corners)//4) * (quad-1)
        Yi_expected_mid_pos = avg_distance_from_seg_start
        Yihat_mid_pos = ((edgebars[0][0] + edgebars[0][1])/2) - start_sample_pos# edgebars[0] as should always be one position only
        sqr_errors_dist += (Yi_expected_mid_pos - Yihat_mid_pos)**2

        Yi_expected_width = avg_width
        Yihat_width = (edgebars[0][1] - edgebars[0][0])
        sqr_errors_width += (Yi_expected_width - Yihat_width)**2

    # calculate MSQRERROR
    sqr_err_bardist = round((1/len(non_edge_bars_per_quad)) * sqr_errors_dist, 3) # watch out here - relies on non_edge_bars being cleaned up previously
    sqr_err_barwidth = round((1/len(non_edge_bars_per_quad)) * sqr_errors_width, 3) # watch out here - relies on non_edge_bars being cleaned up previously


    if sqr_err_bardist > MSE_LIM_BAR_WIDTH:
        return VerifyBarcodeResult(res=False, sqr_err=f"MSE dist {sqr_err_bardist} width {sqr_err_barwidth}", status="bad sqr_err_bardist")
    if sqr_err_barwidth > MSE_LIM_BAR_DISTANCE:
        return VerifyBarcodeResult(res=False, sqr_err=f"MSE dist {sqr_err_bardist} width {sqr_err_barwidth}", status="bad sqr_err_barwidth")
    

    # get orientation
    # orientation is determined by the missing bar from a quadrant/segment
    # in the non_edge_bars_per_quad dict, we should have keys [1-4] with the bar positions
    # missing integer in that range is the offset
    expected_sum = 10 # sum of [1,2,3,4]
    actual_sum = sum(list(non_edge_bars_per_quad.keys()))
    missing_number = expected_sum - actual_sum

    return VerifyBarcodeResult(
        res=True,
        sqr_err=f"MSE dist {sqr_err_bardist} width {sqr_err_barwidth}",
        status="Pass",
        orientation_offset=missing_number)

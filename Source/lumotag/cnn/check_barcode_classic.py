import numpy as np
import random
# import os
# import pickle
# import cv2
# import random
# from dataclasses import dataclass
# from typing import Union
# from functools import reduce
# @dataclass
# class BadRead():
#     reason:str

# @dataclass
# class WhiteBars():
#     white_bar_positions: list [int]
#     white_bar_widths: list[int]


# @dataclass
# class FilteredWhiteBars(WhiteBars):
#     pass


# def filter_white_bars(whitebars: WhiteBars, length_array: int) ->WhiteBars:


#     # filter for bars touching edges - this can be valid
#     valid_indices = [
#         i for i, white_bar_transitions in enumerate(whitebars.white_bar_positions)
#         if 0 not in white_bar_transitions and length_array not in white_bar_transitions
#         and 1 not in white_bar_transitions and length_array-1 not in white_bar_transitions
#     ]

    

#     # Filter both white_bar_positions and white_bar_widths using the valid indices
#     whitebars.white_bar_positions = [
#         whitebars.white_bar_positions[i] for i in valid_indices
#     ]
#     whitebars.white_bar_widths = [
#         whitebars.white_bar_widths[i] for i in valid_indices
#     ]


#     # filter for bad widths - this should invalidate the barcode if its full of noise
#     # NB - do this after first filter
#     if whitebars.white_bar_widths:
#         if abs(max(whitebars.white_bar_widths) - min(whitebars.white_bar_widths)) > min(2, length_array//10):
#             return BadRead(reason=f"bad widths {whitebars.white_bar_widths}")



#     return FilteredWhiteBars(**whitebars.__dict__)


# def check_pattern_valid(filtered_bars: Union[list[FilteredWhiteBars], BadRead], testlength: int):
#     """This is checking for one sample with 2 peaks and one with none"""

#     SYM_ERR_PXLS = 4 # how much symmetrical error can we allow
#     for bar in filtered_bars:
#         if isinstance(bar, BadRead):
#             return False
#         if len(bar.white_bar_positions) not in [0, 2]:
#             return False
    
#     # dictionary will overwrite keys, should be left with 2 keys
#     if len({len(i.white_bar_positions): None for i in filtered_bars}) != 2:
#         return False
    
#     # check they are symmtericall around centre point - this is for specific pattern
#     for filtered_bar in filtered_bars:
#         if len(filtered_bar.white_bar_positions) == 0 : continue
#         midpositions_bars = [sum(i)/len(i) for i in filtered_bar.white_bar_positions]
#         offsets = [(testlength/2)-i for i in midpositions_bars]
#         if not all([
#             max(offsets) > 0,
#             min(offsets) < 0,
#             reduce(lambda a, b: abs(a - b), [abs(testlength/2-i) for i in midpositions_bars]) < SYM_ERR_PXLS
#         ]):
#             return False
#     return True

# def decode_white_bars(data) -> WhiteBars:
#     """
#     Detects white bars in a barcode by identifying transitions from black to white to black.

#     Parameters:
#     - data: np.ndarray, 1D array of uint8 values representing the scanned barcode.

#     Returns:
#     - white_bar_positions: List of (start_index, end_index) tuples for each white bar.
#     - white_bar_widths: List of widths of the white bars.
#     - binary_data: Binarized version of the input data.
#     """
#     MIN_VARIANCE = 5 # if the barcode has small intensity range, probably just noise. Set all to zero
#     # Normalize data
#     data_min = data.min()
#     data_max = data.max()
#     if data_max - data_min < MIN_VARIANCE:
#         normalized_data = np.zeros_like(data, dtype=float)
#     else:
#         normalized_data = (data - data_min) / (data_max - data_min)

#     # Thresholding (fixed threshold at 0.5)
#     binary_data = (normalized_data > 0.5).astype(np.int8)

#     # Find transitions
#     diff_data = np.diff(binary_data)

#     # Find indices where transitions occur
#     white_starts = np.where(diff_data == 1)[0] + 1
#     white_ends = np.where(diff_data == -1)[0] + 1

#     # Handle edge cases
#     if binary_data[0] == 1:
#         # The signal starts with a white bar
#         white_starts = np.insert(white_starts, 0, 0)
#     if binary_data[-1] == 1:
#         # The signal ends with a white bar
#         white_ends = np.append(white_ends, len(binary_data))

#     # Pair up starts and ends to get the positions of white bars
#     white_bar_positions = list(zip(white_starts, white_ends))

#     # Calculate widths of the white bars
#     white_bar_widths = white_ends - white_starts

#     return (WhiteBars(white_bar_positions, white_bar_widths))



# def decode_barcode_bw_transitions(data):
#     """
#     Decodes a barcode by detecting transitions from black to white.

#     Parameters:
#     - data: np.ndarray, 1D array of uint8 values representing the scanned barcode.

#     Returns:
#     - transitions_bw: List of indices where black-to-white transitions occur.
#     - widths: List of widths between black-to-white transitions.
#     - binary_data: Binarized version of the input data.
#     """
#     # Normalize data
#     data_min = data.min()
#     data_max = data.max()
#     if data_max == data_min:
#         normalized_data = np.zeros_like(data, dtype=float)
#     else:
#         normalized_data = (data - data_min) / (data_max - data_min)

#     # Thresholding (fixed threshold at 0.5)
#     binary_data = (normalized_data > 0.5).astype(np.int8)

#     # Finding Transitions from Black to White (0 to 1)
#     diff_data = np.diff(binary_data)
#     bw_transition_indices = np.where(diff_data == 1)[0] + 1  # Add 1 due to diff shift
#     transitions_bw = bw_transition_indices.tolist()

#     # Include start position if the barcode starts with white
#     if binary_data[0] == 1:
#         positions = [0] + transitions_bw
#     else:
#         positions = transitions_bw

#     # Include end position if the barcode ends with black
#     if binary_data[-1] == 0:
#         positions += [len(data)]

#     # Calculate widths between black-to-white transitions
#     widths = np.diff(positions)

#     return transitions_bw, widths.tolist(), binary_data


def files_in_folder(directory, imgtypes: list[str]):
    allFiles = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name[-3:len(name)] in imgtypes:
                allFiles.append(os.path.join(root, name))
    return allFiles

import os
import sys
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
sys.path.append(parent_folder)


import check_barcode
import pickle
import cv2
import random
# load in pickle files and extract data:
folder = os.path.dirname(os.path.abspath(__file__))

pickle_files = files_in_folder(folder, [".pc"])
result_pairs = []
result_data = []
for picklefile in pickle_files:
    with open(picklefile, 'rb') as file:
        result_data.append((pickle.load(file), picklefile))


#result_pairs = [i[0] for i in result_data if "false" in i[1]]
#result_pairs = result_data[0][0] #+ result_data[1][0]
result_pairs = []
for sublist in result_data:
    if "false" in sublist[1]:
        result_pairs.extend(sublist[0])

random.shuffle(result_pairs)

def analyse_and_get_image(test_member):
    height = 500
    scale= height/len(test_member)
    whitebars: check_barcode.filteredWhiteBars = check_barcode.filter_white_bars(
        check_barcode.decode_white_bars(np.array(test_member)),
        length_array=len(test_member)
        )
    out_img1 = cv2.resize(np.asarray(test_member), (200, height), interpolation=cv2.INTER_NEAREST)
    out_img1 = cv2.cvtColor(out_img1, cv2.COLOR_GRAY2BGR)
    if isinstance(whitebars, check_barcode.FilteredWhiteBars):
        for white_bar in whitebars.white_bar_positions:

            bw = min(white_bar[0] * scale, height-1)
            wb = min(white_bar[1] * scale, height-1)
            out_img1[int(bw), :, :] = (0,0,255)
            out_img1[int(wb), :, :] = (0,255,0)

    return whitebars, out_img1


good = 0
bad = 0
for test_pair in result_pairs:
    stacked_img = check_barcode.create_debug_imagepair(test_pair)
    # #np_test_pair = np.array(np.concatenate((test_pair[0], test_pair[1]))).astype("uint8")
    # whitebars1, out_img1 = analyse_and_get_image(test_pair[0])
    # whitebars2, out_img2 = analyse_and_get_image(test_pair[1])
    # print(whitebars1, whitebars2)
    
    # # height = 500
    # # scale= height/len(test_pair[0])
    # # whitebars: FilteredWhiteBars = filter_white_bars(
    # #     decode_white_bars(np.array(test_pair[0])),
    # #     length_array=len(test_pair[0])
    # #     )
    # # out_img1 = cv2.resize(np.asarray(test_pair[0]), (200, height), interpolation=cv2.INTER_NEAREST)
    # # out_img1 = cv2.cvtColor(out_img1, cv2.COLOR_GRAY2BGR)
    # # for white_bar in whitebars.white_bar_positions:

    # #     bw = min(white_bar[0] * scale, height-1)
    # #     wb = min(white_bar[1] * scale, height-1)
    # #     out_img1[int(bw), :, :] = (0,0,255)
    # #     out_img1[int(wb), :, :] = (0,255,0)

    # # out_img2 = cv2.resize(np.asarray(test_pair[1]), (200, height), interpolation=cv2.INTER_NEAREST)
    # # out_img2 = cv2.cvtColor(out_img2, cv2.COLOR_GRAY2BGR)

    whitebars1, _ = analyse_and_get_image(test_pair[0])
    whitebars2, _ = analyse_and_get_image(test_pair[1])
    # midimg = np.zeros(out_img1.shape, np.uint8)
    if check_barcode.check_pattern_valid([whitebars1, whitebars2], len(test_pair[0])):
        good += 1
    else:
        bad += 1

    # stacked_img = np.hstack((
    #     out_img1,
    #     midimg,
    #     out_img2))
    cv2.imshow('graycsale image',stacked_img)
    

    if  check_barcode.check_pattern_valid([whitebars1, whitebars2], len(test_pair[0])):
        key=cv2.waitKey(0)
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break
    # if random.randint(0,200) == 1:
    #     cv2.waitKey(1)
total = good + bad
print((good/total) * 100)
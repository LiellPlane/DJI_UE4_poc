import numpy as np
import random


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
        check_barcode.decode_white_bars(np.array(test_member))[0],
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
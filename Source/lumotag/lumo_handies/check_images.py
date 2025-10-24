import os
import cv2
import numpy as np

def images_in_folder(directory, imgtypes: list[str]):
    allFiles = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name[-4:len(name)] in imgtypes:
                allFiles.append(os.path.join(root, name))
    return allFiles
imgfoler = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
parent_dir = os.path.dirname(imgfoler)
images = images_in_folder(parent_dir, [".jpg"])

images = [i for i in images if "unique" in i]
trigger_dict_close = {}
trigger_dict_long = {}
import re

pattern = r"cnt.*cnt"
for trigger in images:
    matches = re.findall(pattern, trigger)
    if matches and "close" in trigger:
        trigger_dict_close[matches[0]] = trigger
    else:
        trigger_dict_long[matches[0]] = trigger

assert list(trigger_dict_close.keys()) == list(trigger_dict_long.keys())

for peepee in list(trigger_dict_close.keys()):
    closefile = trigger_dict_close[peepee]
    longfile = trigger_dict_long[peepee]

    image1 = cv2.imread(closefile)
    image2 = cv2.imread(longfile)
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    if height1 != height2:
        # Resize image2 to the same height as image1
        image2 = cv2.resize(image2, (width2, height1))

    # Stack images horizontally
    stacked_image = np.hstack((image1, image2))
    img = cv2.resize(stacked_image, (1800,999))
    # Display the result
    cv2.imshow('Stacked Image', img)
    cv2.waitKey(0)
import numpy as np
import os
import pickle
import cv2
import random

TAG = "false"
player1_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "player1_pickles")
false_positive_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "false_positive_pickles")

def files_in_folder(directory, imgtypes: list[str]):
    allFiles = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name[-3:len(name)] in imgtypes:
                allFiles.append(os.path.join(root, name))
    return allFiles

# load in pickle files and extract data:
folder = os.path.dirname(os.path.abspath(__file__))

pickle_files = files_in_folder(folder, [".pc"])
result_pairs = []
for picklefile in pickle_files:
    if TAG not in picklefile:
        continue
    with open(picklefile, 'rb') as file:
        result_data= pickle.load(file)
    for pair in result_data:
        assert len(pair) == 2
        result_pairs.append(pair)

# visual sample a few to make sure OK:
while True:
    test_pair = random.choice(result_pairs)
    out_img1 = cv2.resize(np.asarray(test_pair[0]), (200, 500), interpolation=cv2.INTER_NEAREST)
    out_img1 = cv2.cvtColor(out_img1, cv2.COLOR_GRAY2BGR)
    out_img2 = cv2.resize(np.asarray(test_pair[1]), (200, 500), interpolation=cv2.INTER_NEAREST)
    out_img2 = cv2.cvtColor(out_img2, cv2.COLOR_GRAY2BGR)

    stacked_img = np.hstack((
        out_img1,
        np.zeros(out_img1.shape, np.uint8),
        out_img2))
    cv2.imshow('graycsale image',stacked_img)
    
    key=cv2.waitKey(0)
    if key == 27:#if ESC is pressed, exit loop
        cv2.destroyAllWindows()
        break

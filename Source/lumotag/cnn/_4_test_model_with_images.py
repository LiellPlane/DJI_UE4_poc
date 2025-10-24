import numpy as np
import tensorflow as tf
import os
import pickle
import cv2
import random

modelpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "barcode_model.h5")
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
result_data = []
for picklefile in pickle_files:
    with open(picklefile, 'rb') as file:
        result_data.append(pickle.load(file))


for result in result_data:
    for pair in result[:min([len(i) for i in result_data])]:
        assert len(pair) == 2
        result_pairs.append(pair)

random.shuffle(result_pairs)


loaded_model = tf.keras.models.load_model(modelpath)

# example of valid datatype from training
# array([[198, 136, 143, 154,  79,  17,  44,   7,   0, 226,  16, 216, 186,
#         212, 123,  15, 149, 100,  34, 160,   4, 155, 208, 229, 201, 211,
#         201, 177, 104, 191,  35, 169, 239, 168,  84, 110,  57, 143, 153,
#         176,  15,  58, 156,  38,  19,  83, 241,  41,  76,  45]],
#       dtype=uint8)
# X_noise_eval[0].reshape(1,len(X_noise_eval[0])).shape
# (1, 50)

while True:
    test_pair = random.choice(result_pairs)
    np_test_pair = np.array(np.concatenate((test_pair[0], test_pair[1]))).astype("uint8")
    score = loaded_model.predict(np_test_pair.reshape(1,50))
    print(score)
    out_img1 = cv2.resize(np.asarray(test_pair[0]), (200, 500), interpolation=cv2.INTER_NEAREST)
    out_img1 = cv2.cvtColor(out_img1, cv2.COLOR_GRAY2BGR)
    out_img2 = cv2.resize(np.asarray(test_pair[1]), (200, 500), interpolation=cv2.INTER_NEAREST)
    out_img2 = cv2.cvtColor(out_img2, cv2.COLOR_GRAY2BGR)
 

    midimg = np.zeros(out_img1.shape, np.uint8)
    # if score > 0.95:
    #     midimg[:,:,0] = 255
    # else:
    #     midimg[:,:,1] = 255 

    stacked_img = np.hstack((
        out_img1,
        midimg,
        out_img2))
    cv2.imshow('graycsale image',stacked_img)
    


    key=cv2.waitKey(0)
    if key == 27:#if ESC is pressed, exit loop
        cv2.destroyAllWindows()
        break

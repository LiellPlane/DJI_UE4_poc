import numpy as np
import os
import pickle
import datetime
import os
modelpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "barcode_model.h5")
pickle_pairs_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)))
# Disable GPU

def files_in_folder(directory, imgtypes: list[str]):
    allFiles = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name[-3:len(name)] in imgtypes:
                allFiles.append(os.path.join(root, name))
    return allFiles

# load in pickle files and extract data:
folder = r"D:\lumotag_training_data\more_false_positives"
tag = "false"
pickle_files = files_in_folder(folder, ["pck"])
result_pairs = []
for picklefile in pickle_files:
    if tag not in picklefile:
        continue
    with open(picklefile, 'rb') as file:
        result_data= pickle.load(file)
    for pair in result_data:
        assert len(pair) == 2
        result_pairs.append(pair)
if len(result_pairs) < 1:
    raise Exception
ts = datetime.datetime.now().strftime("%Y%M%d%H%M%S%f")[:-3]
_path = os.path.join(pickle_pairs_path, f"tag{tag}tag{ts}.pc")
with open(_path, 'wb') as file:
    pickle.dump(result_pairs, file)
with open(_path, 'rb') as file:
    check_data= pickle.load(file)

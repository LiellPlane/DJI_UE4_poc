import os

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

list(trigger_dict_close.keys()) == list(trigger_dict_long.keys())



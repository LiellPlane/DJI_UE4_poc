import rabbit_mq
import factory
import messaging
import time
import msgs
import math
from PIL import Image
import cv2
import colorsys
import numpy as np
import json
import os
from os import listdir
#cd ~/
#sudo rm -r DJI_UE4_poc
#git clone https://github.com/LiellPlane/DJI_UE4_poc.git
#cd DJI_UE4_poc/Source/lumotag



coco_labels=[
"unlabeled",
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"street sign",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"hat",
"backpack",
"umbrella",
"shoe",
"eye glasses",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"plate",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"couch",
"potted plant",
"bed",
"mirror",
"dining table",
"window",
"desk",
"toilet",
"door",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"blender",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"]


def viewer(
        inputimage,
        pausetime_Secs=0,
        presskey=False,
        destroyWindow=True):
    try:
        cv2.imshow("img", inputimage.copy())
        if presskey==True:
            cv2.waitKey(0); #any key
        if presskey==False:
            if cv2.waitKey(20) & 0xFF == 27:
                    pass
        if pausetime_Secs>0:
            time.sleep(pausetime_Secs)
        if destroyWindow==True: cv2.destroyAllWindows()
    except Exception as e:
        print(e)

def plasma (w, h):
    """stolen plasma image generator"""
    out = Image.new("RGB", (w, h))
    pix = out.load()
    for x in range (w):
        for y in range(h):
            hue = 4.0 + math.sin(x / 19.0) + math.sin(y / 9.0) \
                + math.sin((x + y) / 25.0) + math.sin(math.sqrt(x**2.0 + y**2.0) / 8.0)
            hsv = colorsys.hsv_to_rgb(hue/8.0, 1, 1)
            pix[x, y] = tuple([int(round(c * 255.0)) for c in hsv])
    return out


def main():
    mssger = rabbit_mq.messenger(
        factory.TZAR_config())

    cnt = 0
    img = np.asarray(plasma(1000, 1000), dtype="uint8")
    
    while True:
        cnt += 1
        time.sleep(0.1)
        print("-> checking in box")
        message = mssger.check_in_box()
        if message is not None:
            img_as_str = msgs.bytes_to_str(message)
            img = msgs.decode_image_from_str(img_as_str)
            viewer(img,0,False,False)
        print("-> end of check")

        print("sending msg")
        img_str = msgs.encode_img_to_str(img)
        str_as_bin = msgs.str_to_bytes(img_str)
        mssger.send_message(str_as_bin)

def main2():
    mssger = rabbit_mq.messenger(
        factory.TZAR_config())

    cnt = 0
    #purge in box
    while True:
        result = mssger.check_in_box(blocking=False)
        if result is None:
            break
        time.sleep(0.1)

    #img = np.asarray(plasma(1000, 1000), dtype="uint8")
    folder_dir = r"D:\captured_match"
    
    for images in os.listdir(folder_dir):
        if not (images.endswith(".png")):
            continue
        img = cv2.imread(folder_dir + "\\" + images)
        img_str = msgs.encode_img_to_str(img)
        str_as_bin = msgs.str_to_bytes(img_str)
        mssger.send_message(str_as_bin)

        while True:
            time.sleep(0.1)
            result = mssger.check_in_box(blocking=True)
            if result is None:
                continue
            result_str = msgs.bytes_to_str(result)

            #terrible code
            if "ANALYSED" in result_str:
                result_dict = json.loads(result_str)
                result_dict["ClassID"] = coco_labels[int(result_dict["ClassID"])]
                print(result_dict)
                break
if __name__ == '__main__':
    main2()

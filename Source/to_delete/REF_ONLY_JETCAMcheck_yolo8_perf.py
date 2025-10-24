import json
import os
import ast
import cv2
import time
import numpy as np
import json
from dataclasses import dataclass, asdict

import factory 
import rabbit_mq
import msgs
import messaging
from time import perf_counter
import copy

def main():
    from ultralytics import YOLO
    mssger = rabbit_mq.MessengerBasic(
    factory.TZAR_config())
    yolo_path = r"C:\Working\ML\yolo\from-rav\trafford_hamilton_best_s.pt"
    model = YOLO(yolo_path)
    imgcnt = 0
    while True:
        imgcnt += 1
        print("waiting for image")
        message = mssger.check_in_box(blocking=True)
        print(f"checking{imgcnt}")
        if message is not None:
            try:
                print(message[0][0:100])
                img_as_str = msgs.bytes_to_str(message[0])
            except Exception as e:
                print(e)
                continue
            # sorry about this
            if "ANALYSED" in img_as_str:
                print("skipping")
                continue
            if "my_id" in img_as_str:
                print("skipping")
                continue
            print("probably an image")
            img = msgs.decode_image_from_str(img_as_str)

            t1_start = perf_counter()
            results = model.predict(
                img, # can be a cv image, array, path 
                imgsz=max(img.shape), # change based on your longest edge, choose 32*n
                classes=[0, 32], # person and sports ball
                save=False, # not essential
                conf=0.1, # ignore lower that this, deafult 0.25
                device="cpu", # cpu, or 0,1,2,3 for cuda
                save_txt=False, # not essential
                name="hamilton_trafford_predict_test", # can be whatever you want
                )
            t1_stop = perf_counter()

            all_dects = {}
            all_dects["SYSTEMINFO"] = "ANALYSED: YOLO8"
            dectdeets = None
            index = 0
            for res in results:
                dectdeets = {}
                boxes = res.boxes
                for cls, conf, xywh in zip(boxes.cls, boxes.conf, boxes.xywh):
                    dectdeets["filename"] = "ANALYSED"
                    dectdeets["img_count"] = imgcnt
                    dectdeets["ClassID"] =  int(cls.numpy())
                    dectdeets["Left"] = int(xywh.numpy()[0] - (xywh.numpy()[2]/2))
                    dectdeets["Top"] = int(xywh.numpy()[1] - (xywh.numpy()[3]/2))
                    dectdeets["Right"] = int(xywh.numpy()[0] + (xywh.numpy()[2]/2))
                    dectdeets["Bottom"] = int(xywh.numpy()[1] + (xywh.numpy()[3]/2))
                    dectdeets["Confidence"] = float(conf.numpy())
                    center = [int(xywh.numpy()[0]),int(xywh.numpy()[1])]
                    dectdeets["Center"] = center
                    dectdeets["index"] = str(index)
                    dectdeets["inference_time_secs"] = str(t1_stop-t1_start)
                    all_dects[index]=copy.deepcopy(dectdeets)
                    index+=1
            output = json.dumps(all_dects)
            output_bytes = msgs.str_to_bytes(output)
            mssger.send_message(output_bytes)
        else:
            time.sleep(0.2)

if __name__ == '__main__':
    main()
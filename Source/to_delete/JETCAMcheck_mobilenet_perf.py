import sys
try:
    import cv2
except Exception as e:
    print("error importing cv2 - attempting again with path")
    sys.path.append('/usr/local/lib/python3.8/site-packages')
    import cv2
    print("successfully imported cv2")
import time
import math
import vpi
import subprocess
import random
import time
import numpy as np
import cv2
from datetime import datetime
from contextlib import contextmanager
import math
import colorsys
from PIL import Image
from jetson_inference import imageNet, detectNet
import jetson_utils
import argparse
import torch
import threading
from PIL import Image
import glob
from dataclasses import asdict
import json
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
import copy
import JETCAM_support
from time import perf_counter

@contextmanager
def time_it(comment):
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        toc: float = time.perf_counter()
        print(f"{comment}:proc time = {1000*(toc - tic):.3f}ms")


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

def gstreamer_pipeline_out_mod():
    # leaky downstream throws away old images - default queue is 5
    # sync = false might be useful 
    # not tested with real cameras

    #MUX playback ID https://stream.mux.com/vL9SJU61FSv8sSQR01F6ajKI702WeK2pXRuLVtw25zquo.m3u8

    return (
        "appsrc ! "
        "videoconvert ! "
        "video/x-raw, framerate=(fraction)25/1, format=RGBA ! "
        "nvvidconv ! "
        "nvv4l2h264enc ! "
        "h264parse ! "
        "flvmux ! "
        "queue leaky=downstream ! "
        "rtmpsink location=rtmp://global-live.mux.com:5222/app/51bc0427-ad29-2909-4979-11ee335d2b53 sync=false"
    )

def inference_imagenet():
    net = imageNet("googlenet")

    img = np.asarray(plasma(1000, 1000), dtype="uint8")

    while True:
        print("plop")
        with time_it(" infer upload image"):
            cuda_mem = jetson_utils.cudaFromNumpy(img)

        with time_it(" infer classify image"):
            # classify the image
            class_idx, confidence = net.Classify(cuda_mem)

            # find the object description
            class_desc = net.GetClassDesc(class_idx)

        # print out the result
        print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))

def for_rav():
        import jetson_utils
        net = detectNet(model="/home/jetcam/mb1025_voc1501/ssd-mobilenet.onnx",
                    input_blob="input_0",
                    output_cvg="scores",
                    output_bbox="boxes", 
                    threshold=0.1)
        # if from numpy - different if from VPI
        cuda_mem = jetson_utils.cudaFromNumpy(img)
        detections = net.Detect(cuda_mem)
        dectdeets={}
        for dect in enumerate(detections):
                            dectdeets["ClassID"] = dect.ClassID
                            dectdeets["Left"] = dect.Left
                            dectdeets["Top"] = dect.Top
                            dectdeets["Right"] = dect.Right
                            dectdeets["Bottom"] = dect.Bottom
                            dectdeets["Confidence"] = dect.Confidence
                            dectdeets["Center"] = dect.Center

def inference_remote():
    # net = detectNet(
    #     "ssd-mobilenet-v2",
    #     threshold=0.1)
    # net = detectNet(
    #     "/home/jetcam/tensorrt_hello/jetson-inference/python/training/detection/ssd/pytorch-ssd/models/hardhatjpg/ssd-mobilenet.onnx",
    #     threshold=0.1)
    
    # net = detectNet(model="/home/jetcam/tensorrt_hello/jetson-inference/python/training/detection/ssd/pytorch-ssd/models/hardhatjpg/ssd-mobilenet.onnx",
    #                 labels="/home/jetcam/tensorrt_hello/jetson-inference/python/training/detection/ssd/pytorch-ssd/models/hardhatjpg/labels.txt",
    #                 input_blob="input_0",
    #                 output_cvg="scores",
    #                 output_bbox="boxes", 
    #                 threshold=0.1)

    # net = detectNet(model="/home/jetcam/tensorrt_hello/jetson-inference/python/training/detection/ssd/ssd512/pytorch-ssd/models/trafford_hamilton/ssd-mobilenet.onnx",
    #                 labels="/home/jetcam/tensorrt_hello/jetson-inference/python/training/detection/ssd/ssd512/pytorch-ssd/models/trafford_hamilton/labels.txt",
    #                 input_blob="input_0",
    #                 output_cvg="scores",
    #                 output_bbox="boxes", 
    #                 threshold=0.1)

    # net = detectNet(model="/home/jetcam/tensorrt_hello/jetson-inference/python/training/detection/ssd/ssd512/pytorch-ssd/models/trafford_hamilton_269_50000ep/ssd-mobilenet.onnx",
    #                 labels="/home/jetcam/tensorrt_hello/jetson-inference/python/training/detection/ssd/data/trafford_hamilton_269_50000ep/labels.txt",
    #                 input_blob="input_0",
    #                 output_cvg="scores",
    #                 output_bbox="boxes", 
    #                 threshold=0.1)
    net = detectNet(model="/home/jetcam/mb1025_voc1501/ssd-mobilenet.onnx",
                    input_blob="input_0",
                    output_cvg="scores",
                    output_bbox="boxes", 
                    threshold=0.1)
    #net = detectNet("ssd-mobilenet-v2", threshold=0.1)
    
    mssger = rabbit_mq.MessengerBasic(
        factory.TZAR_config())
    cnt = 0
    while True:
        cnt += 1
        print("waiting for image")
        message = mssger.check_in_box(blocking=True)
        print(f"checking{cnt}")
        if message is not None:
            print("message detected")
            try:
                print(message)
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

            with time_it("cuda from numpy"):
                cuda_mem = jetson_utils.cudaFromNumpy(img)
            all_dects = {}
            all_dects["SYSTEMINFO"] = "ANALYSED: MobileNet"

            with time_it("detectnet"):
                t1_start = perf_counter()
                detections = net.Detect(cuda_mem)
                t1_stop = perf_counter()
                elapsed_t_secs = t1_stop-t1_start
                print("elapsed_t_secs", elapsed_t_secs)
                print("--------------")
                #print(detections)
                
                dectdeets = None
                for index, deect in enumerate(detections):
                    dectdeets = {}
                    dectdeets["filename"] = "ANALYSED"
                    dectdeets["img_count"] = cnt
                    dectdeets["ClassID"] = deect.ClassID
                    dectdeets["Left"] = deect.Left
                    dectdeets["Top"] = deect.Top
                    dectdeets["Right"] = deect.Right
                    dectdeets["Bottom"] = deect.Bottom
                    dectdeets["Confidence"] = deect.Confidence
                    dectdeets["Center"] = deect.Center
                    dectdeets["index"] = str(index)
                    dectdeets["inference_time_secs"] = str(elapsed_t_secs)
                    print("--------------")
                    print(deect)
                    all_dects[index]=copy.deepcopy(dectdeets)
            output = json.dumps(all_dects)
            output_bytes = msgs.str_to_bytes(output)
            mssger.send_message(output_bytes)
                #print("Object Detectio)n | Network {:.0f} FPS".format(net.GetNetworkFPS()))
        else:
            time.sleep(0.2)
    
def inference_detectnet():
    image_list = []
    
    for filename in glob.glob('/home/jetcam/tensorrt_hello/jetson-inference/python/examples/match_media/*.png'): #assuming gif
        print("loading", filename)
        im=(cv2.imread(filename),filename)
        print(im[0].shape)
        if im[0] is None:
            raise Exception("empty load for", filename)
        image_list.append(im)
    net = detectNet("ssd-mobilenet-v2", threshold=0.1)
    #img = np.asarray(plasma(1000, 1000), dtype="uint8")
    for img, filename in image_list:
        with time_it("cuda from numpy"):
            cuda_mem = jetson_utils.cudaFromNumpy(img)
        with time_it("detectnet"):
            detections = net.Detect(cuda_mem)
            print("--------------")
            #print(detections)
            dectdeets = {}
            dectdeets["filename"] = filename
            for deect in detections:
                dectdeets["ClassID"] = deect.ClassID
                dectdeets["Left"] = deect.Left
                dectdeets["Top"] = deect.Top
                dectdeets["Right"] = deect.Right
                dectdeets["Bottom"] = deect.Bottom
                dectdeets["Confidence"] = deect.Confidence
            print(json.dumps(dectdeets))
            #print("Object Detectio)n | Network {:.0f} FPS".format(net.GetNetworkFPS()))


def main():
    infer_thread = threading.Thread(
                target=inference_detectnet,
                args=())
    infer_thread.start()

    #time.sleep(10000)
    plasma_img = np.asarray(plasma(400, 400), dtype="uint8")
    test_img = cv2.resize(plasma_img, (1920, 1080))
    test_img2 = cv2.resize(plasma_img, (1920, 1080))
    # set up parallel process streams
    streamLeft = vpi.Stream()
    streamRight = vpi.Stream()

    # using gstreamer instead of FFMPEG, Nvidia doesn't
    # support FFMPEG 100% for hardware dec/enc
    out_stream = cv2.VideoWriter(
        filename=gstreamer_pipeline_out_mod(),
        apiPreference=cv2.CAP_GSTREAMER,
        fourcc=0,
        fps=25.0,
        frameSize=(1920, 1080))

    cnt = 0
    while True:
        cnt +=1

        # time-based moving transform
        hom = np.array([
                [1, math.sin(cnt/10), 0],
                [0, 1, 0],
                [0, 0, 1]])

        ts = str(datetime.now().strftime("%H:%M:%S"))
        print(ts)
        timestamp_img = test_img.copy()
        cv2.putText(timestamp_img,
                    ts,
                    (80, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 0, 0),
                    thickness=4)

        with time_it("vpi processing (2)"):
            with vpi.Backend.CUDA:
                # upload image into GPU
                with streamLeft:
                    frame1 = vpi.asimage(timestamp_img)
                with streamRight:
                    frame2 = vpi.asimage(test_img2)

            with vpi.Backend.CUDA:
                # VIC processor can be used here - need to convert
                # image to correct format (NVIDIA VPI doc page)
                # but not much performance gain was experienced
                # https://docs.nvidia.com/vpi/algo_persp_warp.html#algo_persp_warp_perf
                with streamLeft:
                    frame1 = frame1.perspwarp(hom)
                with streamRight:
                    frame2 = frame2.perspwarp(hom)

            # request that streams finish their tasks
            streamLeft.sync()
            streamRight.sync()

        with time_it("output from CPU to mux"):
            # lock GPU memory to pull out buffer
            with frame1.rlock_cpu() as data:
                out_stream.write(data.copy())

def xavier_power_settings(sudo_pass):
    # obviously not secure - for quick and dirty testing
    sudo_password = sudo_pass
    commands = ['sudo nvpmodel -m 8', 'sudo jetson_clocks']
    check_pwr_mode = 'sudo nvpmodel -q'

    for command in commands:
        command = command.split()
        print("command" , command)
        cmd1 = subprocess.Popen(['echo', sudo_password], stdout=subprocess.PIPE)
        cmd2 = subprocess.Popen(['sudo', '-S'] + command, stdin=cmd1.stdout, stdout=subprocess.PIPE)
        print(cmd2.stdout.read().decode())
        time.sleep(2)

    print("checking power mode")
    cmd1 = subprocess.Popen(['echo', sudo_password], stdout=subprocess.PIPE)
    cmd2 = subprocess.Popen(['sudo', '-S'] + check_pwr_mode.split(), stdin=cmd1.stdout, stdout=subprocess.PIPE)
    capture = (cmd2.stdout.read().decode())
    print(capture)
    #if 'MODE_15W_2CORE' not in capture:
    #    raise Exception("XAVIER not in max power mode - try again with correct sudo pass")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawTextHelpFormatter,
        )
    
    parser.add_argument(
        '-sudopassword',
        help='sudo password to enable power settings',
        required=True)

    args = parser.parse_args()

    xavier_power_settings(sudo_pass=args.sudopassword)

    inference_remote()
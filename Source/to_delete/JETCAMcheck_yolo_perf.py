""" Notes

-Need to have opencv built with gstreamer support
print(cv2.getBuildInformation())

-Set Xavier to max power:
(do do manually or providing sudo password as script arg -p PASSWORD)
sudo nvpmodel -m 0
sudo jetson_clocks

-JTOP - helpful activity monitor
sudo apt-get install python3-pip -y
sudo python3 -m pip install --upgrade pip
sudo pip3 install -U jetson-stats
sudo reboot

-testcard output to MUX - NVENC chip should light up
gst-launch-1.0 videotestsrc ! video/x-raw ! nvvidconv ! nvv4l2h264enc maxperf-enable=1 ! h264parse ! flvmux streamable=true ! queue ! rtmpsink location='rtmp://global-live.mux.com:5222/app/51bc0427-ad29-2909-4979-11ee335d2b53'

-to read
https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md
https://github.com/Fuell-ai/acm/blob/jetcam_bits/jetcam/functions/nvidia_gpu_buff_share.py
"""
import msgs
import cv2
import time
import math
import vpi
import numpy as np
from contextlib import contextmanager
import math
import colorsys
from PIL import Image
from jetson_inference import detectNet
import jetson_utils
import threading
import queue
import copy
import json
from datetime import datetime
import subprocess
import argparse
import rabbit_mq
import factory
import messaging
import torch
import torchvision.transforms as torchtransforms
import torchvision.transforms.functional as fn
import tensorrt as trt
from time import perf_counter

from collections import OrderedDict,namedtuple
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def letterbox_pytorch(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # tensor images are expected to be in  (C, H, W)
    #input here is in (H, W, C)
    # Resize and pad image while meeting stride-multiple constraints
    im = im.permute(2, 0, 1) # swap axes - pytorch expects CHW
    shape= list(im.size())[1:] # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    new_unpad = tuple(reversed(new_unpad))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        transform = torchtransforms.Resize(size=new_unpad)
        resized_im = transform(im)

    left = int(round(dh - 0.1))
    top = int(round(dw - 0.1))
    # TODO empty image - put in another function and cache it
    channels = list(resized_im.size())[0]
    empty_tensor = torch.ones((channels), *new_shape,  device='cuda:0')
    # add border - image might be smaller than output size, fill out border and centralise
    resized_height = list(resized_im.size())[1]
    resized_width = list(resized_im.size())[2]
    print("letterboxing")
    print("resized_height", resized_height)
    print("resized_width", resized_width)
    print("left", left)
    print("top", top)
    empty_tensor[:, top:resized_height+top, left:resized_width+left] = resized_im
    return empty_tensor, r, (dw, dh)

@contextmanager
def time_it(comment):
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        toc: float = time.perf_counter()
        #if "total" in comment:
        print(f"{comment}: {1000*(toc - tic):.3f}ms")
        #print("  ")

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
    return np.asarray(out, dtype="uint8")

def gstreamer_out():
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
        "rtmpsink location=rtmp://global-live.mux.com:5222/app/eb27591f-6aa1-aaf9-8be8-978237205f5a sync=false"
    )

def frame_grabber():
    # videoSource.Capture() returns a jetson.utils.cudaImage which is what we need

    #  img = camera.Capture()

    #  if imgOutput is None:
    #       imgOutput = jetson.utils.cudaAllocMapped(width=img.width * 0.5, height=img.height * 0.5, format=img.format)

    #  # rescale the image (the dimensions are taken from the image capsules)
    #  jetson.utils.cudaResize(img, imgOutput)

    #  print(imgOutput)
    #  detections = net.Detect(imgOutput)

    # this command worked on Jetson Nano with dev board, change sensor_id for camera 1/2
    #but is not straight to GPU
    cap = cv2.VideoCapture("nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=(int)3840, height=(int)2464,format=(string)NV12, framerate=(fraction)10/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

def detector(inbox, outbox):
    net = detectNet(
        "ssd-mobilenet-v2",
        threshold=0.3)
    while True:
        if inbox.empty() is False:
            # blocking call here to see how long it takes to 
            # pop image off queue
            with time_it("INF: get object off queue"):
                np_img = inbox.get(block=True)

            # upload to GPU we do not want to do this twice
            # in theory VPI should be able to share a compatible
            # GPU pointer after we have uploaded it during VC
            # stage - WIP
            with time_it("INF: image to cuda"):
                cuda_mem = jetson_utils.cudaFromNumpy(np_img)

            with time_it("INF: detectnet"):
                detections = net.Detect(cuda_mem)

            with time_it("INF: feedback dects"):
                all_dects = {}
                dectdeets = None
                for index, deect in enumerate(detections):
                    dectdeets = {}
                    dectdeets["ClassID"] = deect.ClassID
                    dectdeets["Left"] = deect.Left
                    dectdeets["Top"] = deect.Top
                    dectdeets["Right"] = deect.Right
                    dectdeets["Bottom"] = deect.Bottom
                    dectdeets["Confidence"] = deect.Confidence
                    dectdeets["index"] = str(index)
                    all_dects[index]=copy.deepcopy(dectdeets)
                output = json.dumps(all_dects)
                if outbox.empty():
                    outbox.put(output)
        else:
            time.sleep(0.02)

def postprocess(boxes,r,dwdh):
    dwdh = torch.tensor(dwdh*2).to(boxes.device)
    boxes -= dwdh
    boxes /= r
    return boxes

def detector_yolo(inbox, outbox, ID):
    # #trt engine expecrts torch.tensor object as input image
    w = '/home/jetcam/yolo/yolov7-tiny-nms.trt'
    device = torch.device('cuda:0')
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()

    imgcnt = 0
    cuda_buff = None
    while True:
        if inbox.empty() is False:
            # blocking call here to see how long it takes to 
            # pop image off queue
            with time_it(f"{ID}: get object off queue"):
                cuda_obj = inbox.get(block=True)
            imgcnt += 1

            #print(f"{ID}: image {imgcnt}")
            # with time_it(f"{ID}: convert CI to VPIim"):
            #     with vpi.Backend.CUDA:
            #     # upload image into GPU
            #         frame1 = vpi.asimage(cuda_obj)#.convert(vpi.Format.RGB8)
            # with time_it(f"{ID}: convert VPIim to TorchTensor)"):
            #     with frame1.rlock_cuda() as cuda_buffer:
            #         # Perform another operation using PyTorch
            #         torch_tensor = torch.as_tensor(cuda_buffer, device= torch.device('cuda'))# might need :0


            if cuda_buff is None:
                with time_it(f"{ID} create GPU buffer (once only):"):
                    cuda_buff = jetson_utils.cudaAllocMapped(
                            width=cuda_obj.width,
                            height=cuda_obj.height,
                            format=cuda_obj.format)
                            
            with time_it(f"{ID}::::::::::: total time:::::::"):
                with time_it(f"{ID} copy GPU buffer:"):
                    jetson_utils.cudaMemcpy(cuda_buff, cuda_obj)


                with time_it(f"{ID}: convert VPIim to TorchTensor)"):
                    torch_tensor = torch.as_tensor(cuda_buff, device= torch.device('cuda'))# might need :0
                with time_it(f"{ID}: pytorch letterbox pre process img)"):
                    pt_image, ratio, _ = letterbox_pytorch(torch_tensor, auto=False)
                    #print("PTimage.shape", image.shape)
                    # add another dim, to match expected input
                    # [batch channel height width] where
                    # batch is one image
                    pt_image = pt_image.unsqueeze(0)
                    pt_image/=255 # not sure if need this
                    #print("PTexpand_dims.shape", image.shape)
                    #print("PTiscontiguous", image.is_contiguous())
                    #print("PTtype", image.dtype)
                    # already continguous and float 32
                    #print("PTtype", type(pt_image))
                # with time_it(f"{ID} convert to numpy?"):
                #     np_image = np.asarray(torch_tensor.cpu())
                # with time_it(f"{ID} pre process (CPU)"):
                #     image = np_image.copy()
                #     #print("image.shape", image.shape)
                #     image, _, _ = letterbox(image, auto=False)
                #     #print("letterbox.shape", image.shape)
                #     image = image.transpose((2, 0, 1))
                #     #print("transpose.shape", image.shape)
                #     image = np.expand_dims(image, 0)
                #     #print("expand_dims.shape", image.shape)
                #     image = np.ascontiguousarray(image)
                #     #print("ascontiguousarray.shape", image.shape)
                #     #print("type", type(image[0,0,1,1]))
                #     im = image.astype(np.float32)
                #     im = torch.from_numpy(im).to(torch.device('cuda'))
                #     im/=255 
                #     print("type", type(im))


                with time_it(f"{ID} inference"):
                    binding_addrs['images'] = int(pt_image.data_ptr())
                    context.execute_v2(list(binding_addrs.values()))

                with time_it(f"{ID}: extract dects"):
                    nums = bindings['num_dets'].data
                    boxes = bindings['det_boxes'].data
                    scores = bindings['det_scores'].data
                    classes = bindings['det_classes'].data

                    boxes = boxes[0,:nums[0][0]]
                    scores = scores[0,:nums[0][0]]
                    classes = classes[0,:nums[0][0]]
                    print(f"INF RESULTS in {len(boxes)}")
                    for box,score,cl in zip(boxes,scores,classes):
                        pass
                        print("RESULT::::::", cl,score,box)
        else:
            print("f{ID}: Waiting for image")
            time.sleep(0.02)


def remote_inference_yolo8_nonTRT_cpu():
    from ultralytics import YOLO
    mssger = rabbit_mq.MessengerBasic(
    factory.TZAR_config())
    yolo_path = r"/home/jetcam/yolo/from-rav/yolo_weights_269/269_v8m.pt"
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

def remote_inference_yolo8_nonTRT_gpu():
    from ultralytics import YOLO
    mssger = rabbit_mq.MessengerBasic(
    factory.TZAR_config())
    yolo_path = r"/home/jetcam/yolo/from-rav/yolo_weights_269/269_v8s.pt"
    model = YOLO("yolov8n.pt")
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
                device="0", # cpu, or 0,1,2,3 for cuda
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
                    dectdeets["ClassID"] =  int(cls.cpu().numpy())
                    dectdeets["Left"] = int(xywh.cpu().numpy()[0] - (xywh.cpu().numpy()[2]/2))
                    dectdeets["Top"] = int(xywh.cpu().numpy()[1] - (xywh.cpu().numpy()[3]/2))
                    dectdeets["Right"] = int(xywh.cpu().numpy()[0] + (xywh.cpu().numpy()[2]/2))
                    dectdeets["Bottom"] = int(xywh.cpu().numpy()[1] + (xywh.cpu().numpy()[3]/2))
                    dectdeets["Confidence"] = float(conf.cpu().numpy())
                    center = [int(xywh.cpu().numpy()[0]),int(xywh.cpu().numpy()[1])]
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

def remote_inference_yolo7_trt():
    # #trt engine expecrts torch.tensor object as input image
    #w = '/home/jetcam/yolo/yolov7-tiny-nms.trt'
    w = '/home/jetcam/yolo/from-rav/TRT/trafford269/yolov7-tiny-custom-fp16.trt'
    device = torch.device('cuda:0')
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()

    mssger = rabbit_mq.MessengerBasic(
    factory.TZAR_config())
    imgcnt = 0
    cuda_buff = None
    while True:
        imgcnt += 1
        print("waiting for image")
        message = mssger.check_in_box(blocking=True)
        print(f"checking{imgcnt}")
        if message is not None:
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

            #  lazy way to get into expected format
            file_path = "/home/jetcam/tempimg.jpg"
            cv2.imwrite(file_path, img)
            print("creating input object")
            input = jetson_utils.videoSource(file_path, ["--loop=-1"])
            img_to_anlyse = input.Capture(format='rgb8')
            print("copying buffer")
            cuda_buff = jetson_utils.cudaAllocMapped(
                        width=img_to_anlyse.width,
                        height=img_to_anlyse.height,
                        format=img_to_anlyse.format)
            t1_start = perf_counter()
            jetson_utils.cudaMemcpy(cuda_buff, img_to_anlyse)
            print("converting to tensor")
            torch_tensor = torch.as_tensor(cuda_buff, device= torch.device('cuda'))# might need :0
            pt_image, ratio, (dw,dh) = letterbox_pytorch(torch_tensor, auto=False)
            pt_image = pt_image.unsqueeze(0)
            pt_image/=255 # not sure if need this
            binding_addrs['images'] = int(pt_image.data_ptr())
            print("executing inference")
            context.execute_v2(list(binding_addrs.values()))

            nums = bindings['num_dets'].data
            boxes = bindings['det_boxes'].data
            scores = bindings['det_scores'].data
            classes = bindings['det_classes'].data

            boxes = boxes[0,:nums[0][0]]
            scores = scores[0,:nums[0][0]]
            classes = classes[0,:nums[0][0]]
            print(f"INF RESULTS in {len(boxes)}")
            all_dects = {}
            all_dects["SYSTEMINFO"] = "ANALYSED: YOLO7"
            dectdeets = None
            index = 0
            t1_stop = perf_counter()
            for box,score,cl in zip(boxes,scores,classes):
                dectdeets = {}
                print("->", "ratio", ratio)
                print("cls", int(cl.cpu().numpy()))
                print("score", float(score.cpu().numpy()))
                tlbr=box.cpu().numpy()[:]
                left = int(tlbr[0] * (1/ratio))
                top = int((tlbr[1] - dw) * (1/ratio))
                right = int(tlbr[2] * (1/ratio))
                lower = int((tlbr[3] - dw) * (1/ratio))
                print("ConvertedBox", left, top, right, lower)
                print("box", box.cpu()[:])
                print("box_ratio", box.cpu().numpy()[:]*(1/ratio))
                print("(dw,dh)", (dw,dh))

                dectdeets["filename"] = "ANALYSED"
                dectdeets["img_count"] = imgcnt
                dectdeets["ClassID"] =  int(cl.cpu().numpy())
                dectdeets["Left"] = left
                dectdeets["Top"] = top
                dectdeets["Right"] = right
                dectdeets["Bottom"] = lower
                dectdeets["Confidence"] = float(score.cpu().numpy())
                center = [int((right+left)/2), int((lower+top)/2)]
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

def remote_inference_yolo8_trt_gpu():
    # #trt engine expecrts torch.tensor object as input image
    #w = '/home/jetcam/yolo/yolov7-tiny-nms.trt'
    #'"C:\Working\ML\yolo\_y8_TRT_Hamilton269\yolov8m-custom.trt"'
    w = '/home/jetcam/yolo/_y8_TRT_Hamilton269/yolov8m-custom.trt'
    device = torch.device('cuda:0')
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()

    mssger = rabbit_mq.MessengerBasic(
    factory.TZAR_config())
    imgcnt = 0
    cuda_buff = None
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

            #  lazy way to get into expected format
            file_path = "/home/jetcam/tempimg.jpg"
            cv2.imwrite(file_path, img)
            print("creating input object")
            input = jetson_utils.videoSource(file_path, ["--loop=-1"])
            img_to_anlyse = input.Capture(format='rgb8')
            print("copying buffer")
            cuda_buff = jetson_utils.cudaAllocMapped(
                        width=img_to_anlyse.width,
                        height=img_to_anlyse.height,
                        format=img_to_anlyse.format)
            t1_start = perf_counter()
            jetson_utils.cudaMemcpy(cuda_buff, img_to_anlyse)
            print("converting to tensor")
            torch_tensor = torch.as_tensor(cuda_buff, device= torch.device('cuda'))# might need :0
            pt_image, ratio, (dw,dh) = letterbox_pytorch(torch_tensor, auto=False)
            pt_image = pt_image.unsqueeze(0)
            pt_image/=255 # not sure if need this
            binding_addrs['images'] = int(pt_image.data_ptr())
            print("executing inference")
            context.execute_v2(list(binding_addrs.values()))

            #print([bindings[i] for i in bindings.keys() if i != "images"])
            #print(bindings.keys())
            #print(bindings.values())
            nums = bindings['num_dets'].data
            boxes = bindings['bboxes'].data
            scores = bindings['scores'].data
            classes = bindings['labels'].data
            print(bindings['num_dets'])
            no_of_dects = nums[0][0]
            if no_of_dects > 0:
                raise Exception("whoohoo")
            print("no_of_dects",no_of_dects.cpu().numpy())
            boxes = boxes[0,:nums[0][0]]
            scores = scores[0,:nums[0][0]]
            classes = classes[0,:nums[0][0]]
            print(f"INF RESULTS in {len(boxes)}")
            all_dects = {}
            all_dects["SYSTEMINFO"] = "ANALYSED: YOLO7"
            dectdeets = None
            index = 0
            t1_stop = perf_counter()
            for box,score,cl in zip(boxes,scores,classes):
                dectdeets = {}
                print("->", "ratio", ratio)
                print("cls", int(cl.cpu().numpy()))
                print("score", float(score.cpu().numpy()))
                tlbr=box.cpu().numpy()[:]
                left = int(tlbr[0] * (1/ratio))
                top = int((tlbr[1] - dw) * (1/ratio))
                right = int(tlbr[2] * (1/ratio))
                lower = int((tlbr[3] - dw) * (1/ratio))
                print("ConvertedBox", left, top, right, lower)
                print("box", box.cpu()[:])
                print("box_ratio", box.cpu().numpy()[:]*(1/ratio))
                print("(dw,dh)", (dw,dh))

                dectdeets["filename"] = "ANALYSED"
                dectdeets["img_count"] = imgcnt
                dectdeets["ClassID"] =  int(cl.cpu().numpy())
                dectdeets["Left"] = left
                dectdeets["Top"] = top
                dectdeets["Right"] = right
                dectdeets["Bottom"] = lower
                dectdeets["Confidence"] = float(score.cpu().numpy())
                center = [int((right+left)/2), int((lower+top)/2)]
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



def detector_yolo_batch(inbox, outbox, ID):
    # #trt engine expecrts torch.tensor object as input image
    w = '/home/jetcam/yolo/yolov7-tiny-nms.trt'
    device = torch.device('cuda:0')
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()

    imgcnt = 0
    cuda_buff = None
    batch_tensor = None
    while True:
        if inbox.empty() is False:
            # blocking call here to see how long it takes to 
            # pop image off queue
            with time_it(f"{ID}: get object off queue"):
                cuda_obj = inbox.get(block=True)
            imgcnt += 1

            #print(f"{ID}: image {imgcnt}")
            # with time_it(f"{ID}: convert CI to VPIim"):
            #     with vpi.Backend.CUDA:
            #     # upload image into GPU
            #         frame1 = vpi.asimage(cuda_obj)#.convert(vpi.Format.RGB8)
            # with time_it(f"{ID}: convert VPIim to TorchTensor)"):
            #     with frame1.rlock_cuda() as cuda_buffer:
            #         # Perform another operation using PyTorch
            #         torch_tensor = torch.as_tensor(cuda_buffer, device= torch.device('cuda'))# might need :0


            if cuda_buff is None:
                cuda_buff = []
                with time_it(f"{ID} create GPU buffer (once only):"):
                    for cudaimg in cuda_obj:
                        cuda_buff.append(jetson_utils.cudaAllocMapped(
                                width=cudaimg.width,
                                height=cudaimg.height,
                                format=cudaimg.format))
                        
                    # assumptions here are that all images are same size
                    # and all have 3 channels
                    # needs to in form CHW
                    yolo_input_img_size = 640
                    batch_tensor = torch.zeros(
                        len(cuda_obj),
                        3,
                        yolo_input_img_size,
                        yolo_input_img_size,
                        device='cuda:0')
            with time_it(f"{ID}::::::::::: total time:::::::"):
                with time_it(f"{ID} copy GPU buffer:"):
                    for obj, buff in zip(cuda_obj, cuda_buff):
                        jetson_utils.cudaMemcpy(buff, obj)
                with time_it(f"{ID}: convert VPIim to TorchTensor)"):
                    torch_tensors = []
                    for buff in cuda_buff:
                        torch_tensors.append(torch.as_tensor(buff, device= torch.device('cuda')))# might need :0
                with time_it(f"{ID}: pytorch letterbox pre process img)"):
                    pt_images = []
                    for index, torchtensor in enumerate(torch_tensors):
                        pt_image, _, _ = letterbox_pytorch(torchtensor, auto=False)
                        batch_tensor[index, :, :, :] = pt_image

                        # add another dim, to match expected input
                        # [batch channel height width] where
                        # batch is one image
                        #pt_image = pt_image.unsqueeze(0)
                        #pt_image/=255 # not sure if need this
                        #pt_images.append(pt_image)
                        #batch_tensor
                    batch_tensor/=255 # not sure if need this
                
                # for index, torchtensor in enumerate(torch_tensors):
                #     if batch_tensor[index, :, :, :] == batch_tensor[index+1, :, :, :]:
                #         print("identical tensors wtf")
                #     if index == len(torch_tensors)-2:
                #         index = 100
                with time_it(f"{ID} inference"):
                    binding_addrs['images'] = int(batch_tensor.data_ptr())
                    context.execute_v2(list(binding_addrs.values()))

                with time_it(f"{ID}: extract dects"):
                    nums = bindings['num_dets'].data
                    boxes = bindings['det_boxes'].data
                    scores = bindings['det_scores'].data
                    classes = bindings['det_classes'].data
                    boxes = boxes[0,:nums[0][0]]
                    scores = scores[0,:nums[0][0]]
                    classes = classes[0,:nums[0][0]]
                    for box,score,cl in zip(boxes,scores,classes):
                        print("INF RESULT  {index}::::::", cl,score,box)
        else:
            print("f{ID}: Waiting for image")
            time.sleep(0.02)


def detector_cuda(inbox, outbox, ID):
    net = detectNet(
        "ssd-mobilenet-v2",
        threshold=0.3)
    

    cuda_buff = None
    while True:
        if inbox.empty() is False:
            # blocking call here to see how long it takes to 
            # pop image off queue
            with time_it(f"{ID}: get object off queue"):
                cuda_obj = inbox.get(block=True)

            if cuda_buff is None:
                with time_it(f"{ID} create GPU buffer (once only):"):
                    cuda_buff = jetson_utils.cudaAllocMapped(
                            width=cuda_obj.width,
                            height=cuda_obj.height,
                            format=cuda_obj.format)
            with time_it(f"{ID}::::::::::: total time :::::::"):
                # copy image or something goes weird
                # allocate this outside of loop
                with time_it(f"{ID} copy GPU buffer:"):
                    jetson_utils.cudaMemcpy(cuda_buff, cuda_obj)


                # upload to GPU we do not want to do this twice
                # in theory VPI should be able to share a compatible
                # GPU pointer after we have uploaded it during VC
                # stage - WIP
                #with time_it("INF: image to cuda"):
                #    cuda_mem = jetson_utils.cudaFromNumpy(np_img)

                with time_it(f"{ID} detectnet"):
                    detections = net.Detect(cuda_buff)

                with time_it(f"{ID}: feedback dects"):
                    all_dects = {}
                    dectdeets = None
                    for index, deect in enumerate(detections):
                        dectdeets = {}
                        dectdeets["ClassID"] = deect.ClassID
                        dectdeets["Left"] = deect.Left
                        dectdeets["Top"] = deect.Top
                        dectdeets["Right"] = deect.Right
                        dectdeets["Bottom"] = deect.Bottom
                        dectdeets["Confidence"] = deect.Confidence
                        dectdeets["index"] = str(index)
                        all_dects[index]=copy.deepcopy(dectdeets)
                    output = json.dumps(all_dects)
                    if outbox.empty():
                        outbox.put(output)
        else:
            print(f"{ID}: Waiting for image")
            time.sleep(0.02)

def main():
    _in_box = queue.Queue(maxsize=3)
    _dects_box = queue.Queue(maxsize=3)
    detection_worker = threading.Thread(
                target=detector,
                args=(_in_box, _dects_box, ))

    detection_worker.start()

    input_size = (1920, 1080) #(3840, 2160)
    output_size =  (1920, 1080)
    file_path = "/home/jetcam/tensorrt_hello/jetson-inference/data/images/humans_0.jpg"
    img_people = cv2.imread(file_path)
    img_people = cv2.resize(img_people, input_size)

    # set up parallel process streams
    streamLeft = vpi.Stream()
    streamRight = vpi.Stream()

    # using gstreamer instead of FFMPEG, Nvidia doesn't
    # support FFMPEG 100% for hardware dec/enc
    #ensure opencv is built with gstreamer support
    out_stream = cv2.VideoWriter(
        filename=gstreamer_out(),
        apiPreference=cv2.CAP_GSTREAMER,
        fourcc=0,
        fps=25.0,
        frameSize=output_size)

    cnt = 0
    while True:
        cnt +=1
        # time-based moving transform
        hom = np.array([
                [1, (math.sin(cnt/10)), 0],
                [0, 1, 0],
                [0, 0, 1]])

        print("------")
        with time_it("VC: upload to GPU (2)"):
            with vpi.Backend.CUDA:
                # upload image into GPU
                with streamLeft:
                    frame1 = vpi.asimage(img_people)
                with streamRight:
                    frame2 = vpi.asimage(img_people)

        with time_it("VC: perp processing & sync (2)"):
            with vpi.Backend.CUDA:
                # VIC processor can be used here - need to convert
                # image to correct format (NVIDIA VPI doc page)
                # but not much performance gain 
                # if we run out of GPU it will be useful
                # https://docs.nvidia.com/vpi/algo_persp_warp.html#algo_persp_warp_perf
                with streamLeft:
                    frame1 = frame1.perspwarp(hom)
                with streamRight:
                    frame2 = frame2.perspwarp(hom)

                # wait for GPU streams to finish their tasks
                streamLeft.sync()
                streamRight.sync()

        result_dict  = None
        if _dects_box.empty() is False:
            with time_it("VC: get detections off queue"):
                try:
                    result_dict = _dects_box.get(block=False)
                    result_dict = json.loads(result_dict)
                except queue.Empty:
                    pass

        with time_it("VC: output GPU to CPU (1)"):
            # lock GPU memory to pull out buffer
            # here it is assumed the payload is
            # 1080p
            with frame1.rlock_cpu() as data:
                img_copy = data.copy()

        if _in_box.empty() :
            with time_it("VC: put image on queue (2)"):
                _in_box.put(img_copy)
                _in_box.put(img_copy)

        with time_it("VC: draw on rectangles"):
            ts = str(datetime.now().strftime("%H:%M:%S"))
            cv2.putText(
                img_copy,
                ts,
                (80, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3,
                color=(255, 0, 0),
                thickness=4)

            if result_dict is not None:
                for dect in result_dict.values():
                    print(dect)
                    cv2.rectangle(
                        img_copy,
                        (int(dect["Left"]),int(dect["Top"])),
                        (int(dect["Right"]),int(dect["Bottom"])),
                        (255, 0, 0),
                        3)

        with time_it("VC: output to mux"):
            out_stream.write(img_copy)

# resize an image
def resize(img, resize_factor):
	resized_img = jetson_utils.cudaAllocMapped(width=img.width * resize_factor[0],
								  height=img.height * resize_factor[1],
                                  format=img.format)

	jetson_utils.cudaResize(img, resized_img)
	return resized_img

def main_videocap():
    _in_box = queue.Queue(maxsize=3)
    _dects_box = queue.Queue(maxsize=3)

    workers = []
    for id in range (0,1):  
        workers.append(threading.Thread(
                    target=detector_cuda,
                    args=(_in_box, _dects_box, f"IF{id}", )))
        
        workers[-1].start()

    input_size = (1920, 1080) #(3840, 2160)
    output_size =  (1920, 1080)


    file_path = "/home/jetcam/tensorrt_hello/jetson-inference/data/images/humans_0.jpg"
    img_people = cv2.imread(file_path)
    img_people = cv2.resize(img_people, input_size)
    file_path_save = file_path.replace(".jpg", "_copy.jpg")
    cv2.imwrite(file_path_save, img_people)

    # more args etc
    # https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#source-code
    # videosource returns its own GPU buffer so don't have to
    # define one with cudalloc
    input = jetson_utils.videoSource(file_path_save, ["--loop=-1"])
    img_people = input.Capture(format='rgb8')
    #img_people = resize(img_people, input_size)

    # set up parallel process streams
    streamLeft = vpi.Stream()
    streamRight = vpi.Stream()

    # using gstreamer instead of FFMPEG, Nvidia doesn't
    # support FFMPEG 100% for hardware dec/enc
    #ensure opencv is built with gstreamer support
    out_stream = cv2.VideoWriter(
        filename=gstreamer_out(),
        apiPreference=cv2.CAP_GSTREAMER,
        fourcc=0,
        fps=25.0,
        frameSize=output_size)
    
    # not in loop while pulling from images - disc read time
    input_img_1 = input.Capture(format='rgb8')
    input_img_2 = input.Capture(format='rgb8')
    
    cnt = 0
    while True:
        cnt +=1
        # time-based moving transform
        hom = np.array([
                [1, (math.sin(cnt/10)), 0],
                [0, 1, 0],
                [0, 0, 1]])

        print("------")
        with time_it("VC: upload to GPU (2)"):
            with vpi.Backend.CUDA:
            # upload image into GPU
                with streamLeft:
                    frame1 = vpi.asimage(input_img_1)#.convert(vpi.Format.RGB8)
                with streamRight:
                    frame2 = vpi.asimage(input_img_2)#.convert(vpi.Format.RGB8)

        with time_it("VC: perp processing & sync (2)"):
            with vpi.Backend.CUDA:
                # VIC processor can be used here - need to convert
                # image to correct format (NVIDIA VPI doc page)
                # but not much performance gain 
                # if we run out of GPU it will be useful
                # https://docs.nvidia.com/vpi/algo_persp_warp.html#algo_persp_warp_perf
                with streamLeft:
                    frame1 = frame1.perspwarp(hom)
                with streamRight:
                    frame2 = frame2.perspwarp(hom)

                # wait for GPU streams to finish their tasks
            streamLeft.sync()
            streamRight.sync()

        result_dict  = None
        if _dects_box.empty() is False:
            with time_it("VC: get detections off queue"):
                try:
                    result_dict = _dects_box.get(block=False)
                    result_dict = json.loads(result_dict)
                except queue.Empty:
                    pass

        with time_it("VC: output GPU to CPU (1)"):
            # lock GPU memory to pull out buffer
            # here it is assumed the payload is
            # 1080p
            with frame1.rlock_cpu() as data:
                img_copy = data.copy()

        if _in_box.empty() :
            with time_it("VC: put image on queue (2)"):
                _in_box.put(input_img_1)
                _in_box.put(input_img_2)
        #time.sleep(1000)
        with time_it("VC: draw on rectangles"):
            ts = str(datetime.now().strftime("%H:%M:%S"))
            cv2.putText(
                img_copy,
                ts,
                (80, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3,
                color=(255, 0, 0),
                thickness=4)

            if result_dict is not None:
                print(f"detections found?{len(result_dict.values())}")
                for dect in result_dict.values():
                    print(dect)
                    cv2.rectangle(
                        img_copy,
                        (int(dect["Left"]),int(dect["Top"])),
                        (int(dect["Right"]),int(dect["Bottom"])),
                        (255, 0, 0),
                        3)

        with time_it("VC: output to mux"):
            #print(img_copy.shape)
            out_stream.write(img_copy)


def main_videocap_yolo_batch():
    _in_box = queue.Queue(maxsize=3)
    _dects_box = queue.Queue(maxsize=3)

    workers = []
    for id in range (0,1):  
        workers.append(threading.Thread(
                    target=detector_yolo_batch,
                    args=(_in_box, _dects_box, f"IF{id}", )))
        
        workers[-1].start()

    input_size = (1920, 1080) #(3840, 2160)
    output_size =  (1920, 1080)


    file_path = [
        "/home/jetcam/tensorrt_hello/jetson-inference/data/images/humans_5.jpg",
        "/home/jetcam/tensorrt_hello/jetson-inference/data/images/humans_1.jpg",
        "/home/jetcam/tensorrt_hello/jetson-inference/data/images/humans_2.jpg",
        "/home/jetcam/tensorrt_hello/jetson-inference/data/images/humans_3.jpg",
        "/home/jetcam/tensorrt_hello/jetson-inference/data/images/humans_4.jpg",
        "/home/jetcam/tensorrt_hello/jetson-inference/data/images/humans_0.jpg"]
    
    sources = []
    captures = []
    for img in file_path:
        img_people = cv2.imread(img)
        if img_people is None:
            raise Exception("Image is empty", img)
        img_people = cv2.resize(img_people, input_size)
        file_path_save = img.replace(".jpg", "_copy.jpg")
        cv2.imwrite(file_path_save, img_people)
        sources.append(
            jetson_utils.videoSource(file_path_save, ["--loop=-1"]))
        captures.append(sources[-1].Capture(format='rgb8'))

    # more args etc
    # https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#source-code
    # videosource returns its own GPU buffer so don't have to
    # define one with cudalloc
    #input = jetson_utils.videoSource(file_path_save, ["--loop=-1"])
    #img_people = input.Capture(format='rgb8')
    #img_people = resize(img_people, input_size)

    # set up parallel process streams
    streamLeft = vpi.Stream()
    streamRight = vpi.Stream()

    # using gstreamer instead of FFMPEG, Nvidia doesn't
    # support FFMPEG 100% for hardware dec/enc
    #ensure opencv is built with gstreamer support
    out_stream = cv2.VideoWriter(
        filename=gstreamer_out(),
        apiPreference=cv2.CAP_GSTREAMER,
        fourcc=0,
        fps=25.0,
        frameSize=output_size)
    
    # not in loop while pulling from images - disc read time
    input_img_1 = captures[1]
    input_img_2 = captures[2]
    
    cnt = 0
    while True:
        cnt +=1
        # time-based moving transform
        hom = np.array([
                [1, (math.sin(cnt/10)), 0],
                [0, 1, 0],
                [0, 0, 1]])

        print("------")
        with time_it("VC: upload to GPU (2)"):
            with vpi.Backend.CUDA:
            # upload image into GPU
                with streamLeft:
                    frame1 = vpi.asimage(input_img_1)#.convert(vpi.Format.RGB8)
                with streamRight:
                    frame2 = vpi.asimage(input_img_2)#.convert(vpi.Format.RGB8)

        with time_it("VC: perp processing & sync (2)"):
            with vpi.Backend.CUDA:
                # VIC processor can be used here - need to convert
                # image to correct format (NVIDIA VPI doc page)
                # but not much performance gain 
                # if we run out of GPU it will be useful
                # https://docs.nvidia.com/vpi/algo_persp_warp.html#algo_persp_warp_perf
                with streamLeft:
                    frame1 = frame1.perspwarp(hom)
                with streamRight:
                    frame2 = frame2.perspwarp(hom)

                # wait for GPU streams to finish their tasks
        streamLeft.sync()
        streamRight.sync()

        result_dict  = None
        if _dects_box.empty() is False:
            with time_it("VC: get detections off queue"):
                try:
                    result_dict = _dects_box.get(block=False)
                    result_dict = json.loads(result_dict)
                except queue.Empty:
                    pass

        with time_it("VC: output GPU to CPU (1)"):
            # lock GPU memory to pull out buffer
            # here it is assumed the payload is
            # 1080p
            with frame1.rlock_cpu() as data:
                img_copy = data.copy()

        if _in_box.empty() :
            with time_it(f"VC: put image on queue (list {len(captures)} captures)"):
                _in_box.put(captures)
        #time.sleep(1000)
        with time_it("VC: draw on rectangles"):
            ts = str(datetime.now().strftime("%H:%M:%S"))
            cv2.putText(
                img_copy,
                ts,
                (80, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3,
                color=(255, 0, 0),
                thickness=4)

            if result_dict is not None:
                print(f"detections found?{len(result_dict.values())}")
                for dect in result_dict.values():
                    print(dect)
                    cv2.rectangle(
                        img_copy,
                        (int(dect["Left"]),int(dect["Top"])),
                        (int(dect["Right"]),int(dect["Bottom"])),
                        (255, 0, 0),
                        3)

        with time_it("VC: output to mux"):
            #print(img_copy.shape)
            out_stream.write(img_copy)
        time.sleep(2)

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
    if '20W' not in capture:
       raise Exception("XAVIER not in max power mode - try again with correct sudo pass")

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

    main_videocap()

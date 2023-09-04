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
import random
import msgs
from typing import List, Optional, Tuple, Union
from pathlib import Path
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
from numpy import ndarray

from collections import OrderedDict,namedtuple


def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)

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

def postprocess(boxes,r,dwdh):
    dwdh = torch.tensor(dwdh*2).to(boxes.device)
    boxes -= dwdh
    boxes /= r
    return boxes



# resize an image
def resize(img, resize_factor):
	resized_img = jetson_utils.cudaAllocMapped(width=img.width * resize_factor[0],
								  height=img.height * resize_factor[1],
                                  format=img.format)

class TRTModule(torch.nn.Module):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }

    def __init__(self, weight: Union[str, Path],
                 device: Optional[torch.device]) -> None:
        super(TRTModule, self).__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=device)
        self.__init_engine()
        self.__init_bindings()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()
        num_bindings = model.num_bindings
        names = [model.get_binding_name(i) for i in range(num_bindings)]

        self.bindings: List[int] = [0] * num_bindings
        num_inputs, num_outputs = 0, 0

        for i in range(num_bindings):
            if model.binding_is_input(i):
                num_inputs += 1
            else:
                num_outputs += 1

        self.num_bindings = num_bindings
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]
        self.idx = list(range(self.num_outputs))

    def __init_bindings(self) -> None:
        idynamic = odynamic = False
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape'))
        inp_info = []
        out_info = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                idynamic |= True
            inp_info.append(Tensor(name, dtype, shape))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert self.model.get_binding_name(i) == name
            dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                odynamic |= True
            out_info.append(Tensor(name, dtype, shape))

        if not odynamic:
            self.output_tensor = [
                torch.empty(info.shape, dtype=info.dtype, device=self.device)
                for info in out_info
            ]
        self.idynamic = idynamic
        self.odynamic = odynamic
        self.inp_info = inp_info
        self.out_info = out_info

    def set_profiler(self, profiler: Optional[trt.IProfiler]):
        self.context.profiler = profiler \
            if profiler is not None else trt.Profiler()

    def set_desired(self, desired: Optional[Union[List, Tuple]]):
        if isinstance(desired,
                      (list, tuple)) and len(desired) == self.num_outputs:
            self.idx = [self.output_names.index(i) for i in desired]

    def forward(self, *inputs) -> Union[Tuple, torch.Tensor]:

        assert len(inputs) == self.num_inputs
        contiguous_inputs: List[torch.Tensor] = [
            i.contiguous() for i in inputs
        ]

        for i in range(self.num_inputs):
            self.bindings[i] = contiguous_inputs[i].data_ptr()
            if self.idynamic:
                self.context.set_binding_shape(
                    i, tuple(contiguous_inputs[i].shape))

        outputs: List[torch.Tensor] = []

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.odynamic:
                shape = tuple(self.context.get_binding_shape(j))
                output = torch.empty(size=shape,
                                     dtype=self.out_info[i].dtype,
                                     device=self.device)
            else:
                output = self.output_tensor[i]
            self.bindings[j] = output.data_ptr()
            outputs.append(output)

        self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)
        self.stream.synchronize()

        return tuple(outputs[i]
                     for i in self.idx) if len(outputs) > 1 else outputs[0]

# detection model classes
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# colors for per classes
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}

def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im

def det_postprocess(data: Tuple[ndarray, ndarray, ndarray, ndarray]):
    assert len(data) == 4
    num_dets, bboxes, scores, labels = (i[0] for i in data)
    nums = num_dets.item()
    bboxes = bboxes[:nums]
    scores = scores[:nums]
    labels = labels[:nums]
    return bboxes, scores, labels

def yolo8_trt_test():
    #https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/infer-det.py
    device = torch.device('cuda:0')
    w = '/home/jetcam/yolo/_y8_TRT_Hamilton269/yolov8s-custom.trt'
    test_img = "/home/jetcam/tensorrt_hello/jetson-inference/data/images/humans_5.jpg"
    Engine = TRTModule(w, device)
    H, W = Engine.inp_info[0].shape[-2:]
     # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
    bgr = cv2.imread(str(test_img))
    draw = bgr.copy()
    bgr, ratio, dwdh = letterbox(bgr, (W, H))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
    tensor = torch.asarray(tensor, device=device)
    # inference
    data = Engine(tensor)
    bboxes, scores, labels = det_postprocess(data)
    bboxes -= dwdh
    bboxes /= ratio
    print("---results in---")
    for (bbox, score, label) in zip(bboxes, scores, labels):
        print("---result---")
        bbox = bbox.round().int().tolist()
        cls_id = int(label)
        cls = CLASSES[cls_id]
        color = COLORS[cls]
        print("bbox" , bbox)
        print("cls_id" , cls_id)
        # cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
        # cv2.putText(draw,
        #             f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.75, [225, 255, 255],
        #             thickness=2)

def remote_inference_yolo8_trt_cpu_img():
    device = torch.device('cuda:0')
    w = '/home/jetcam/yolo/_y8_TRT_off_shelf/yolov8n-fp16.trt'
    test_img = "/home/jetcam/tensorrt_hello/jetson-inference/data/images/humans_5.jpg"
    Engine = TRTModule(w, device)
    H, W = Engine.inp_info[0].shape[-2:]
     # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])




    mssger = rabbit_mq.MessengerBasic(
    factory.TZAR_config())
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
            bgr = msgs.decode_image_from_str(img_as_str)
            draw = bgr.copy()
            bgr, ratio, dwdh = letterbox(bgr, (W, H))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
            tensor = torch.asarray(tensor, device=device)
            # inference
            data = Engine(tensor)
            bboxes, scores, labels = det_postprocess(data)
            bboxes -= dwdh
            bboxes /= ratio
            print("---results in---")
            for (bbox, score, label) in zip(bboxes, scores, labels):
                print("---result---")
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = CLASSES[cls_id]
                color = COLORS[cls]
                print("bbox" , bbox)
                print("cls_id" , cls_id)
                # cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
                # cv2.putText(draw,
                #             f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.75, [225, 255, 255],
                #             thickness=2)





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

    remote_inference_yolo8_trt_cpu_img()

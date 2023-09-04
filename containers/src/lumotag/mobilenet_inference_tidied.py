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

import cv2
import time
import math
import vpi
import numpy as np
from contextlib import contextmanager
import math
from jetson_inference import detectNet
import jetson_utils
import threading
import queue
import copy
import json
from datetime import datetime
import subprocess
import argparse

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

def detector_cuda(inbox, outbox, ID):
    # this is default ssd - example of how we will load
    net = detectNet(model="/home/jetcam/mb1025_voc1501/ssd-mobilenet.onnx",
                input_blob="input_0",
                output_cvg="scores",
                output_bbox="boxes", 
                threshold=0.1)
    # net = detectNet(
    #     "ssd-mobilenet-v2",
    #     threshold=0.3)
    
    cuda_buff = None
    while True:
        if inbox.empty() is False:
            # blocking call here to see how long it takes to 
            # pop image off queue
            with time_it(f"{ID}: get object off queue"):
                cuda_obj = inbox.get(block=True)
            #image type is 'jetson.utils.cudaImage'

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

                with time_it(f"{ID} detectnet"):
                    detections = net.Detect(cuda_buff)

                with time_it(f"{ID}: feedback dects"):
                    all_dects = {}
                    dectdeets = None

                    # output is <class 'jetson.inference.detectNet.Detection'>
                    # single object is <detectNet.Detection object>
                    #{'ClassID': 2, 'Left': 555.9375, 'Top': 181.142578125,
                    # 'Right': 759.375, 'Bottom': 324.580078125,
                    # 'Confidence': 0.168701171875, 'index': '21'}
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

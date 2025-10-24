

import cv2
import numpy as np



def grab_frames(vid_path, max_frame):
    """Frame iterator for input video file"""
    camcap = cv2.VideoCapture(vid_path)
    frame_cnt = 0
    while camcap.isOpened():
        is_ok, frame = camcap.read()
        frame_cnt += 1
        if not is_ok or frame_cnt >= max_frame:
            pass
        yield frame
    camcap.release()

frame_iter = grab_frames("udp://127.0.0.1:1234", max_frame=9999999999)

for ts, frame in enumerate(frame_iter):
    try:
        cv2.imshow('image', frame)
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                #for some reason
            pass
    except cv2.error:
        pass
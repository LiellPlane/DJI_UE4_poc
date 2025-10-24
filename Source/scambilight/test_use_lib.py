import async_cam_lib
import cv2
from time import perf_counter
from contextlib import contextmanager
import time
import random

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

def main():
    image_capture2 = async_cam_lib.Scamblight_Camera_Async(async_cam_lib.ScambiLight_Cam_vidmodes)

    weewee = 0
    while True:
        with time_it("get image"):
            fart = next(image_capture2)
        #lumo_viewer(fart,0,0,0,False,False)
        print("final output", fart.shape, weewee)
        cv2.imwrite(f"/home/scambilight/0{weewee}.jpg", fart)
        if random.randint(1,10) < 5:
            time.sleep(random.random())
        if random.randint(1,10) < 5:
            plop = fart.copy()
        weewee = weewee + 1
        
if __name__ == '__main__':
    main()

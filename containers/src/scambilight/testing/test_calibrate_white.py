import sys

import cv2
import numpy as np
import time
import math
import numpy as np


def ImageViewer_Quick_no_resize(winname, inputimage,pausetime_Secs=0,presskey=False,destroyWindow=True):
    if inputimage is None:
        print("input image is empty")
        return
    ###handy quick function to view images with keypress escape andmore options
    cv2.imshow(winname, inputimage.copy()); 


    if presskey==True:
        cv2.waitKey(0); #any key
   
    if presskey==False:
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                #for some reason
                pass
            
    if pausetime_Secs>0:
        time.sleep(pausetime_Secs)
    if destroyWindow==True: cv2.destroyAllWindows()




def main1():
    fudge_factor = 2
    img2 = cv2.imread(r"C:\Working\nonwork\SCAMBILIGHT\fisheye_taped_colour_correction.png")
    ImageViewer_Quick_no_resize("img_corrected", img2,0,False,False)
    ravel_calib_img = img2.ravel()
    ravel_calib_img = (255 - img2) /fudge_factor
    calib_img = ravel_calib_img.reshape(img2.shape)
    ImageViewer_Quick_no_resize("img_corrected", calib_img,0,False,False)
    cap = cv2.VideoCapture(r"C:\Working\nonwork\SCAMBILIGHT\fisheye_taped.mp4")
    while True:
        suc, prev = cap.read()
        while suc is False:
            print("failed to grab image")
            time.sleep(0.05)
            suc, prev = cap.read()
        #prev = check_img2
        ImageViewer_Quick_no_resize("img_corrected", prev,0,True,False)
        #print("fart")
        # ravel_img = prev.ravel()
        # ravel_img = ravel_img.astype('float64')
        # ravel_corrected = ravel_img + calib_img.ravel()
        # ravel_corrected = np.clip(ravel_corrected, 0, 255)
        # ravel_corrected = ravel_corrected.astype('uint8')
        # img4 = ravel_corrected.reshape(img2.shape)

        img4 = prev.astype('float64') + calib_img
        img4 = np.clip(img4, 0, 255)
        img4 = img4.astype('uint8')
        ImageViewer_Quick_no_resize("img_corrected", img4,0,True,False)
if __name__ == "__main__":
    main1()
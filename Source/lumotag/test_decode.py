import decode_clothID_v2 as decode_clothID
import os
import sys
import cv2
import time
def ImageViewer_Quickv2(inputimage,pausetime_Secs=0,presskey=False,destroyWindow=True):
    ###handy quick function to view images with keypress escape andmore options
    CopyOfImage=cv2.resize(inputimage.copy(),(800,800))
    cv2.imshow("img", CopyOfImage); 
    if presskey==True:
        cv2.waitKey(0); #any key
    if presskey==False:
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                #for some reason
                pass
    if pausetime_Secs>0:
        time.sleep(pausetime_Secs)
    if destroyWindow==True: cv2.destroyAllWindows()
def read_img(img_filepath):
    return cv2.imread(img_filepath)
workingdata = decode_clothID.WorkingData()

workingdata.debug= True

input_imgs = decode_clothID.GetAllFilesInFolder_Recursive(r"D:\testshapes")
#input_imgs = decode_clothID.GetAllFilesInFolder_Recursive(r"D:\OutputImages")


print(f"{len(input_imgs)} images found")


def crop_in(img, pc_x, pc_y):
     height, width = img.shape
     new_height = (pc_y /100) * height
     new_width = (pc_x /100) * width
     crop_in_y = (height - new_height)/2
     crop_in_x = (width - new_width)/2
     #img is [width, height]
     return img[
          int(crop_in_y):int(new_height),
          int(crop_in_x):int(new_width)]


for img_filepath in input_imgs: 
    img = read_img(img_filepath)
    workingdata.debug_subfldr = img_filepath.split("\\")[-1].split(".jpg")[-2]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img = crop_in(img, pc_x=50, pc_y=50)
    print("_________")
    print(f"{img_filepath} {img.shape}")
    arse, playerfound = decode_clothID.find_lumotag(img, workingdata)
    if arse is not None: 
        ImageViewer_Quickv2(arse,0,False,True)

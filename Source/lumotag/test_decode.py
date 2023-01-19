import decode_clothID
import os
import sys
import cv2
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

workingdata.debug= False

input_imgs = decode_clothID.GetAllFilesInFolder_Recursive(r"C:\Working\nonwork\lumotag\live_tests")

print(f"{len(input_imgs)} images found")

for img_filepath in input_imgs:
    img = read_img(img_filepath)
    workingdata.debug_subfldr = img_filepath.split("\\")[-1].split(".jpg")[-2]
    arse, playerfound = decode_clothID.find_lumotag(img, workingdata)
    print(playerfound)
    ImageViewer_Quickv2(arse,0,False,True)
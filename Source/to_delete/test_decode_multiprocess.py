from multiprocessing import Process, Queue, shared_memory
from subprocess import Popen, PIPE
from heapq import heappush, heappop
from functools import reduce
import decode_clothID
import os
import sys
import cv2
import time

def multiprocess_decode_wrapper(
        myqueue: Queue,
        shared_mem_object: shared_memory.SharedMemory):
    
    pass


class SharedMemory():
    def __init__(self, obj_bytesize: int,
                 discrete_ids: list
                 ):
        """Memory which can be shared between processes.

            obj_bytesize: expected size of payload

            discrete_ids: for each element create a
            shared memory object and associated with ID"""
        self._bytesize = obj_bytesize
        self.mem_ids = {}
        for my_id in discrete_ids:
            try:
                self.mem_ids[my_id] = (shared_memory.SharedMemory(
                    create=True,
                    size=obj_bytesize,
                    name=my_id))
            except FileExistsError:
                print("Warning: shared memory {my_id} has not been cleaned up")
                self.mem_ids[my_id] = (shared_memory.SharedMemory(
                    create=False,
                    size=obj_bytesize,
                    name=my_id))



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

def main():


    img_dims = (640, 480)

    cams = ["HQ_full_frame", "HQ_centroid"]

    
    product_dims = reduce(lambda acc, curr: acc * curr, img_dims)

    shared_mem_handler = SharedMemory(
                            obj_bytesize=product_dims,
                            discrete_ids=["1", "2"]
                                            )


    workingdata = decode_clothID.WorkingData()

    workingdata.debug= True

    input_imgs = decode_clothID.GetAllFilesInFolder_Recursive(r"D:\testshapes")


    print(f"{len(input_imgs)} images found")

    for img_filepath in input_imgs: 
        img = read_img(img_filepath)
        workingdata.debug_subfldr = img_filepath.split("\\")[-1].split(".jpg")[-2]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, img_dims)  # BGR
        arse, playerfound = decode_clothID.find_lumotag(img, workingdata)
        print("shape", arse.shape, playerfound)
        ImageViewer_Quickv2(arse,0,False,True)
    
if __name__ == '__main__':
    main()
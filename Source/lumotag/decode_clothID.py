import sys
import cv2
import shutil
from enum import Enum, auto
import os
import numpy as np
sys.path.append(r"C:\Working\GIT\TestLab\TestLab")
#from matplotlib import pyplot as plt
import math
import statistics
import random
import time
from contextlib import contextmanager
from typing import Iterator

@contextmanager
def time_it() -> Iterator[None]:
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        toc: float = time.perf_counter()
        print(f"Computation time = {1000*(toc - tic):.3f}ms")

def GetAllFilesInFolder_Recursive(root):
    ListOfFiles=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            FullpathOfFile=(os.path.join(path, name))
            ListOfFiles.append(FullpathOfFile)
    return ListOfFiles

def DeleteFiles_RecreateFolder(FolderPath):
    Deltree(FolderPath)
    os.mkdir(FolderPath)

def Deltree(Folderpath):
      # check if folder exists
    if len(Folderpath)<6:
        raise("Input:" + str(Folderpath),"too short - danger")
        raise ValueError("Deltree error - path too short warning might be root!")
        return
    if os.path.exists(Folderpath):
         # remove if exists
         shutil.rmtree(Folderpath)
    else:
         # throw your exception to handle this special scenario
         #raise Exception("Unknown Error trying to Deltree: " + Folderpath)
         pass
    return

class AutoStrEnum(str, Enum):
    """
    StrEnum where auto() returns the field name.
    See https://docs.python.org/3.9/library/enum.html#using-automatic-values
    """
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name

class Debug_Images(AutoStrEnum):
    ERROR_no_contours = auto()
    GOOD_CANDIDATE_ContourCount = auto()
    POSITIVE_ID_ContourCount = auto()
    unfiltered_contours = auto()
    Filtered_area_contours = auto()
    filtered_circularity_contours = auto()
    original_input = auto()
    clahe_equalisation = auto()
    initial_thresh = auto()
    input_to_contours = auto()
    macro_candidates = auto()
    all_contours = auto()
    fitered_contours = auto()
    ID_BADGE = auto()

class WorkingData():
    def __init__(self) -> None:
        #self.input_imgs = r"C:\Working\nonwork\lumotag\Patterns\_001\Render2"
        self.input_imgs = r"C:\Working\nonwork\lumotag\test_outside_images"
        #self.input_imgs = r"C:\Working\nonwork\lumotag\temp_imgs_test"
        self.debugimgs = r"D:\lumodebug"
        self.debug = True
        self.debug_img_cnt = 0
        self.debug_subfldr = None
        if self.debug is True:
            DeleteFiles_RecreateFolder(self.debugimgs)
        self.claheprocessor = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
    @staticmethod
    def get_blob_params():
        DefaultBlobParams= cv2.SimpleBlobDetector_Params()
        DefaultBlobParams.filterByArea = True
        DefaultBlobParams.minArea = 40
        DefaultBlobParams.maxArea = 30000
        DefaultBlobParams.minDistBetweenBlobs = 2
        DefaultBlobParams.filterByColor = False
        DefaultBlobParams.filterByConvexity = False
        DefaultBlobParams.minCircularity = 0.5
        DefaultBlobParams.filterByInertia = False
        DefaultBlobParams.minInertiaRatio = 0.4
        return DefaultBlobParams

    def img_view_or_save_if_debug(self, img, description, resize = True):
        
        if self.debug is True:
            out_img = img.copy()
            if resize is True:
                resize_x =  int(img.shape[1]*(1000/img.shape[1]))
                resize_y =  int(img.shape[0]*(1000/img.shape[1]))
                out_img = cv2.resize(out_img, (resize_x,resize_y), interpolation = cv2.INTER_AREA)
            if self.debug_subfldr is None:
                filename = f"{self.debugimgs}\\0{self.debug_img_cnt}_{description}.jpg"
            else:
                if not os.path.exists(f"{self.debugimgs}\\{self.debug_subfldr}"):
                    os.mkdir(f"{self.debugimgs}\\{self.debug_subfldr}")
                filename = f"{self.debugimgs}\\{self.debug_subfldr}\\0{self.debug_img_cnt}_{description}.jpg"
            cv2.imwrite(filename,out_img)
            print(f"DEBUG = TRUE: saving debug file to {filename}")
            self.debug_img_cnt += 1

def PlotAndSave(Title,Filepath,Data,maximumvalue):
    #this causes crashes
    #save out plot of 1D data
    try:
        plt.plot(Data)
        plt.ylabel(Title)
        plt.ylim([0, max(Data)])
        plt.savefig(Filepath)
        plt.cla()
        plt.clf()
        plt.close()
    except Exception as e:
        print("Error with matpyplot",e)

def read_img(img_filepath):
    return cv2.imread(img_filepath)
def clahe_equalisation(img, claheprocessor):
    #luminosity
    lab_image=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab_image)
    #equ = cv2.equalizeHist(l)
    #updated_lab_img1=cv2.merge((equ,a,b))
    if claheprocessor is None:
        claheprocessor = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
    clahe_img= claheprocessor.apply(l)
    updated_lab_img1=cv2.merge((clahe_img,a,b))
    CLAHE_img = cv2.cvtColor(updated_lab_img1,cv2.COLOR_LAB2BGR)
    return CLAHE_img
def mono_img(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
def invert_img(img):
    return np.invert(img)
def equalise_img(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)
def blur_img(img, filtersize = 7):
    return cv2.GaussianBlur(img,(7,7),0)
def blur_average(img, filtersize = 7):
    kernel = np.ones((filtersize,filtersize),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    return dst
def normalise(img):
    image2_Norm = cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
    return image2_Norm
def threshold_img(img, low=0, high=255):
    #_ , th3 = cv2.threshold(img, low, 255,cv2.THRESH_BINARY)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,1)
    #_,th3 = cv2.threshold(img,low,high,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th3 = cv2.adaptiveThreshold(img,high,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return th3
def threshold_img_static(img, low=0, high=255):
    #_ , th3 = cv2.threshold(img, low, 255,cv2.THRESH_BINARY)
    #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,1)
    _,th3 = cv2.threshold(img,low,high,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th3 = cv2.adaptiveThreshold(img,high,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return th3
def edge_img(gray):
    #edges = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
    #edges = cv2.Canny(image=img, threshold1=100, threshold2=200)
    # Smoothing without removing edges.
    gray_filtered = cv2.bilateralFilter(gray, 5, 25, 25)

    # Applying the canny filter
    edges = cv2.Canny(gray, 60, 120)
    edges_filtered = cv2.Canny(gray_filtered, 0, 60)

    # Stacking the images to print them together for comparison
    #images = np.hstack((gray, edges, edges_filtered))
    return edges_filtered
def get_hist(img):
    #fig = plt.figure()
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #plt.hist(img.ravel(),256,[0,256]); plt.show()
    # plt.plot(hist)
    # plt.ylabel("histogram")
    # plt.ylim([0, max(hist)])
    # graph_image = np.array(fig.canvas.get_renderer()._renderer)
    # plt.cla()
    # plt.clf()
    # plt.close()
    
    # graph_image = np.array(fig.canvas.get_renderer()._renderer)
    return hist
def cut_square(img):
    length = 100 
    center_x = int(img.shape[0]/2)
    center_y = int(img.shape[1]/2)
    top = center_y - length
    lower = center_y + length
    left = center_x - length
    right = center_x + length
    cut = img[left:right,top:lower,:]
    return cut
def contours_img(img):
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    cv2.drawContours(out, contours, -1, 255,1)
    #cv2.drawContours(image=out, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    return out

def check_ID_contours_match_spec(hierarchy, circularities):
    #print(hierarchy)
    temp_hier=[]
    min_circularity = 0.75
    min_no_circles = 1
    for _index  in range (len(hierarchy[0])):
        if hierarchy[0][_index][2] == -1: # cont has no children
            if hierarchy[0][_index][3] != -1: # cont parent is not top level
                temp_hier.append(hierarchy[0][_index])
    for _index  in range (len(hierarchy[0])):
        if hierarchy[0][_index][2] == -1: #contour with no children
            # check if we have a group of 4 contours with
            # no children
            # get current position and following 3
            child_block = hierarchy[0][_index:_index+4,2]
            parent_block = hierarchy[0][_index:_index+4,3]
            # NB "set" removes duplicates
            if all([len(set(child_block)) == 1,
                    len(child_block) == 4,
                    len(set(parent_block)) == 1,
                    len([i for i in circularities[_index:_index+4] if i > min_circularity])>min_no_circles]):
                return True
            else:
                # jump to next block rather than reprocess
                _index = _index + 4
                continue
    return False

def decode_ID_image(img,dataobject : WorkingData):
    """provide single ID, thresholded binary image
    cv2.RETR_TREE will return a hierachy where we can find a parent
    contour with N children. 
    The following two outputs are from valid IDs with 4 elements in 
    a block. Each internal element has no children so the 3rd
    value should be -1, and each will have the same parent. Thereby
    the rule may be that if there are 4 instances in the 4th column of
    the parent contour, and each child has no other children (-1 in 3rd
    column), then we can mark this ID as a candidate
    cv2.findContours hierachy output
    ex 1
    [[[ 5 -1  1 -1]
    [ 2 -1 -1  0]
    [ 3  1 -1  0]
    [ 4  2 -1  0]
    [-1  3 -1  0]
    [-1  0 -1 -1]]]
    ex 2
    [[[ 1 -1 -1 -1]
    [ 2  0 -1 -1]
    [ 3  1 -1 -1]
    [-1  2  4 -1]
    [ 5 -1 -1  3]
    [ 6  4 -1  3]
    [ 7  5 -1  3]
    [-1  6 -1  3]]]
    
    index = contourid

    |next cnt in same tier|previous cnt in same tier|child|parent|"""
    
    # create ID badge
    id_badge = np.zeros((50,50,3), np.uint8)
    id_badge[:,:,1] = 255
    id_badge = cv2.putText(id_badge, "P3", (4,40), cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,0),thickness=2)
    id_badge = cv2.rotate(id_badge, cv2.ROTATE_90_CLOCKWISE)
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#RETR_EXTERNAL #RETR_TREE
     
    if contours is None or len(contours) == 0:
        
        #dataobject.img_view_or_save_if_debug(img,f"{Debug_Images.ERROR_no_contours.value}")
        return None, False
    #img_check_contours = img.copy()
    #img_check_contours = cv2.cvtColor(img_check_contours,cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(image=img_check_contours, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    contours_area = []


    #img_check_contours = img.copy()
    #img_check_contours = np.zeros_like(cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB))
    
    #for i, cnt in enumerate(contours):
    #    cv2.drawContours(image=img_check_contours, contours=[cnt], contourIdx=-1, color=(255,int(255/(i+1)),int(255/(i+1))), thickness=1, lineType=cv2.LINE_AA)

    #np.diff(np.where(has_children == -1))
    #print("checking check id input")
    #_3DVisLabLib.ImageViewer_Quick_no_resize(img,0,True,False)
    #print("cnts before anything filters", len(contours))
    # calculate area and filter into new array
    for con, hier in zip(contours,hierarchy[0]):
        area = cv2.contourArea(con)
        if 25 < area < 1000000:
            contours_area.append((con, hier))
    
    contours_cirles = []
    #print("cnts after area filter", len(contours_area))
    # check if contour is of circular shape
    circularities = []
    for con, hier in contours_area:
        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        if -9999 < circularity < 9999:
            contours_cirles.append((con, hier))
            circularities.append(circularity)
    #print("cnts after circle filter", len(contours_cirles))

    filtered_hierarchy = np.expand_dims(np.array([i[1] for i in contours_cirles]), axis=0)
    img_check_contours = None #LAZY CODE do this properly 
    if dataobject.debug is True:
        img_check_contours = img.copy()
        #img_check_contours = cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB)
        img_check_contours = np.zeros_like(cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB))
        
        #cv2.drawContours(image=img_check_contours, contours=[i[0] for i in contours_cirles], contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        
        for i, cnt in enumerate([i[0] for i in contours_cirles]):
            cv2.drawContours(image=img_check_contours, contours=[cnt], contourIdx=-1, color=(random.randint(50,255),random.randint(50,255),int(255/(i+1))), thickness=1, lineType=cv2.LINE_AA)
            #_3DVisLabLib.ImageViewer_Quick_no_resize(img_check_contours,0,True,False)
    #dataobject.img_view_or_save_if_debug(img_check_contours,f"Debug_Images.fitered_contours.value{len(contours_cirles)}")

    if len(contours_cirles) > 4 : # ID body and 4 internal markers:
        pass
        #dataobject.img_view_or_save_if_debug(img_check_contours,f"Debug_Images.GOOD_CANDIDATE_ContourCount.value{len(contours_cirles)}")
    else:
        return None, False

    if not check_ID_contours_match_spec(filtered_hierarchy, circularities):
        #_3DVisLabLib.ImageViewer_Quick_no_resize(img_check_contours,0,True,False)
        return None, False
    #_3DVisLabLib.ImageViewer_Quick_no_resize(img_check_contours,0,True,False)
    dataobject.img_view_or_save_if_debug(img_check_contours,f"POSITIVE_ID_ContourCount{len(contours_cirles)}")


    out = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    id_badge = cv2.resize(id_badge,(out.shape[1],out.shape[0]))
    #print(len(contours_cirles))

    cv2.drawContours(image=out, contours=[i[0] for i in contours_cirles], contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    #_3DVisLabLib.ImageViewer_Quick_no_resize(out,0,True,False)
    #cv2.drawContours(out, contours_cirles, , 255,1)
    return id_badge, True

def get_ID_bodies(img, dataobject : WorkingData):
    """provide thresholded image (might have to inverted to avoid segments
    on edge of image being classed as external), will filter contours for circularity"""
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    with time_it():
        print("get_ID_bodies:: contours")
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # print("unflitered contours")
    # chk_cnts = img.copy()
    # chk_cnts = cv2.cvtColor(chk_cnts, cv2.COLOR_GRAY2RGB)
    # chk_cnts = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    if dataobject.debug is True:
        img_check_contours = img.copy()
        #img_check_contours = cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB)
        img_check_contours = np.zeros_like(cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB))
        for i, cnt in enumerate([i for i in contours]):
            cv2.drawContours(image=img_check_contours, contours=[cnt], contourIdx=-1, color=(random.randint(20,255),random.randint(20,255),random.randint(20,255)), thickness=1, lineType=cv2.LINE_AA)
        #cv2.drawContours(image=shw_cnts, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        # _3DVisLabLib.ImageViewer_Quick_no_resize(chk_cnts,0,True,False)
        dataobject.img_view_or_save_if_debug(img_check_contours, Debug_Images.unfiltered_contours.value)

    contours_area = []
    # calculate area and filter into new array
    for con in contours:
        area = cv2.contourArea(con)
        if 400 < area < 1000000:
            contours_area.append(con)

    if dataobject.debug is True:
        img_check_contours = img.copy()
        #img_check_contours = cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB)
        img_check_contours = np.zeros_like(cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB))
        for i, cnt in enumerate([i for i in contours_area]):
            cv2.drawContours(image=img_check_contours, contours=[cnt], contourIdx=-1, color=(random.randint(20,255),random.randint(20,255),random.randint(20,255)), thickness=1, lineType=cv2.LINE_AA)
        #cv2.drawContours(image=shw_cnts, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        # _3DVisLabLib.ImageViewer_Quick_no_resize(chk_cnts,0,True,False)
        dataobject.img_view_or_save_if_debug(img_check_contours, Debug_Images.Filtered_area_contours.value)


    # print("contours filtered by area")
    # chk_cnts = img.copy()
    # chk_cnts = cv2.cvtColor(chk_cnts, cv2.COLOR_GRAY2RGB)
    # chk_cnts = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    # cv2.drawContours(image=chk_cnts, contours=contours_area, contourIdx=-1, color=(255, 0, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
    # _3DVisLabLib.ImageViewer_Quick_no_resize(chk_cnts,0,True,False)

    contours_cirles = []
    # check if contour is of circular shape
    for con in contours_area:
        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        #contours_cirles.append(con)
        #if 0.6 < circularity < 1.2:
        if 0.3 < circularity < 3.2:
            contours_cirles.append(con)
    
    if dataobject.debug is True:
        img_check_contours = img.copy()
        #img_check_contours = cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB)
        img_check_contours = np.zeros_like(cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB))
        for i, cnt in enumerate([i for i in contours_cirles]):
            cv2.drawContours(image=img_check_contours, contours=[cnt], contourIdx=-1, color=(random.randint(20,255),random.randint(20,255),random.randint(20,255)), thickness=1, lineType=cv2.LINE_AA)
        #cv2.drawContours(image=shw_cnts, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        # _3DVisLabLib.ImageViewer_Quick_no_resize(chk_cnts,0,True,False)
        dataobject.img_view_or_save_if_debug(img_check_contours, Debug_Images.filtered_circularity_contours.value)

    out = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    cv2.drawContours(image=out, contours=contours_cirles, contourIdx=-1, color=(0, 255, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        #cv2.drawContours(out, contours_cirles, , 255,1)

    #print("contours filtered by circularity")
    #chk_cnts = img.copy()
    #chk_cnts = cv2.cvtColor(chk_cnts, cv2.COLOR_GRAY2RGB)
    #chk_cnts = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    #cv2.drawContours(image=chk_cnts, contours=contours_cirles, contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
    #_3DVisLabLib.ImageViewer_Quick_no_resize(chk_cnts,0,True,False)


    return out, contours_cirles

def get_tiled_intensity(img, n_tiles_edge):
    """input mono image
    returns list(int), list(int)of max/min values per tile"""
    edge_len_pxls = int(img.shape[0]/n_tiles_edge)
    if edge_len_pxls <20:
        # for production use logging and send error, don't crash out
        raise ValueError("length not valid for lumo application")
    toprange=img.shape[0]-(img.shape[0]%edge_len_pxls) # ignore remainder (probably should centre it)
    siderange=img.shape[1]-(img.shape[1]%edge_len_pxls)
    print("----------------------------------")
    maxes=[]
    mins=[]
    for vert in range(0,toprange,edge_len_pxls):
        for horiz in range(0,siderange,edge_len_pxls):
            maxes.append(img[vert:vert+edge_len_pxls,horiz:horiz+edge_len_pxls].max())
            mins.append(img[vert:vert+edge_len_pxls,horiz:horiz+edge_len_pxls].min())

    # dont need to sort them 
    maxes.sort()
    mins.sort()
    return maxes, mins

def get_dist(sample: list[int]):
    std_d=np.std(sample)
    mean=np.mean(sample)
    return std_d, mean
    
def analyse_candidate_contours(original_img,
                                original_img_grayscale,
                                masked_img,
                                thresholded_img,
                                contours : tuple [np.ndarray],
                                dataobject : WorkingData):
    """supply monitoring image and image which has masked area of the irregular contours found
    for ID patches (not rectangular bounding boxes
    
    original_image = np array n/n/3 (colour image)
    masked_img = binary image
    contour"""
    playerfound = False
    #if masked_img.max()>1:
    #    raise ValueError("masked image input to analyse_candidate_contours not binary value")
    #if original_img.shape[2]!=3:
    #    raise ValueError("input image not colour")
    #kernel = np.ones((5, 5), np.uint8)
    #img_with_contours = original_img_grayscale.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # can collect the images here and get a nice animation
        #_3DVisLabLib.ImageViewer_Quickv2_UserControl(img_with_contours,0,True,False)
        #continue
        # pull out the masked bounding area - as will be irregular
        # use this to mask off the rectangular bounding box
        #mask_img_area = masked_img[y:y + h, x:x + w]


        
        #get original image and reprocess
        original_samp = original_img_grayscale[y:y + h, x:x + w].copy()
        #dataobject.img_view_or_save_if_debug(original_samp,"original_sample")
        #imgsize = original_samp.shape[0]*original_samp.shape[1]
        # if imgsize < 10000:
        #     original_samp = blur_img(original_samp,filtersize= 3)
        #     original_samp = cv2.bilateralFilter(original_samp,9,75,75)
        #     print("3")
        # elif imgsize < 100000:
        #     original_samp = blur_img(original_samp,filtersize= 3)
        #     print("5")
        # elif imgsize < 1000000:
        #     original_samp = blur_img(original_samp,filtersize= 5)
        #     print("7")
        # else:
        #     original_samp = blur_img(original_samp,filtersize= 7)
        #     print("9")
        #original_samp = cv2.bilateralFilter(original_samp,3,10,10)
        original_samp=normalise(original_samp)
        original_samp=threshold_img_static(original_samp, low = 127, high = 255)
        #dataobject.img_view_or_save_if_debug(original_samp,"equalise_static")
        #clahe_eq = clahe_equalisation(original_img[y:y + h, x:x + w].copy())
        #clahe_eq = cv2.cvtColor(clahe_eq,cv2.COLOR_BGR2GRAY)
        #clahe_eq=threshold_img_static(clahe_eq, low = 127, high = 255)
        #dataobject.img_view_or_save_if_debug(clahe_eq,Debug_Images.clahe_equalisation.value)
        #continue
        #print("test area")
        #continue
        #threshold_sample = thresholded_img[y:y + h, x:x + w]
        #_3DVisLabLib.ImageViewer_Quick_no_resize(threshold_sample,0,True,False)
        # need to erode mask so it falls inside the footprint of the element
        #mask_img_area = cv2.erode(mask_img_area, kernel, iterations=1)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(mask_img_area*255,0,True,False)
        #continue
        #isolated_thresholded_ID_candidate = thresholded_img[y:y + h, x:x + w]
        #_3DVisLabLib.ImageViewer_Quick_no_resize(isolated_thresholded_ID_candidate,0,True,False)
        #continue
        #masked_input=isolated_thresholded_ID_candidate*(cv2.cvtColor(mask_img_area,cv2.COLOR_BGR2GRAY))[:,:]
        #_3DVisLabLib.ImageViewer_Quick_no_resize(masked_input,0,True,False)
        #continue
        #masked_ID=threshold_img(masked_input,low=127) # watch the threshold here
        # maybe use another thresholding variant
        #_3DVisLabLib.ImageViewer_Quick_no_resize(masked_ID,0,True,False)
        #continue

        decoded_ID, playerfound= decode_ID_image(original_samp, dataobject)
        #if decoded_ID is not None:
        #    img_with_contours = cv2.rectangle(original_img, (x,y), (x+w,y+h), (0,0,255), 2)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(decoded_ID,0,True,False)
        #continue
        # draw ROIs back on image for illustration
        # this is in rectangles - so could cut off parts of IDs if very
        # deformed cloth
        if decoded_ID is not None:#
            #TODO why is the image from HQ camera 4 channels? 
            #original_img[y:y + h, x:x + w,0:3] = decoded_ID#cv2.cvtColor(decoded_ID,cv2.COLOR_BGR2GRAY)
            original_img = cv2.rectangle(
                    original_img,
                    (x, y),
                    (x + w, y + h),
                    (0,255, 255),
                    5
                    )
            dataobject.img_view_or_save_if_debug(original_img, Debug_Images.ID_BADGE.value)


    #_3DVisLabLib.ImageViewer_Quick_no_resize(img_with_contours,0,True,False)
    return original_img, playerfound

def cumulative_dist_histogram():
    plop

def dilate(InputImage):
    kernel = np.ones((3, 3), np.uint8)
    img_blur = cv2.medianBlur(InputImage, 3)
    dilated_image = cv2.dilate(img_blur, kernel, iterations = 1)
    #eroded_image = cv2.erode(dilated_image, kernel, iterations = 5)
    return dilated_image
def median_blur(inputimage, kernalsize):
    return  cv2.medianBlur(inputimage, kernalsize)


# Set up the blob detector.
#blob_detector = cv2.SimpleBlobDetector_create(workingdata.get_blob_params())


# img_proc_chain=[]
# img_proc_chain.append(cut_square)
# #img_proc_chain.append(mono_img)
# #img_proc_chain.append(equalise_img)
# #img_proc_chain.append(blur_img)
# #img_proc_chain.append(threshold_img)

def find_lumotag(inputimg, dataobject : WorkingData):

    """analyse input image for specific lumotag pattern"""
    #~2ms
    with time_it():
        print("grayscale")
    if len(inputimg.shape)>2:
        img_grayscale = cv2.cvtColor(inputimg,cv2.COLOR_BGR2GRAY)
    else:
        img_grayscale = inputimg
    dataobject.img_view_or_save_if_debug(inputimg, Debug_Images.original_input.value, resize=False)
    #copy original image into folder
    #orig_img = img.copy()
    
    #~20ms
    with time_it():
        print("equalisation")
        orig_img=clahe_equalisation(inputimg.copy(), dataobject.claheprocessor)
        dataobject.img_view_or_save_if_debug(orig_img, Debug_Images.clahe_equalisation.value)
        ''''test area'''
   
   #this section about 80ms
    with time_it():
        print("mono_img")
    gray_orig = mono_img(orig_img)
    with time_it():
        print("median_blur")
        #blurred = median_blur(gray_orig,7)
        blurred = cv2.blur(gray_orig,(7,7)) # fastest filter
        dataobject.img_view_or_save_if_debug(blurred, Debug_Images.initial_thresh.value)
        #edge_im = edge_img(blurred)
    with time_it():
        print("threshold_img")
        squr_img=threshold_img(blurred,low=127)
    with time_it():
        print("invert_img")
        squr_img=invert_img(squr_img)


    dataobject.img_view_or_save_if_debug(squr_img, Debug_Images.input_to_contours.value)
    with time_it():
        
        squr_img, contours=get_ID_bodies(squr_img, dataobject)
        print("get_ID_bodies total")
    dataobject.img_view_or_save_if_debug(squr_img, Debug_Images.macro_candidates.value)
    #squr_img_gray = cv2.cvtColor(squr_img,cv2.COLOR_BGR2GRAY)
    #with time_it():
    #    print("squr_img_mask")
    #squr_img_mask= cv2.cvtColor(np.clip(squr_img,0,1),cv2.COLOR_BGR2GRAY)
    with time_it():
        print("analyse_candidate_contours")
        analyse_IDs, playerfound = analyse_candidate_contours(original_img=inputimg.copy(),
                                                original_img_grayscale = img_grayscale,
                                                masked_img = None,
                                                thresholded_img= None,
                                                contours = contours,
                                                dataobject = dataobject)
    if analyse_IDs is not None:
        dataobject.img_view_or_save_if_debug(analyse_IDs, Debug_Images.ID_BADGE.value)
        return analyse_IDs,playerfound
    return inputimg, False
    
def test_live():

    workingdata = WorkingData()

    workingdata.debug= False

    input_imgs = GetAllFilesInFolder_Recursive(workingdata.input_imgs)

    print(f"{len(input_imgs)} images found")

    for img_filepath in input_imgs:
        img = read_img(img_filepath)
        workingdata.debug_subfldr = img_filepath.split("\\")[-1].split(".jpg")[-2]
        find_lumotag(img, workingdata)

def old_testing():

    workingdata = WorkingData()

    input_imgs = GetAllFilesInFolder_Recursive(workingdata.input_imgs)

    print(f"{len(input_imgs)} images found")

    for img_filepath in input_imgs:
        img = read_img(img_filepath)
        img_grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #set a debug image subfolder - if this is NONE then will just save all images sequentially in same level
        workingdata.debug_subfldr = img_filepath.split("\\")[-1].split(".jpg")[-2]
        workingdata.img_view_or_save_if_debug(img, Debug_Images.original_input.value, resize=False)
        #copy original image into folder
        #orig_img = img.copy()
        orig_img=clahe_equalisation(img.copy())
        workingdata.img_view_or_save_if_debug(orig_img, Debug_Images.clahe_equalisation.value)
        #square cut area
        #squr_img=orig_img.copy()#cut_square(orig_img)
        #squr_img=mono_img(orig_img)


        ''''test area'''
        gray_orig = mono_img(orig_img)
        
        #dilated = dilate (gray_orig)
        #blurred = blur_img(gray_orig,filtersize= 7)
        #blurred = blur_average(blurred,filtersize= 5)
        blurred = median_blur(gray_orig,7)
        workingdata.img_view_or_save_if_debug(blurred, Debug_Images.initial_thresh.value)
        #edge_im = edge_img(blurred)
        squr_img=threshold_img(blurred,low=127)
        
        squr_img=invert_img(squr_img)
        workingdata.img_view_or_save_if_debug(squr_img, Debug_Images.input_to_contours.value)
        squr_img, contours=get_ID_bodies(squr_img, workingdata)
        workingdata.img_view_or_save_if_debug(squr_img, Debug_Images.macro_candidates.value)
        squr_img_gray = cv2.cvtColor(squr_img,cv2.COLOR_BGR2GRAY)
        squr_img_mask= cv2.cvtColor(np.clip(squr_img,0,1),cv2.COLOR_BGR2GRAY)
        analyse_IDs = analyse_candidate_contours(original_img=img.copy(),
                                                original_img_grayscale = img_grayscale,
                                                masked_img = None,
                                                thresholded_img= None,
                                                contours = contours,
                                                dataobject = workingdata)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(squr_img,0,True,False)
        continue
        ''''fin test'''
        #hist = get_hist(squr_img)
        #tempimg="d:\plop.jpg"
        #PlotAndSave("plop",tempimg,hist,hist.max())
        #histogram_img=cv2.imread(tempimg)
        squr_img=threshold_img(squr_img,low=127)
        squr_img=invert_img(squr_img)
        threshold_original = squr_img.copy()
        #_3DVisLabLib.ImageViewer_Quick_no_resize(squr_img,0,True,False)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(squr_img,0,True,False)
        #continue
        squr_img, contours=get_ID_bodies(squr_img)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(squr_img,0,True,False)
        squr_img_mask= np.clip(squr_img,0,1)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(squr_img,0,True,False)
        #continue

        analyse_IDs = analyse_candidate_contours(original_img = orig_img,
                                                masked_img = squr_img_mask,
                                                thresholded_img= threshold_original,
                                                contours = contours)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(squr_img,0,True,False)
        continue
        #clip to 0 and 1 for masking
        squr_img_mask= np.clip(squr_img,0,1)
        # use mask on input colour image
        sample_area_masked=sample_area.copy()
        squr_img= sample_area_masked*squr_img_mask

        #masked_input_sqr=
        #
        #squr_img=edge_img(squr_img)
        #squr_img=blur_img(squr_img)
        thresh_low = squr_img.min()
        thresh_high= squr_img.max()




        orig_img_edge=img.copy()
        orig_img_edge=mono_img(orig_img_edge)
        #orig_img_edge=blur_img(orig_img_edge)
        orig_img_edge=edge_img(orig_img_edge)
        

        #original area threshold
        #img=blur_img(img)
        #img=threshold_img(img,low=thresh_low,high=thresh_high)

        #for img_proc in img_proc_chain:
        #    img = img_proc(img)
        # Detect blobs from the image.
        #keypoints = blob_detector.detect(original_img_bw)
        #contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(image=orig_img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS - This method draws detected blobs as red circles and ensures that the size of the circle corresponds to the size of the blob.
        #blobs = cv2.drawKeypoints(original_img_bw, keypoints, original_img_bw, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        quilt_image: np.ndarray = np.ndarray((
                orig_img.shape[0],
                orig_img.shape[1]*3,
                orig_img.shape[2],),
                dtype=('uint8'))
        quilt_image[0:orig_img.shape[0],0:orig_img.shape[1],:] = orig_img
        resized=cv2.resize(squr_img, tuple(reversed(orig_img.shape[0:2])), interpolation = cv2.INTER_AREA)  
        resized_histogram=cv2.resize(histogram_img, tuple(reversed(orig_img.shape[0:2])), interpolation = cv2.INTER_AREA) 
        try:
            quilt_image[0:orig_img.shape[0],orig_img.shape[1]:,0] = resized
            quilt_image[0:orig_img.shape[0],orig_img.shape[1]:,1] = resized
            quilt_image[0:orig_img.shape[0],orig_img.shape[1]:,2] = resized
        except:
            quilt_image[0:orig_img.shape[0],orig_img.shape[1]:orig_img.shape[1]*2,:] = resized

        quilt_image[0:orig_img.shape[0],orig_img.shape[1]*2:orig_img.shape[1]*3,:] = resized_histogram

        quilt_image[0:sample_area.shape[0],0:sample_area.shape[1],:] = sample_area
        

        quilt_image=cv2.resize(quilt_image, (int(quilt_image.shape[1]/2),int(quilt_image.shape[0]/2)), interpolation = cv2.INTER_AREA)  
        _3DVisLabLib.ImageViewer_Quick_no_resize(quilt_image,0,True,False)



"""Requirements.text
pip install numpy
pip install opencv-python
conda install -c conda-forge matplotlib

"""
import json
#from posix import POSIX_FADV_DONTNEED   
import re
import enum
import cv2
print ("CV verions: " + cv2.__version__)
import numpy as np
import os
import sys
import math
import pickle
import io
import inspect
import time
import shutil
import copy
import multiprocessing
from multiprocessing import Pool
import random
import logging
import collections
import glob
import json
#from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from collections import namedtuple
from subprocess import Popen, PIPE
from collections import OrderedDict


#import plyfile
#from plyfile import PlyData, PlyElement
plt.rcParams['figure.figsize'] = [15, 10]

#experiment with locking common used images during calibration sequence
#research how to use this I dont think this is recommended
#TODO
ImageFileLock=multiprocessing.Lock()

def ImageViewer_Quick(inputimage):
    ###handy quick function to view images with keypress escape
    CopyOfImage=cv2.resize(inputimage.copy(),(800,800))
    cv2.imshow("img", CopyOfImage); cv2.waitKey(0); cv2.destroyAllWindows()
    
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


    
def ImageViewer_Quick_no_resize(inputimage,pausetime_Secs=0,presskey=False,destroyWindow=True):
    if inputimage is None:
        print("input image is empty")
        return
    ###handy quick function to view images with keypress escape andmore options
    cv2.imshow("img", inputimage.copy()); 
    
    if presskey==True:
        cv2.waitKey(0); #any key
   
    if presskey==False:
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                #for some reason
                pass
            
    if pausetime_Secs>0:
        time.sleep(pausetime_Secs)
    if destroyWindow==True: cv2.destroyAllWindows()

def ImageViewer_Quickv2_UserControl(inputimage,pausetime_Secs=0,presskey=False,destroyWindow=True):
    ###handy quick function to view images with keypress escape andmore options
    if inputimage is None:
        return None
    CopyOfImage=cv2.resize(inputimage.copy(),(800,600))
    cv2.imshow("img", CopyOfImage); 
    UserRequest=""
    if presskey==True:
        while(1):
            cv2.imshow('img',CopyOfImage)
            k = cv2.waitKey(33)
            #return character from keyboard input ascii code
            if k != -1:#no input detected
                try:
                    UserRequest=(chr(k))
                    break
                except:
                    UserRequest=None
            else:
                continue
             
    if presskey==False:
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                #for some reason
                pass
    
    if pausetime_Secs>0:
        time.sleep(pausetime_Secs)
    if destroyWindow==True: cv2.destroyAllWindows()

    return UserRequest
    
def get_optimal_font_scale(text, width,Height):
    #find best fit for font scale - NOTE this can be improved by passing in font details
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        new_Height = textSize[0][1]
        if (new_width <= width) and (new_Height <= Height):
            return scale/10
    
def GetPositionOfWindow_Number(ProcessNumber):
    CarriageReturn=3#what window do we move onto next row
    YRow = math.floor(ProcessNumber/CarriageReturn)
    Xrow = ProcessNumber%CarriageReturn
    return(Xrow*400,YRow*600)
    
def GetPositionOfWindow(WindowName):
    print(WindowName)
    #python doesnt have native switch/case structures?!?!?!? JESUS CHRIST
    if WindowName=="pod1primary":
        return (0,0)
    if WindowName=="pod2primary":
        return (400,0)
    if WindowName=="pod3primary":
        return (800,0)
    if WindowName=="pod1secondary":
        return (0,600)
    if WindowName=="pod2secondary":
        return (400,600)
    if WindowName=="pod3secondary":
        return (800,600)
    return 500,500

class Paths_Common:
    
    ProjectPath='C:/UDemyOpenCV1/CUSTOM/'
    CALIBS_path=ProjectPath + 'CALIBS/'
    FAILS_path=ProjectPath + 'FAILS/'
    MVS_path=ProjectPath + 'MVS/'
    CorrectionDistortions_path=ProjectPath + 'CORRECTDISTORTION/'
    ParamLoop_path=ProjectPath + 'PARAMETER_LOOP/'
    PointClouds_path=ProjectPath + 'PointClouds/'
    AcquiredImages=ProjectPath+ "Calibs_Subjects/"
class NamedTuples_Common:
    FaceDetectorReturns = namedtuple('FaceDetectorReturns', ['FaceCount','OriginalImage', 'ImageWithFaceBorders', 'ListOfFaceBorderCorners'])
    BoundingBox=namedtuple('BoundingBox',['StartX', 'StartY','EndX','EndY'])
    FaceLandMarkReturns= namedtuple('FaceLandMarkReturns', ['FaceCount','OriginalImage', 'ImageWithLandMarks', 'ListOfFaceLandMarks'])
    FacePoseReturns=namedtuple('FacePoseReturns',['FaceCount','OriginalImage','ImageWithPose','RotVector','TransVector','CameraDetails','XY2DPositionNose',"Notes"])
    ObjectTracking_ImagesFeatures= namedtuple('ObjectTrackingFeatures', ['FileName','RawImages','ImageWithFeaturess', 'Keypoints', 'Descriptors'])

    #experimental
    FaceStabilisation= namedtuple('FaceStabilisation', ['FaceLandMarkA','FaceLandMarkB','OutputImageSizeH', 'OutputImageSizeW', 'FaceMetric_A','FaceMetric_B'])
   
    
class Dict_Common:
    
    SIFT_default=dict(nfeatures=0,nOctaveLayers=3,contrastThreshold=0.04,edgeThreshold=10)
    
    ORB_default=dict(nfeatures=20000,scaleFactor=1.3,
                       nlevels=2,edgeThreshold=0,
                       firstLevel=0, WTA_K=2,
                       scoreType=0,patchSize=155)
    
    SIFT_Testing=dict(nfeatures=50,contrastThreshold=0.04,edgeThreshold=10)
    
    ORB_Testing=dict(nfeatures=20000,scaleFactor=1.02,
                       nlevels=4,edgeThreshold=0,
                       firstLevel=4, WTA_K=2,
                       scoreType=0,patchSize=100)
    
   

    # define a dictionary that maps the indexes of the facial
    # landmarks to specific face regions
    FACIAL_LANDMARKS_IDXS = OrderedDict([
    	("mouth", (48, 68)),
    	("right_eyebrow", (17, 22)),
    	("left_eyebrow", (22, 27)),
    	("right_eye", (36, 42)),
    	("left_eye", (42, 48)),
    	("nose", (27, 36)),
    	("jaw", (0, 17))
    ])


#COMMON VARIABLES GO HERE
class Str_Common:#not immutable TODO
    
    

    FError_VGood=0.005#Epipolar constraint error
    FError_Acceptable=0.005#Epipolar constraint error
    FError_Failure=0.07#Epipolar constraint error
    Green=(0,255,0)#GBR
    Orange=(0,165,255)#GBR
    Red=(0,0,255)#GBR
    EMPTY="EMPTY"
    SIFT="sift"
    ORB="orb"
    MultiProcessTIMEOUTerr="Time out Error - is process run as %run xxx.py??"
    NCC_TM_SQDIFF_NORMED="TM_SQDIFF_NORMED"
    NCC_TM_CCOEFF_NORMED="TM_CCOEFF_NORMED"
    DenseFailsCode_OK=1
    DenseFailsCode_ResSize=2
    DenseFailsCode_Except=3
    DenseFailsCode_NoThreshold=4
    DenseFailsCode_NoNorming=5
    DenseFailsCode_NoImages=6
    NoOfFaces="NoOHCF"#
    chars_NonNumeric = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*,./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
    chars_NonNumeric_Floating = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*,./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
    
    MeshLabPickPts_point='point'
    MeshLabPickPts_Name='name'
    MeshLabPickPts_active1='active="1"'
    MeshLabPickPts_x='x'
    MeshLabPickPts_y='y'
    MeshLabPickPts_z='z'

    # Using enum class create enumerations
    class Softwares3D(enum.Enum):
        UnrealEngine="Unreal Engine"
        Maya="Maya"
        OpenCV="OpenCV"
        Blender="Blender"

def clamp(n, min_n=-999999, max_n=999999):
    if n < min_n:
        return min_n
    elif n > max_n:
        return max_n
    else:
        return n

def yesno(question):
    """Simple Yes/No Function."""
    prompt = f'{question} ? (y/n): '
    ans = input(prompt).strip().lower()
    if ans not in ['y', 'n']:
        print(f'{ans} is invalid, please try again...')
        return yesno(question)
    if ans == 'y':
        return True
    return False


class FaceDetector_type(enum.Enum):
    DLIB="DLIB"
    OpenCV="OpenCV"
    TensorFlow="TensorFlow"
    @classmethod
    def from_str(cls,label): 
        #check if any substrings are in test string to return enum
        if any(x in label.lower() for x in ["tensor","flow","tensorflow","tf"]):
            return True, cls.TensorFlow
        if any(x in label.lower() for x in ["opencv","ocv"]):
            return True, cls.OpenCV
        if any(x in label.lower() for x in ["dlib"]):
            return True, cls.DLIB
        return False, "FAILURE TO PARSE"

class FaceLandMarkDetector_type(enum.Enum):
    DLIB="DLIB"
    TensorFlow="TensorFlow"
    @classmethod
    def from_str(cls,label):
        #check if any substrings are in test string to return enum
        if any(x in label.lower() for x in ["tensor","flow","tensorflow","tf"]):
            return True, cls.TensorFlow
        if any(x in label.lower() for x in ["dlib"]):
            return True, cls.DLIB
        return False, "FAILURE TO PARSE"

class PrinciplePoint_state(enum.Enum):
    TRUE="TRUE"
    FALSE="FALSE"
    @classmethod
    def from_str(cls,label):
        #check if any substrings are in test string to return enum
        if any(x in label.lower() for x in ["true"]):
            return True, cls.TRUE
        if any(x in label.lower() for x in ["false"]):
            return True, cls.FALSE
        return False, "FAILURE TO PARSE"


class Numbers_Common:
    DegreePerRadian=57.29577951 #(180/pi)
    
def DefaultBlobParams():
    #DOT GRID CALIBRATION PATTERN blob tool parameters that seem to work well for now
    DefaultBlobParams= cv2.SimpleBlobDetector_Params()
    DefaultBlobParams.filterByArea = True
    
    #DefaultBlobParams.minArea = 400#small size
    DefaultBlobParams.minArea = 4000#big
    #DefaultBlobParams.maxArea = 3000#small
    DefaultBlobParams.maxArea = 30000#big
    DefaultBlobParams.minDistBetweenBlobs = 20
    DefaultBlobParams.filterByColor = True
    DefaultBlobParams.filterByConvexity = False
    DefaultBlobParams.minCircularity = 0.7
    DefaultBlobParams.filterByInertia = True
    DefaultBlobParams.minInertiaRatio = 0.4
    return DefaultBlobParams

class CalibrateCameras_InputArgs:
    FileOriginCamera=""
    FilePairedCamera=""
    SessionString=""
    MaxFailedRetries=3
    DebugImages=False
    ProcessNumber=0
    ProcessName=Str_Common.EMPTY
    
     
class SparseReconstruct_InputArgs:
     MODIFIED_Param_DictionarySIFT=[]
     MODIFIED_Param_DictionaryORB=[]
     CalibrationFile=""
     SubjectSession_CommonPrefix=""
     FeatureTypeToUse=""
     SaveString=""
     ProcessNumber=0
     ProcessName=Str_Common.EMPTY
     LiveImageWait=False
     InteralCounter1=0
     DenseNCC_ChannelWidth=14
     DenseNCC_TemplateSize=6
     DenseNCC_AcceptThreshold=0.95
     DenseNCC_ResultType=Str_Common.NCC_TM_SQDIFF_NORMED
     Dense_DivideDensity=1
     Dense_AutoF_CameraA_Name=Str_Common.EMPTY
     Dense_AutoF_CameraB_Name=Str_Common.EMPTY
     
class USER_OPTIONS:
    FeatureMatcher=""
    MultiProcess_Max=8
    ParallelProcess=True
    TimeOutMinutes=10
    AcquisitionSet=""
    CommonPrefix=""
    AdaptionLoops=2
    DebugImages=False
    LiveImagesWait=True
    InternalCounter1=0
    DenseNCC_ChannelWidth=14
    DenseNCC_TemplateSize=6
    DenseNCC_AcceptThreshold=0.95
    DenseNCC_ResultType=Str_Common.NCC_TM_SQDIFF_NORMED
    Dense_DivideDensity=1
    
    Dense_CalibrationFile=Str_Common.EMPTY
    
    
def display(text,img,cmap='gray'):
    fig = plt.figure(figsize=(15,10 ))
    ax = fig.add_subplot(111)
    plt.title(text)
    ax.imshow(img,cmap='gray')
    
def draw_Keypoints_NP_array(InputImage, LandmarksNParray, color = (255, 255, 0),size=2):
    """draw landmarks on an input image from an NP array, returns modified image and input image"""
    WorkingImage=InputImage.copy()
    for kp in LandmarksNParray:
            x = int(kp[0,0])
            y = int(kp[0,1])
            cv2.circle(WorkingImage, (x, y), int(size), color,2)
    #debug - draw center of image

    return WorkingImage

def draw_Centre_of_image(InputImage, color = (0, 0, 255),size=20):
    """draw point in centre of image"""
    WorkingImage=InputImage.copy()
    cv2.circle(WorkingImage, (int(WorkingImage.shape[1]/2), int(WorkingImage.shape[0]/2)), int(size), color,2)
    cv2.circle(WorkingImage, (int(WorkingImage.shape[1]/2), int(WorkingImage.shape[0]/2)), int(size*2), color,2)
    cv2.circle(WorkingImage, (int(WorkingImage.shape[1]/2), int(WorkingImage.shape[0]/2)), int(size/2), color,2)
    return WorkingImage

def draw_keypoints(vis, keypoints, color = (255, 0, 0)):
    #v1
    #this function in cv2 keeps breaking
    #we need to make our own
    
    vis=cv2.cvtColor(vis,cv2.COLOR_GRAY2BGR)
    for kp in keypoints:
            x, y = kp.pt
            size=kp.size
            cv2.circle(vis, (int(x), int(y)), int(2), color,2)
    return vis

def draw_keypoints_v2(vis, keypoints, color = (255, 0, 0)):
    #v1
    #this function in cv2 keeps breaking
    #we need to make our own
    
    #vis=cv2.cvtColor(vis,cv2.COLOR_GRAY2BGR)
    for kp in keypoints:
            x, y = kp.pt
            size=kp.size
            cv2.circle(vis, (int(x), int(y)), int(2), color,2)
    return vis


def GetAllFilesInFolder_Recursive(root):
    ListOfFiles=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            FullpathOfFile=(os.path.join(path, name))
            ListOfFiles.append(FullpathOfFile)
    return ListOfFiles

def SIFT_Unfiltered_Features(imageA,Input_ParameterDictionary):
    ImageLog=[]
    ImageTextLog=[]
    Features_Report=["-------"]
    Features_Report.append("SIFT matcher used")
    #ORB descriptor
    #create some parameters we can swap in and out easily
    
    draw_params=dict(matchColor=(0,255,0),singlePointColor=(255,0,0))
    #create orb object
    
    #ORB=cv2.ORB_create(**Dict_Common.ORB_Testing)#**Dict_Common.ORB_default)
    ORB=cv2.ORB_create(nfeatures=20000)#**Dict_Common.ORB_default)
    
    kp1, des1 = ORB.detectAndCompute(imageA,None)
    
    
    #dictionary comes from main reconstruction file
    #SIFT_features=cv2.xfeatures2d.SIFT_create(**Dict_Common.SIFT_default)
    #kp1, des1 = SIFT_features.detectAndCompute(imageA,None)
    
    
    
    
    Features_Report.append("Raw Keypoints ImgA = " + str(len(kp1)))
    KeypointsORB=imageA.copy()#temp image for displaying features
    display("ORB " + str(len(kp1)) +" unfiltered keypoints Img A", draw_keypoints(KeypointsORB,kp1))
    
    ImageLog.append(draw_keypoints(KeypointsORB,kp1))
    ImageTextLog.append("SIFT " + str(len(kp1)) +" unfiltered keypoints Img A")
    
    
    l_pts1 = []
    l_pts1 = cv2.KeyPoint_convert(kp1)
   
    l_pts1=l_pts1.tolist()
    
    
    ##testing - get area around feature points
    ##turn this off for now
#    l_pts1A=[]
#    l_pts1B=[]
#    l_pts1C=[]
#    l_pts1D=[]
#    l_pts1E=[]
#    l_pts1F=[]
#    l_pts1G=[]
#    l_pts1H=[]
#    print(len(l_pts1))
#    PixelSpread=5
#    for I in l_pts1:
#        l_pts1A.append((I[0]+PixelSpread,I[1]+PixelSpread))
#        l_pts1B.append((I[0]-PixelSpread,I[1]-PixelSpread))
#        l_pts1C.append((I[0]+PixelSpread,I[1]))
#        l_pts1D.append((I[0]-PixelSpread,I[1]))
#        l_pts1E.append((I[0],I[1]+PixelSpread))
#        l_pts1F.append((I[0],I[1]-PixelSpread))
#        l_pts1G.append((I[0]+PixelSpread,I[1]-PixelSpread))
#        l_pts1H.append((I[0]-PixelSpread,I[1]+PixelSpread))
#        
#    l_pts1=l_pts1+  l_pts1A +   l_pts1B + l_pts1C + l_pts1D + l_pts1E + l_pts1F +l_pts1G + l_pts1H
#    print(len(l_pts1))
#    
    
    return l_pts1, Features_Report,ImageLog,ImageTextLog



def SIFT_Feature_and_Match(imageA, imageB,Input_ParameterDictionary,Use_Brute_Force):
    ImageLog=[]
    ImageTextLog=[]
    Features_Report=["-------"]
    BruteForce=Use_Brute_Force
    Features_Report.append("SIFT matcher used")
    #ORB descriptor
    #create some parameters we can swap in and out easily
    
    draw_params=dict(matchColor=(0,255,0),singlePointColor=(255,0,0))
    #create orb object
    #dictionary comes from main reconstruction file
    #TODO initialising SIFT each cycle not optimal!
    SIFT_features=cv2.xfeatures2d.SIFT_create(**Input_ParameterDictionary)
    #SIFT_features=cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=3,
    #                           contrastThreshold=0.04,edgeThreshold=10)
    
    kp1, des1 = SIFT_features.detectAndCompute(imageA,None)
    kp2, des2 = SIFT_features.detectAndCompute(imageB,None)
    Features_Report.append("Raw Keypoints ImgA = " + str(len(kp1)))
    Features_Report.append("Raw Keypoints ImgB = " + str(len(kp2)))
    KeypointsORB=imageA.copy()#temp image for displaying features
    #display("SIFT " + str(len(kp1)) +" unfiltered keypoints Img A", draw_keypoints(KeypointsORB,kp1))
    
    ImageLog.append(draw_keypoints(KeypointsORB,kp1))
    ImageTextLog.append("SIFT " + str(len(kp1)) +" unfiltered keypoints Img A")
    
    
    l_good = [] 
    l_pts1 = []
    l_pts2 = []
    
    
        
    
        
    #putting on crosscheck kills FLANN compatibility 
    if BruteForce==False:
        Features_Report.append("Using FLANN matcher")
        matchDistance=0.5
        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matchesknn = bf.knnMatch(des1,des2, k=2)
                
                 
        Features_Report.append("KNN ratio match - distance = " + str(matchDistance))
        
        #use NORM_HAMMING for ORB matches
        # If ORB is using WTA_K == 3 or 4, cv.NORM_HAMMING2 should be used.
        Features_Report.append("KNN matches found = " + str(len(matchesknn)))
        
        # ratio test
        for i,(match1,match2) in enumerate(matchesknn):
            if match1.distance < matchDistance*match2.distance:
                l_good.append([match2])
                l_pts2.append(kp2[match1.trainIdx].pt)
                l_pts1.append(kp1[match1.queryIdx].pt)
        
        Features_Report.append("Ratio matches left  = " + str(len(l_good)))
        
        flann_matches = cv2.drawMatchesKnn(imageA,kp1,imageB,kp2,l_good,None,**draw_params)
        #display("FLANN matches", flann_matches)
        
        ImageLog.append(flann_matches)
        ImageTextLog.append("FLANN matches")
    #matches should be sorted by the crosscheck - can we take the top N?
    #how do we interrogate the "matches" object?
    if BruteForce==True:
        raise Exception("Should not be brute forcing sift ?? cannot proceed - _3dvislab library")
        Features_Report.append("BAD!!! SHOULDNT BE IN HERE!! BRUTE FORCE matcher")
            
        # create Brute-force Matcher object
        # If ORB is using WTA_K == 3 or 4, cv2.NORM_HAMMING2 should be used.
        bf = cv2.BFMatcher()
        #crossheck = true executes a sort of ratio test like we do for SIFT
        
        # Match descriptors.
        matches= bf.match(des1,des2)
        Features_Report.append("Brute Force matches = " + str(len(matches)))
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 100 matches.
        img3=imageA.copy()
        #skim top matches
        ShowMatches=len(matches)
        ShowMatches=min(len(matches),ShowMatches)#protect array access
        ShowMatches=30

        img3=cv2.drawMatches(imageA,kp1,imageB,kp2,matches[:ShowMatches],img3,flags=2,**draw_params)
        #display(str(ShowMatches) + " of " + str(len(matches)) + " brute force ORB matches",img3)
        ImageLog.append(img3)
        ImageTextLog.append(str(ShowMatches) + " of " + str(len(matches)) + " brute force ORB matches")
        cv2.imwrite('CORRECTDISTORTION/BruteForce_Matches.jpg',img3)

        # pop out matches into our point array
        for (match1) in (matches[:]):
                l_good.append([match1])
                l_pts2.append(kp2[match1.trainIdx].pt)
                l_pts1.append(kp1[match1.queryIdx].pt)
            
#     # pop out matches into our point array
#        for (match1) in enumerate(matches):
#            #print(match1.distance)
#            l_good.append([match1])
#            l_pts2.append(kp2[match1.trainIdx].pt)
#            l_pts1.append(kp1[match1.queryIdx].pt)
#        
   
    Features_Report.append("matches returned  = " + str(len(l_pts1)))
    Features_Report.append("-------")
    return l_pts1, l_pts2, Features_Report,ImageLog,ImageTextLog


class MultiProcessPooler:
    #see if we can shove all the multiprocess stuff in a class
    #WARNING! I dont think this is instance friendly.. lets see.. TODO
    
    #NOTES:
    #Remember to wrap the code which calls the POOL in this way:
#    if __name__ == '__main__':  <<-- need this in Windows
#    
#    import SparseReconstruct <<-- put the libraries you want to use under it
#    import _3DVisLabLib
#    import fileinput
#    import glob



    def __init__(self,ProcessSize=0,ClassObjToProcess=[],UserUpdated=False,CheckStr=""):
            self.ProcessSize=ProcessSize
            self.ClassObjToProcess=ClassObjToProcess
            self.UserUpdated=UserUpdated
            self.CheckStr=CheckStr
            
    #UserUpdated=False
    #ProcessSize=0
    cpus = multiprocessing.cpu_count()
    #ClassObjToProcess=[]
    ErrorStr="Parallel Process TimeOut Error - is process run in cmd prompt as  %run xxx.py??"
    UpdateStr="Parallel Processing: timeout in "
    def Reset(self):
        self.ClassObjToProcess=[] 
        self.ClassObjToProcess[:]=[] 
        self.ProcessSize=0
    def AddInputParObject(self,InputParameterObj):
        self.ClassObjToProcess.append(copy.deepcopy(InputParameterObj))
        self.ProcessSize=self.ProcessSize+1
    def StartProcesses(self,Function,Timeout_seconds=60,ParallelProcess=8):
        
        if ParallelProcess>=multiprocessing.cpu_count():
            ParallelProcess==max((int(multiprocessing.cpu_count()-1)),2)
        print(str(len(self.ClassObjToProcess)) + " ParallelProcess using Pool of " + str(ParallelProcess))
        p = Pool(processes=ParallelProcess)
        rs = p.imap_unordered(Function, self.ClassObjToProcess)
        UpdateString=""
        timeout = time.time() + Timeout_seconds
        time_start=time.time()
        while True:
            time.sleep(0.1)#TODO play with this - as WHILE loop can annhilate CPU
            completed = rs._index
            if (completed == self.ProcessSize):
                
                #this tool is for big processes - so if it finishes
                #really fast there might be an error/dead threads hanging about
                if ((time.time())-time_start)<3:
                    print("WARNING: Multiprocess finished too fast, either threads failed or dead threads stuck in memory. Reset PC or set Multiprocess=False")
                else:
                    print("finished all processes")
                    
                break
        
            if  time.time() > timeout:
                p.terminate()
                p.join()
                print(MultiProcessPooler.ErrorStr)
                break
            if ((round(time.time()) % 4)==0) and (self.UserUpdated==False):
                
                #WHY IS THIS SO TORTUROUS IN PYTHON
                #FIX THIS MESS LATER 
                #TODO
                #every two seconds pump back assurances
                if not(self.CheckStr==round(timeout-time.time())):
                    if self.UserUpdated==False:#check again i have no idea why this isnt working
                        UpdateString=(MultiProcessPooler.UpdateStr + str(round(timeout-time.time())) + " sec")
                        if self.CheckStr!=UpdateString:
                            print(UpdateString)
                            self.UserUpdated=True
                self.CheckStr=round(timeout-time.time())
                
            else:
                self.UserUpdated=False
                
def OrbKeyPointsOnly(imageA,Input_ParameterDictionary):
    """only get feature matching keypoints"""
    ORB=cv2.ORB_create(**Input_ParameterDictionary)
    kp1, des1 = ORB.detectAndCompute(imageA,None)
    return kp1, des1

def AkazeKeyPointsOnly(imageA):
    """only get feature matching keypoints"""
    detector = cv2.AKAZE_create()
    (kp1, des1) = detector.detectAndCompute(imageA, None)
    return kp1, des1

def Akaze_FeatureMatch(kp1, descs1,kp2, descs2,MaxDistance,image1,image2):
    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])

    OutputMatchImage=None
    if image1 is not None and image2 is not None:
    # Draw only "good" matches
        OutputMatchImage = cv2.drawMatches(img1 = image1,
                                keypoints1 = kp1,
                                img2 = image2,
                                keypoints2 = kp2,
                                matches1to2 = good,
                                outImg = None,
                                flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                
def Orb_FeatureMatch(kp1, des1,kp2, des2,MaxDistance,Image1,Image2,MatchRatio, DefaultDUmmyValue):
    #ratio recommmended by Lowe is 0.75
    l_good = [] 
    l_pts1 = []
    l_pts2 = []
    # create Brute-force Matcher object
    # If ORB is using WTA_K == 3 or 4, cv2.NORM_HAMMING2 should be used.
    #bf = cv2.BFMatcher()#we can sort out matches ourself

    #lets try and find a metric for best performing matched images
    LengthKp1=len(kp1)
    LengthKp2=len(kp2)
    BestMatch_len=max(LengthKp1,LengthKp2)



    bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = bf.knnMatch(des1,des2, k=2)


    # Apply ratio test
    #lowe_ratio = 0.7
    Count=0
    Distance=0
    for match1,match2 in matches:
        if match1.distance < MatchRatio*match2.distance:
            l_good.append([match2])
            l_pts2.append(kp2[match1.trainIdx].pt)
            l_pts1.append(kp1[match1.queryIdx].pt)
            Count=Count+1
            Distance=Distance+match1.distance

    #https://docs.opencv.org/3.4/db/d70/tutorial_akaze_matching.html
    #TODO this needs thinking about a bit more
    if Count>0:
        FinalMatchMetric=Count/BestMatch_len
        FinalMatchMetric=1-FinalMatchMetric#lower is better
    else:
        FinalMatchMetric=DefaultDUmmyValue#default value will be removed at end of process

    #FinalMatchMetric=min(FinalMatchMetric,DefaultDUmmyValue)

    OutputMatchImage=None
    if Image1 is not None and Image2 is not None:
        OutputMatchImage=cv2.drawMatchesKnn(Image1,kp1,Image2,kp2,l_good,None,flags=2)
        
    return l_good,OutputMatchImage,l_pts1,l_pts2,FinalMatchMetric




    # If ORB is using WTA_K == 3 or 4, cv2.NORM_HAMMING2 should be used.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)#we can sort out matches ourself
    # Match descriptors.
    matches= bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    #filtered matches
    
    matches_filtered=[]
    for matchobject in matches:
        if matchobject.distance<MaxDistance:
            matches_filtered.append(matchobject)

    matches = sorted(matches_filtered, key = lambda x:x.distance)
    # pop out matches into our point array
    for (match1) in (matches[:]):
        l_good.append([match1])
        l_pts2.append(kp2[match1.trainIdx].pt)
        l_pts1.append(kp1[match1.queryIdx].pt)

    OutputMatchImage=None
    if Image1 is not None and Image2 is not None:
        OutputMatchImage=cv2.drawMatches(Image1,kp1,Image2,kp2,matches[:30],OutputMatchImage,flags=2)
        
    return matches,OutputMatchImage

def ORB_Feature_and_Match_SortImg(imageA, imageB,Input_ParameterDictionary,Use_Brute_Force):
    ImageLog=[]
    ImageTextLog=[]
    Features_Report=["-------"]
    BruteForce=Use_Brute_Force
    Features_Report.append("ORB matcher used")
    #ORB descriptor
    #create some parameters we can swap in and out easily
    
    draw_params=dict(matchColor=(0,255,0),singlePointColor=(255,0,0))
    #create orb object
    #dictionary comes from main reconstruction file
    ORB=cv2.ORB_create(**Input_ParameterDictionary)
    kp1, des1 = ORB.detectAndCompute(imageA,None)
    kp2, des2 = ORB.detectAndCompute(imageB,None)
    Features_Report.append("Raw Keypoints ImgA = " + str(len(kp1)))
    Features_Report.append("Raw Keypoints ImgB = " + str(len(kp2)))
    KeypointsORB=imageA.copy()#temp image for displaying features
    #display("ORB " + str(len(kp1)) +" unfiltered keypoints Img A", draw_keypoints(KeypointsORB,kp1))
    ImageLog.append(draw_keypoints(KeypointsORB,kp1))
    ImageTextLog.append("ORB " + str(len(kp1)) +" unfiltered keypoints Img A")
    
    
    l_good = [] 
    l_pts1 = []
    l_pts2 = []
    
    
    #putting on crosscheck kills FLANN compatibility 
    if BruteForce==False:
        Features_Report.append("Using FLANN matcher")
        matchDistance=0.2
        #FLANN Parameters for ORB (need different mode for SIFT etc)
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
        search_params = dict(checks=100)   # or pass empty dictionary
        #create flann matcher
        flann=cv2.FlannBasedMatcher(index_params,search_params)
        
         
        Features_Report.append("KNN ratio match - distance = " + str(matchDistance))
        
        #use NORM_HAMMING for ORB matches
        # If ORB is using WTA_K == 3 or 4, cv.NORM_HAMMING2 should be used.
        matchesknn=flann.knnMatch(des1,des2, k=2)#returns K best matches
        Features_Report.append("KNN matches found = " + str(len(matchesknn)))
        
        # ratio test
        for i,(match1,match2) in enumerate(matchesknn):
            if match1.distance < matchDistance*match2.distance:
                l_good.append([match2])
                l_pts2.append(kp2[match1.trainIdx].pt)
                l_pts1.append(kp1[match1.queryIdx].pt)
        
        Features_Report.append("Ratio matches left  = " + str(len(l_good)))
        
        flann_matches = cv2.drawMatchesKnn(imageA,kp1,imageB,kp2,l_good,None,**draw_params)
        #display("FLANN matches", flann_matches)
        ImageLog(flann_matches)
        ImageTextLog("FLANN matches")
    #matches should be sorted by the crosscheck - can we take the top N?
    #how do we interrogate the "matches" object?
    if BruteForce==True:
        Features_Report.append("Using BRUTE FORCE matcher")
            
        # create Brute-force Matcher object
        # If ORB is using WTA_K == 3 or 4, cv2.NORM_HAMMING2 should be used.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)#we can sort out matches ourself
        #crossheck = true executes a sort of ratio test like we do for SIFT
        
        # Match descriptors.
        matches= bf.match(des1,des2)
        Features_Report.append("Brute Force matches = " + str(len(matches)))
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        #UnsortedMatches=copy.deepcopy(matches)

        #print("Top match=",matches[0].distance)
        #filtered matches
        matches_filtered=[]
        for matchobject in matches:
            if matchobject.distance<9999999999:
                matches_filtered.append(matchobject)
        
        matches = sorted(matches_filtered, key = lambda x:x.distance)


        # Draw first 100 matches.
        img3=imageA.copy()
        #skim top matches
        ShowMatches=len(matches)
        ShowMatches=min(len(matches),ShowMatches)#protect array access
        img3=cv2.drawMatches(imageA,kp1,imageB,kp2,matches[:30],img3,flags=2,**draw_params)
        
        #display(str(ShowMatches) + " of " + str(len(matches)) + " brute force ORB matches",img3)
        ImageLog.append(img3)
        ImageTextLog.append(str(ShowMatches) + " of " + str(len(matches)) + " brute force ORB matches")
        #cv2.imwrite('CORRECTDISTORTION/BruteForce_Matches.jpg',img3)

        # pop out matches into our point array
        for (match1) in (matches[:]):
            l_good.append([match1])
            l_pts2.append(kp2[match1.trainIdx].pt)
            l_pts1.append(kp1[match1.queryIdx].pt)
            
#     # pop out matches into our point array
#        for (match1) in enumerate(matches):
#            #print(match1.distance)
#            l_good.append([match1])
#            l_pts2.append(kp2[match1.trainIdx].pt)
#            l_pts1.append(kp1[match1.queryIdx].pt)
#        
   
    Features_Report.append("matches returned  = " + str(len(l_pts1)))
    Features_Report.append("-------")
    return l_pts1, l_pts2, Features_Report,ImageLog,ImageTextLog,matches

            
            
def ORB_Feature_and_Match(imageA, imageB,Input_ParameterDictionary,Use_Brute_Force):
    ImageLog=[]
    ImageTextLog=[]
    Features_Report=["-------"]
    BruteForce=Use_Brute_Force
    Features_Report.append("ORB matcher used")
    #ORB descriptor
    #create some parameters we can swap in and out easily
    
    draw_params=dict(matchColor=(0,255,0),singlePointColor=(255,0,0))
    #create orb object
    #dictionary comes from main reconstruction file
    ORB=cv2.ORB_create(**Input_ParameterDictionary)
    kp1, des1 = ORB.detectAndCompute(imageA,None)
    kp2, des2 = ORB.detectAndCompute(imageB,None)
    Features_Report.append("Raw Keypoints ImgA = " + str(len(kp1)))
    Features_Report.append("Raw Keypoints ImgB = " + str(len(kp2)))
    KeypointsORB=imageA.copy()#temp image for displaying features
    #display("ORB " + str(len(kp1)) +" unfiltered keypoints Img A", draw_keypoints(KeypointsORB,kp1))
    ImageLog.append(draw_keypoints(KeypointsORB,kp1))
    ImageTextLog.append("ORB " + str(len(kp1)) +" unfiltered keypoints Img A")
    
    
    l_good = [] 
    l_pts1 = []
    l_pts2 = []
    
    
    #putting on crosscheck kills FLANN compatibility 
    if BruteForce==False:
        Features_Report.append("Using FLANN matcher")
        matchDistance=0.2
        #FLANN Parameters for ORB (need different mode for SIFT etc)
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
        search_params = dict(checks=100)   # or pass empty dictionary
        #create flann matcher
        flann=cv2.FlannBasedMatcher(index_params,search_params)
        
         
        Features_Report.append("KNN ratio match - distance = " + str(matchDistance))
        
        #use NORM_HAMMING for ORB matches
        # If ORB is using WTA_K == 3 or 4, cv.NORM_HAMMING2 should be used.
        matchesknn=flann.knnMatch(des1,des2, k=2)#returns K best matches
        Features_Report.append("KNN matches found = " + str(len(matchesknn)))
        
        # ratio test
        for i,(match1,match2) in enumerate(matchesknn):
            if match1.distance < matchDistance*match2.distance:
                l_good.append([match2])
                l_pts2.append(kp2[match1.trainIdx].pt)
                l_pts1.append(kp1[match1.queryIdx].pt)
        
        Features_Report.append("Ratio matches left  = " + str(len(l_good)))
        
        flann_matches = cv2.drawMatchesKnn(imageA,kp1,imageB,kp2,l_good,None,**draw_params)
        #display("FLANN matches", flann_matches)
        ImageLog(flann_matches)
        ImageTextLog("FLANN matches")
    #matches should be sorted by the crosscheck - can we take the top N?
    #how do we interrogate the "matches" object?
    if BruteForce==True:
        Features_Report.append("Using BRUTE FORCE matcher")
            
        # create Brute-force Matcher object
        # If ORB is using WTA_K == 3 or 4, cv2.NORM_HAMMING2 should be used.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)#we can sort out matches ourself
        #crossheck = true executes a sort of ratio test like we do for SIFT
        
        # Match descriptors.
        matches= bf.match(des1,des2)
        Features_Report.append("Brute Force matches = " + str(len(matches)))
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        #print("Top match=",matches[0].distance)
        #filtered matches
        matches_filtered=[]
        for matchobject in matches:
            if matchobject.distance<9999999999:
                matches_filtered.append(matchobject)
        
        matches = sorted(matches_filtered, key = lambda x:x.distance)


        # Draw first 100 matches.
        img3=imageA.copy()
        #skim top matches
        ShowMatches=len(matches)
        ShowMatches=min(len(matches),ShowMatches)#protect array access
        img3=cv2.drawMatches(imageA,kp1,imageB,kp2,matches[:30],img3,flags=2,**draw_params)
        
        #display(str(ShowMatches) + " of " + str(len(matches)) + " brute force ORB matches",img3)
        ImageLog.append(img3)
        ImageTextLog.append(str(ShowMatches) + " of " + str(len(matches)) + " brute force ORB matches")
        #cv2.imwrite('CORRECTDISTORTION/BruteForce_Matches.jpg',img3)

        # pop out matches into our point array
        for (match1) in (matches[:]):
            l_good.append([match1])
            l_pts2.append(kp2[match1.trainIdx].pt)
            l_pts1.append(kp1[match1.queryIdx].pt)
            
#     # pop out matches into our point array
#        for (match1) in enumerate(matches):
#            #print(match1.distance)
#            l_good.append([match1])
#            l_pts2.append(kp2[match1.trainIdx].pt)
#            l_pts1.append(kp1[match1.queryIdx].pt)
#        
   
    Features_Report.append("matches returned  = " + str(len(l_pts1)))
    Features_Report.append("-------")
    return l_pts1, l_pts2, Features_Report,ImageLog,ImageTextLog


def GetFilesSortedByDate(dirpath):
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    return a

def ReturnPrefixOfFiles(ArrayOf_Files, PrefixChar):
    Prefix=[]
    AssociatedFileName=[]
    tempstring=""
    for sItem in ArrayOf_Files:
        IndexOfChar=sItem.find(PrefixChar)
        if IndexOfChar != -1:
            tempstring=sItem
            tempstring=sItem[:IndexOfChar]
            Prefix.append(tempstring)
            AssociatedFileName.append(sItem)
    return Prefix,AssociatedFileName

def CleanUpXYMatches_With_Ferror(Array_Of_F_Errors,FundamentalMatrix, Points1,Points2,Threshold):
    #will only clean up points that are not on corresponding epiline
    #so be careful
    Log=["CleanUpXYMatches_With_Ferror"]
    Log.append("Input Threshold: " + str(Threshold))
    NewPointsA=[]#TODO i dont understand python datatypes yet-
    NewPointsB=[]
    #so do it this way until you figure it out
    
    #test some random points to make sure we havent
    #been fed in nonsense
    RandomIndex1=(round(random.random()*len(Points1)))-1
    RandomIndex2=(round(random.random()*len(Points1)))-1
    RandomIndex1_Ferr=useF_to_test_point(FundamentalMatrix,Points1[RandomIndex1],Points2[RandomIndex1],0)
    RandomIndex2_Ferr=useF_to_test_point(FundamentalMatrix,Points1[RandomIndex2],Points2[RandomIndex2],0)
    #watch testing these - might need some kind of epsilon test
    #for these floats
    if round(Array_Of_F_Errors[RandomIndex1],6)!=round(RandomIndex1_Ferr,6):
        #print("CleanUpXYMatches_With_Ferror Error index" + str(RandomIndex1_Ferr))
        #print(str(Array_Of_F_Errors[RandomIndex1]) + " is not close to " + str (RandomIndex1_Ferr))
        Log.append(str(Array_Of_F_Errors[RandomIndex1]) + " is not close to " + str (RandomIndex1_Ferr))
        return False,None,None, Log
    if round(Array_Of_F_Errors[RandomIndex2],6)!=round(RandomIndex2_Ferr,6):
        #print("CleanUpXYMatches_With_Ferror Error index" + str(RandomIndex2_Ferr))
        #print(str(Array_Of_F_Errors[RandomIndex2]) + " is not close to " + str (RandomIndex2_Ferr))
        Log.append(str(Array_Of_F_Errors[RandomIndex2]) + " is not close to " + str (RandomIndex2_Ferr))
        return False,None,None, Log
    #array of errors should match up with incoming pairs of points as tested above
    #all 3 arrays should be same length
    if not(len(Array_Of_F_Errors)==len(Points1)==len(Points2)):
        #print("Arrays not all same length CleanUpXYMatches_With_Ferror")
        Log.append("Arrays not all same length CleanUpXYMatches_With_Ferror")
        return False,None,None, Log
    #no errors here - we can continue
    for Index in range (len(Array_Of_F_Errors)):
        if abs(Array_Of_F_Errors[Index])<=abs(Threshold):
            NewPointsA.append(Points1[Index])
            NewPointsB.append(Points2[Index])
    
    #TODO this is bad code - need more time to learn python
    FilteredPointsA=np.zeros((len(NewPointsA),2),dtype=np.int32)
    FilteredPointsB=np.zeros((len(NewPointsA),2),dtype=np.int32)
    
    if (len(FilteredPointsB))<1:
        #print("CleanUpXYMatches_With_Ferror filtered points less than 1")
        Log.append("CleanUpXYMatches_With_Ferror filtered points less than 1")
        return False,None,None, Log
    
    for Index in range (len(NewPointsA)):
        FilteredPointsA[Index]=NewPointsA[Index]
        FilteredPointsB[Index]=NewPointsB[Index]
        
        
    #random quality check again
    RandomIndex3=(round(random.random()*len(NewPointsA)))-1
    RandomIndex3_Ferr=useF_to_test_point(FundamentalMatrix,NewPointsA[RandomIndex3],NewPointsB[RandomIndex3],0)
    if abs(RandomIndex3_Ferr)>Threshold:
        Log.append("ERROR CleanUpXYMatches_With_Ferror, quality check fail after filter ")
        #print("ERROR CleanUpXYMatches_With_Ferror, quality check fail after filter ")
        return False,None,None, Log
    Log.append("filter out points F error " + str(len(NewPointsA)) + " remaining from " + str(len(Array_Of_F_Errors)))
    
    return True, FilteredPointsA,FilteredPointsB, Log
 


def testlibrary(input):
    print(input)


def Fzero_Quality_Check_FeaturePair(FeaturePoints1,FeaturePoints2,TestIndex,FundamentalMatrix,ColourImgA,ColourImgB):
    #sanity check - print manual method of obtaining epipolar constraint
    #and new automatic method - to see if we have variables in correct place
    #test the formula to 
    #see if we get ZERO (as x^T*F*x’=(0) ) and can prove we understand the math
 
    test_validity=min(len(FeaturePoints1),TestIndex)
    if test_validity!=TestIndex:
        print("Index too big, reduced from " + str(TestIndex) + " to " + str(test_validity) + " in Fzero_Quality_Check_FeaturePair")
        TestIndex=test_validity-1#
    
    if len(FeaturePoints1)!=len(FeaturePoints2):
        print("Feature points arent the same!! in Fzero_Quality_Check_FeaturePair" )
        TestIndex=min(len(FeaturePoints1),len(FeaturePoints2))
    
    #TestIndex=TestIndex-1#adjust this to check random corresponding feature
    pointXY=(FeaturePoints1[TestIndex,0],FeaturePoints1[TestIndex,1])
    PointXY_Prime=(FeaturePoints2[TestIndex,0],FeaturePoints2[TestIndex,1])
    
   
    
    #corresponding pair should be shown above - now lets see if the fundamental matrix formula gets us something close to ZERO
    #formula = X'T * F * X' = 0
    #PointX transform * FundamentalMatrix * COrrespondingPointX=0
    
    #i think the XY matrixs have to be in form "X,Y,1".. dunno why though/
    PointXY_Matrix = np.ones(shape=(3))
    PointXY_Prime_Matrix = np.ones(shape=(3))
    PointXY_Matrix[0]=pointXY[0]
    PointXY_Matrix[1]=pointXY[1]
    PointXY_Prime_Matrix[0]=PointXY_Prime[0]
    PointXY_Prime_Matrix[1]=PointXY_Prime[1]
    
    # convert to actual matrices maybe????
    #this step can be done using numpy matrix casting, or just dot products of basic arrays
    np_XYMatrix=np.matrix(PointXY_Matrix)
    np_XYMatrixPrime=np.matrix(PointXY_Prime_Matrix)
    np_FMatrix=np.matrix(FundamentalMatrix)
    np_Error=(np_XYMatrixPrime*np_FMatrix)*np_XYMatrix.transpose()#make sure you multiply the matrices in the correct order
    #this should be close to zero according to formula
    #print("Manual error " + str(round(np_Error[0,0],6)))
    np_auto_error=useF_to_test_point(FundamentalMatrix,pointXY,PointXY_Prime,0)
    #print("Second method" + str(round(np_auto_error,6)))
  
    #generate images
    imgTestMatch=ColourImgA.copy()
    imgTestMatch2=ColourImgB.copy()
    cv2.circle(img=imgTestMatch,center=pointXY,radius=20,color=GetColour_of_F_Error(np_auto_error),thickness=5)
    cv2.circle(img=imgTestMatch2,center=PointXY_Prime,radius=20,color=GetColour_of_F_Error(np_auto_error),thickness=5)
    
    cv2.circle(img=imgTestMatch,center=pointXY,radius=1,color=(255,0,0),thickness=2)
    cv2.circle(img=imgTestMatch2,center=PointXY_Prime,radius=1,color=(255,0,0),thickness=2)
    
    AverageCol, imgTestMatch=SampleAreaOfImage(imgTestMatch,pointXY[0],pointXY[1],AreaSpan=200)
    AverageCol, imgTestMatch2=SampleAreaOfImage(imgTestMatch2,PointXY_Prime[0],PointXY_Prime[1],AreaSpan=200)
    
    
    
    return imgTestMatch, imgTestMatch2, np_auto_error
def GetColour_of_F_Error(F_Error):
    #input an error (by testing two pairs of co-ordinates using
    #epipolar constraint/Fundamental matrix)
    #and return a standardised colour
    FColour=(1,2,3)
    
    if abs(F_Error)<Str_Common.FError_VGood:
        FColour=Str_Common.Green
    if abs(F_Error)>=Str_Common.FError_Acceptable:
        FColour=Str_Common.Orange
    if abs(F_Error)>Str_Common.FError_Failure:
        FColour=Str_Common.Red
    return FColour

def GetStatsPlotHist_of_F_Errors(CheckF_Errors_Array):
    #lets get some stats
    CountLargeErrors=0
    ErrorThreshold_LargerThan=Str_Common.FError_Failure
    for Indxer in range(len(CheckF_Errors_Array)):
        if abs(CheckF_Errors_Array[Indxer])>ErrorThreshold_LargerThan:
            CountLargeErrors=CountLargeErrors+1
    print(str(ErrorThreshold_LargerThan)+ " FAIL error count " + str(CountLargeErrors))

    CountLargeErrors=0
    ErrorThreshold_LessThan=Str_Common.FError_VGood
    for Indxer in range(len(CheckF_Errors_Array)):
        if abs(CheckF_Errors_Array[Indxer])<ErrorThreshold_LessThan:
            CountLargeErrors=CountLargeErrors+1
    print(str(ErrorThreshold_LessThan)+ " vgood error count " + str(CountLargeErrors))
    
    #plot histogram - the matlab thing
    #doesnt play nice for some reason like this -
    #TODO need to see why 
    
    #plt.hist(CheckF_Errors_Array,256,[min(CheckF_Errors_Array),max(CheckF_Errors_Array)])
    #plt.show()
    #buf = io.BytesIO()
    #plt.savefig(buf, format='jpg')
    #buf.seek(0)
    #im = Image.open(buf)
    #im.show()
    
    #generate random gaussian data to give an example of good error distribution
    mu, sigma = 0, 0.001 # mean and standard deviation
    ExampleError = np.random.normal(mu, sigma, 500)
    return ExampleError





def GetList_Of_F_Error_SingleArray(l_StereoPairs_pnts1,l_StereoPairs_pnts2,FundamentalMatrix):
    ErrorList=[]
    for testindex in range(len(l_StereoPairs_pnts1)):
        temp=(useF_to_test_point(FundamentalMatrix,
                                              (l_StereoPairs_pnts1[testindex][0],
                                              l_StereoPairs_pnts1[testindex][1]),
                                             (l_StereoPairs_pnts2[testindex][0],
                                              l_StereoPairs_pnts2[testindex][1]),0))
        ErrorList.append((temp))
    return ErrorList





def GetList_Of_F_Error(lCalibs_StereoPairs_pnts1,lCalibs_StereoPairs_pnts2,FundamentalMatrix):
    ErrorList=[]
    for groupindex in range(len(lCalibs_StereoPairs_pnts1)):
        for testindex in range(len(lCalibs_StereoPairs_pnts1[groupindex])):
            temp=(useF_to_test_point(FundamentalMatrix,
                                              (lCalibs_StereoPairs_pnts1[groupindex][testindex][0][0],
                                              lCalibs_StereoPairs_pnts1[groupindex][testindex][0][1]),
                                             (lCalibs_StereoPairs_pnts2[groupindex][testindex][0][0],
                                              lCalibs_StereoPairs_pnts2[groupindex][testindex][0][1]),0))
            ErrorList.append(temp) 
    return ErrorList


def SampleAreaOfImage(ColourImage,lPointX,lPointY,AreaSpan):
    #get image dimensions - might break on B&W images
    Height, width, dimension=ColourImage.shape
    #make sure sensible input for Span
    Span=max(AreaSpan,1)
    #stay within image boundary
    PointXNeg=max(lPointX-Span, 0)
    PointXpos=min(lPointX+Span, width)
    PointYNeg=max(lPointY-Span, 0)
    PointYpos=min(lPointY+Span, Height)
    
    if PointXNeg<1:
        Nothing=[]
    if PointYNeg<1:
        Nothing=[]
    if PointXpos<1:
        Nothing=[]
    if PointYpos<1:
        Nothing=[]
        
        
    region_of_interest = (PointXNeg, PointYNeg, PointXpos, PointYpos) # left, top, bottom, right
    cropped_img = ColourImage[region_of_interest[1]:region_of_interest[3], region_of_interest[0]:region_of_interest[2]]
    MeanColour=cv2.mean(cropped_img)
    return MeanColour, cropped_img

def SumAreaOfImage(Input2D_Array,lPointX,lPointY,AreaSpan):
    #get image dimensions - might break on B&W images
    Height, width= Input2D_Array.shape
    #make sure sensible input for Span
    Span=max(AreaSpan,1)
    #stay within image boundary
    PointXNeg=max(lPointX-Span, 0)
    PointXpos=min(lPointX+Span, width)
    PointYNeg=max(lPointY-Span, 0)
    PointYpos=min(lPointY+Span, Height)
    
    if PointXNeg<1:
        Nothing=[]
    if PointYNeg<1:
        Nothing=[]
    if PointXpos<1:
        Nothing=[]
    if PointYpos<1:
        Nothing=[]
        
        
    region_of_interest = (PointXNeg, PointYNeg, PointXpos, PointYpos) # left, top, bottom, right
    cropped_img = (Input2D_Array[region_of_interest[1]:region_of_interest[3], region_of_interest[0]:region_of_interest[2]])
    #check that area hasnt been cropped 
    #if so - make up missing elements by using MEAN
    #this is not good implementation but will give us an idea
    #of the performance
    MakeUpSummed=0
    if (cropped_img.shape[0]<(AreaSpan*2)) or (cropped_img.shape[0]<(AreaSpan*2)):
        MissingElements=((AreaSpan*2)*(AreaSpan*2))-(cropped_img.shape[0]*cropped_img.shape[1])
        MeanOfCroppedSample=np.mean(cropped_img)
        MakeUpSummed= (MissingElements*MeanOfCroppedSample)+9999999
    
    Summed_Area=np.sum(cropped_img)+MakeUpSummed
    return Summed_Area, cropped_img

 
class StereoCalibDataStructure():#v1
    def __init__(self, str_SequenceTitle,
                    str_SequenceCamA,
                    str_SequenceCamB,
                    GridX,GridY,
                    Good_PairCount,
                    Calibs_StereoPairs_pnts1,
                    Calibs_StereoPairs_pnts2,
                    IMGSCalibs_SingleCam_pnts1,
                    IMGSCalibs_SingleCam_pnts2,
                    objPnts):
        self.str_SequenceTitle = str_SequenceTitle
        self.str_SequenceCamA = str_SequenceCamA
        self.str_SequenceCamB = str_SequenceCamB
        self.GridX=GridX
        self.GridY=GridY
        self.objPnts=objPnts
        self.Good_PairCount=Good_PairCount
        self.Calibs_StereoPairs_pnts1=Calibs_StereoPairs_pnts1
        self.Calibs_StereoPairs_pnts2=Calibs_StereoPairs_pnts2
        self.IMGSCalibs_SingleCam_pnts1=IMGSCalibs_SingleCam_pnts1
        self.IMGSCalibs_SingleCam_pnts2=IMGSCalibs_SingleCam_pnts2
        
#make new instance of class
#MyStereoCalib_details=StereoCalibDataStructure(None,None,None,None,None,None,None,None,None,None,None)

def TestOrientation(InputArrayOfXY,ExpectedIndexOfMax_SUMXY):#v1
    #input array of coordinates in weird tuple form [(x,y),(x,y)]
    #specify what index where expected biggest SUM of X and Y is
    #(that is to say - the corner with biggest X and biggest Y)
    #return true or false
    #TODO probably can be done faster - dont need every element maybe just check corners?
    #don't know what edge cases we are too expect at the moment
    tempmax=0
    Index_MaxGridXYSum=0
    tempmax=0
    ii=0
    Match=False
    for ii in range (len(InputArrayOfXY)):
        tempSum=InputArrayOfXY[ii][-1,0]*InputArrayOfXY[ii][-1,1]
        #print(tempSum)
        if tempmax<tempSum:
            Index_MaxGridXYSum=ii
            tempmax=tempSum
    #print("XYsum maximum found at position ", Index_MaxGridXYSum)
    if ExpectedIndexOfMax_SUMXY==Index_MaxGridXYSum:
        Match=True
    return Match

def MakeFolder(FolderPath):
    #single tier folder
    os.mkdir(FolderPath)

def MakeFolders(FolderPath):
    #nested folders
    os.makedirs(FolderPath)


def DeleteFiles_RecreateFolder(FolderPath):
    Deltree(FolderPath)
    os.mkdir(FolderPath)
    
def DelFiles_inFolder(Folderpath):#v1
    DIR = os.listdir(Folderpath)
    for i in range(len(DIR)):
        os.remove(Folderpath+DIR[i])
    return

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


#TRIANGULATION uses a SVD composition to compute the solution
#the points (each row in the above matrix represents a 4D point) are normalized to unit vectors.
#homogeneous point is defined as[X Y Z 1]
def GenerateTestPntCloudsAndImgs_Calibs(lCalibs_StereoPairs_pnts1,
                                        lCalibs_StereoPairs_pnts2,
                                        lPmatrix_manualA,
                                        lPmatrix_manualB,
                                        lIMGSCalibs_SingleCam_pnts1,
                                       lstr_SequenceCamA,
                                       lstr_SequenceCamB,
                                        lstr_SequenceTitle,
                                       lFLDR_POINTCLOUDS):
    for TestCalibrationIndex in range(len(lCalibs_StereoPairs_pnts1)):
        #ADD PROTECTION HERE FROM EMPTY CALIBS Todo
        #homogeneous to Euclidean space:
        TriangulatedPoints4D=cv2.triangulatePoints(lPmatrix_manualA,#2XN array
                                                   lPmatrix_manualB,#2XN array
                                                   lCalibs_StereoPairs_pnts1[TestCalibrationIndex],
                                                   lCalibs_StereoPairs_pnts2[TestCalibrationIndex])
        #so we need to take each 4D point and convert to 3D so we can plot it
        #prepare an array for our new calculations
        TriangulatedPoints3D=TriangulatedPoints4D[0:3,:].copy()#now XYZ instead of XYZW
        TriangulatedPoints3D[:,:]=0
        #now we convert from projective coords to euclidian
        #opencv function convertPointsFromHomogeneous() exists but is useless whiney garbage for such a basic task - just do it manually
        #formula is easy new euclid point XYZ = (X/W,Y/W,Z/W) where W is the fourth element in each projective coordinate 
        Dimension4D,Length4D= TriangulatedPoints4D.shape
        XYZ_Array = ["" for x in range(Length4D)]#create string array to hold XYZs for saving to a file
        for ii in range (Length4D):
                #print (TriangulatedPoints4D[3,ii])#in form XYZW - W is 3rd element
                TriangulatedPoints3D[0,ii]=TriangulatedPoints4D[0,ii]/TriangulatedPoints4D[3,ii]
                TriangulatedPoints3D[1,ii]=TriangulatedPoints4D[1,ii]/TriangulatedPoints4D[3,ii]
                TriangulatedPoints3D[2,ii]=TriangulatedPoints4D[2,ii]/TriangulatedPoints4D[3,ii]
                Xelem=str(round(TriangulatedPoints3D[0,ii]/100,3))
                Yelem=str(round(TriangulatedPoints3D[1,ii]/100,3))
                Zelem=str(round(TriangulatedPoints3D[2,ii]/100,3))
                XYZ_Array[ii]=Xelem + " " + Yelem + " " + Zelem +""
                if ii>0:
                    Xsq=(TriangulatedPoints3D[0,ii]-TriangulatedPoints3D[0,ii-1])*(TriangulatedPoints3D[0,ii]-TriangulatedPoints3D[0,ii-1])
                    Ysq=(TriangulatedPoints3D[1,ii]-TriangulatedPoints3D[1,ii-1])*(TriangulatedPoints3D[1,ii]-TriangulatedPoints3D[1,ii-1])
                    Zsq=(TriangulatedPoints3D[2,ii]-TriangulatedPoints3D[2,ii-1])*(TriangulatedPoints3D[2,ii]-TriangulatedPoints3D[2,ii-1])
                    #print(np.sqrt(Xsq+Ysq+Zsq))
        PointCloudImageCamA = cv2.imread(lIMGSCalibs_SingleCam_pnts1[TestCalibrationIndex])
        Filename=lstr_SequenceCamA + lstr_SequenceCamB + lstr_SequenceTitle + "_Calib" + str(TestCalibrationIndex) 
        cv2.imwrite(lFLDR_POINTCLOUDS +'/' + Filename + ".jpg",PointCloudImageCamA)
        np.savetxt(lFLDR_POINTCLOUDS +'/' + Filename + ".xyz", XYZ_Array, delimiter=" ", newline = "\n", fmt="%s")
        
        
        
#TRIANGULATION uses a SVD composition to compute the solution
#the points (each row in the above matrix represents a 4D point) are normalized to unit vectors.
#homogeneous point is defined as[X Y Z 1]
def GenerateSubjectPntCloudsAndImgs_Calibs(lCalibs_StereoPairs_pnts1,
                                        lCalibs_StereoPairs_pnts2,
                                        lPmatrix_manualA,
                                        lPmatrix_manualB,
                                        lIMGSCalibs_SingleCam_pnts1,
                                       lstr_SequenceCamA,
                                       lstr_SequenceCamB,
                                        lstr_SequenceTitle,
                                       lFLDR_POINTCLOUDS,
                                          list_ColourArray,
                                          SeperatorCharactor):

        TriangulatedPoints4D=cv2.triangulatePoints(lPmatrix_manualA,#2XN array
                                                   lPmatrix_manualB,#2XN array
                                                   lCalibs_StereoPairs_pnts1,
                                                   lCalibs_StereoPairs_pnts2)
        
        DistanceFilteredPoints=0
        #so we need to take each 4D point and convert to 3D so we can plot it
        #prepare an array for our new calculations
        TriangulatedPoints3D=TriangulatedPoints4D[0:3,:].copy()#now XYZ instead of XYZW
        TriangulatedPoints3D[:,:]=0
        #now we convert from projective coords to euclidian
        #opencv function convertPointsFromHomogeneous() exists but is useless whiney garbage for such a basic task - just do it manually
        #formula is easy new euclid point XYZ = (X/W,Y/W,Z/W) where W is the fourth element in each projective coordinate 
        Dimension4D,Length4D= TriangulatedPoints4D.shape
        XYZ_Array = ["" for x in range(Length4D)]#create string array to hold XYZs for saving to a file
        for ii in range (Length4D):
                #print (TriangulatedPoints4D[3,ii])#in form XYZW - W is 3rd element
                #p;. ]TriangulatedPoints4D[3,ii]=0.001
                TriangulatedPoints3D[0,ii]=TriangulatedPoints4D[0,ii]/TriangulatedPoints4D[3,ii]
                TriangulatedPoints3D[1,ii]=TriangulatedPoints4D[1,ii]/TriangulatedPoints4D[3,ii]
                TriangulatedPoints3D[2,ii]=TriangulatedPoints4D[2,ii]/TriangulatedPoints4D[3,ii]
                Xelem=str(round(TriangulatedPoints3D[0,ii]/100,3))
                Yelem=str(round(TriangulatedPoints3D[1,ii]/100,3))
                Zelem=str(round(TriangulatedPoints3D[2,ii]/100,3))
                #colour for point cloud
                REDelem=str(round(list_ColourArray[ii][0]))
                GREENelem=str(round(list_ColourArray[ii][1]))
                BLUEelem=str(round(list_ColourArray[ii][2]))
                
#                Distance=np.sqrt((TriangulatedPoints3D[0,ii]*TriangulatedPoints3D[0,ii])+ (TriangulatedPoints3D[1,ii]*TriangulatedPoints3D[1,ii])+(TriangulatedPoints3D[2,ii]*TriangulatedPoints3D[2,ii]))
#                
#                if Distance>10:
#                    DistanceFilteredPoints=DistanceFilteredPoints+1
#                    Xelem=str(0)
#                    Yelem=str(0)
#                    Zelem=str(0)
#                
                
#                if abs(list_ColourArray[ii][0]-36)<20:
#                    if abs(list_ColourArray[ii][1]-174)<20:
#                        if abs(list_ColourArray[ii][2]-236)<20:
#                            #force default values
#                            print("Forced teehe")
#                            Xelem=str(-1.14)
#                            Yelem=str(-0.885)
#                            Zelem=str(8.348)

                
                XYZ_Array[ii]=Xelem + SeperatorCharactor + Yelem + SeperatorCharactor + Zelem +SeperatorCharactor + REDelem + SeperatorCharactor + GREENelem + SeperatorCharactor + BLUEelem +SeperatorCharactor
       # PointCloudImageCamA = cv2.imread(lIMGSCalibs_SingleCam_pnts1[TestCalibrationIndex])
        Filename=lstr_SequenceCamA + lstr_SequenceCamB + lstr_SequenceTitle + "" + str("0") 
        #cv2.imwrite(lFLDR_POINTCLOUDS +'/' + Filename + ".jpg",PointCloudImageCamA)
        np.savetxt(lFLDR_POINTCLOUDS + Filename + ".txt", XYZ_Array, delimiter=" ", newline = "\n", fmt="%s")        
        if DistanceFilteredPoints>0:
            print(str(DistanceFilteredPoints) + " triangulated points have been filtered out! see : GenerateSubjectPntCloudsAndImgs_Calibs")
   

    
def UNUSEDrotate_matrix(matrix, degree):#v1
    #if abs(degree) not in [0, 90, 180, 270, 360]:
        # raise error or just return nothing or original
    if degree == 0:
        return matrix
    elif degree > 0:
        return rotate(zip(*matrix[::-1]), degree-90)
    else:
        return rotate(zip(*matrix)[::-1], degree+90)
    
    

def CorrectImageDistortion(FolderPath,ImagesA,ImagesB,l_matrixA,l_distortionsA,l_matrixB,l_distortionsB,index):
#v1
    # sanity manual check for distortion matrix from calibratecamera function
#user can manually peep in this folder and see if the correction is sensible
    TestUnDistort = cv2.imread(ImagesA[index])
    dst = cv2.undistort(TestUnDistort, l_matrixA, l_distortionsA, None, None)
    cv2.imwrite(FolderPath+'/A_distorted' + str(index) +'.jpg',TestUnDistort)
    cv2.imwrite(FolderPath+'/A_corrected' + str(index) +'.jpg',dst)

    TestUnDistort = cv2.imread(ImagesB[index])
    dst = cv2.undistort(TestUnDistort, l_matrixB, l_distortionsB, None, None)
    cv2.imwrite(FolderPath+'/B_distorted' + str(index) +'.jpg',TestUnDistort)
    cv2.imwrite(FolderPath+'/B_corrected' + str(index) +'.jpg',dst)
    return



def Matrix_null_Space(a, rtol=1e-5):#v1
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()




def UNUSED_FindProjectionMatrix(CameraMatrix,RotationVector,TranslationVector,AlternativeMethod):
    #v1
    #manually calcualte projection matrix
    #might need to look at final calculation
    #also we only use first element of vector arrays... do we need to make an
    #average or some kind of error minimization??? TODO
    #Two methods give slightly different answers 
    
    #P=CameraMatrix * [R|transformation matrix]
    rotation_mat = np.zeros(shape=(3, 3))
    newRotMat = cv2.Rodrigues(RotationVector, rotation_mat)[0]
    print("manual projection matrix not working yet")
    if AlternativeMethod==False:
        P = np.column_stack((np.matmul(CameraMatrix,newRotMat), TranslationVector))
        return P
    if AlternativeMethod==True:
        #add column to new Rotation matrix - then multiply by camera matrix
        b = np.pad(newRotMat, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        #lame code to add extra column - definitely better ways of doing this
        b[0,3]=TranslationVector[0]
        b[1,3]=TranslationVector[1]
        b[2,3]=TranslationVector[2]
        P=np.mat(CameraMatrix)*np.mat(b)
        #P=np.matmul(CameraMatrix,b)
        return P
    


def FindProjectionMatrix_frmStereoCal(CameraMatrix,RotationMatrix,TranslationVector,AlternativeMethod):
    #v1
    #manually calcualte projection matrix
    #using output from STEREOCALIBRATE
    #Two methods give slightly different answers 
    
    #P=CameraMatrix * [R|transformation vector]
    #t=-RC (C is co-ords of camera centre in world??)
    if AlternativeMethod==False:
        P = np.column_stack((np.matmul(CameraMatrix,RotationMatrix), TranslationVector))
        return P
    if AlternativeMethod==True:
        #add column to new Rotation matrix - then multiply by camera matrix
        b = np.pad(RotationMatrix, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        #lame code to add extra column - definitely better ways of doing this
        b[0,3]=TranslationVector[0]
        b[1,3]=TranslationVector[1]
        b[2,3]=TranslationVector[2]
        P=np.mat(CameraMatrix)*np.mat(b)
        #P=np.matmul(CameraMatrix,b)
        return P
    
    
    
    
def GetImagePairs(FullPath,CaptureTitle,PairMemberID_A,PairMemberID_B):
    #v1
    #when pointing to a folder, this will return 2 lists of paired stereo images 
    #example of use #isOK,PairMemberA,PairMemberB=GetImagePairs("C:/UDemyOpenCV1/CUSTOM/Calibs_Subjects/","Calibrationposes_0","pod1secondary","pod2secondary")
    #TODO check the folder exists first
    ListAllFiles=os.listdir(FullPath)
    FilteredFiles_A=[]
    FilteredFiles_B=[]
    isOK=True #bad logic - not failsafe - update when function gets more complex
    for Filename in ListAllFiles:
        FilenameLwr=Filename.lower()
        if FilenameLwr.endswith(".jpg"):
            if FilenameLwr.find(CaptureTitle.lower()) >-1:
                if FilenameLwr.find(PairMemberID_A.lower()) >-1:
                    FilteredFiles_A.append(FullPath + FilenameLwr)
                if FilenameLwr.find(PairMemberID_B.lower()) >-1:
                    FilteredFiles_B.append(FullPath + FilenameLwr)
    #error checking
    if len(FilteredFiles_A)<1:
        isOK=False   
        print("GetImagePairs Failure: Cannot find A ", PairMemberID_A)
    if len(FilteredFiles_B)<1:
        isOK=False   
        print("GetImagePairs Failure: Cannot find B ", PairMemberID_B)
    if len(FilteredFiles_A)!=len(FilteredFiles_A):
        isOK=False
        print("GetImagePairs Failure: Image pairs do not match")
    return isOK, FilteredFiles_A,FilteredFiles_B





def RotateCalibrationPoints(Calibpoints,CalibGridRows,CalibGridCols,Degree_TBA):
    #v1
    #input an (X,Y) tuple array of calibration points and rotate 
    #create matrix for X and Y coordinates
    Xgrid=np.zeros((10, 10))
    Ygrid=np.zeros((10, 10))
    Xgrid_rotated=np.zeros((10, 10))
    Ygrid_rotated=np.zeros((10, 10))
    #cheap method to strip out weird tuple array format of points and 
    #pop them into two matrices (so we have flexibility for transforms)
    for i in range (10):
        for ii in range (10):
            Xgrid[i,ii]=Calibpoints[ii+(10*i)][-1,0]#x
            Ygrid[i,ii]=Calibpoints[ii+(10*i)][-1,1]#Y
    #VERY cheap method of rotating a matrix
    #this can be improved TODO!!
    RotatedX=np.rot90(Xgrid)
    #RotatedX=np.rot90(RotatedX)
    #RotatedX=np.rot90(RotatedX)
    RotatedY=np.rot90(Ygrid)
    #RotatedY=np.rot90(RotatedY)
    #RotatedY=np.rot90(RotatedY)
    #re-populate the tuple array
    CalibpointsRotated=Calibpoints.copy()
    for a in range (10):
        for b in range (10):
            CalibpointsRotated[b+(10*a)][-1,0]=RotatedX[a,b]
            CalibpointsRotated[b+(10*a)][-1,1]=RotatedY[a,b]
    return CalibpointsRotated






def DrawCentersCircles(img1,pts1):
    #v1
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    for pt1 in (pts1):
        color = tuple(np.random.randint(0,255,3).tolist())
        RidiculousX=pt1[-1,0]#is X, why is this necessary 
        RidiculousY=pt1[0,-1]#is Y
        img1 = cv2.circle(img1,(RidiculousX,RidiculousY),5,color,-1)
    return img1


def GetMedianSizeBlob(keypoints):
    #v1
    for pt1 in (keypoints):
        color = tuple(np.random.randint(0,255,3).tolist())
        RidiculousX=pt1[-1,0]#is X, why is this necessary 
        RidiculousY=pt1[0,-1]#is Y
        img1 = cv2.circle(img1,(RidiculousX,RidiculousY),5,color,-1)
    return img1



def FlipXYpnt_ArrayDims(ArrayToFlip):#really bad code to flip dimensions
    #from (2,N) to (N,2)... don't know Python yet TODO
    (x,y)=ArrayToFlip.shape
    ReallyBadConversion=np.ndarray(shape=(x,1,2),dtype=float)
    ConvertedArrayToFlip=np.zeros((y,x),dtype=float)
    for i in range(len(ArrayToFlip)):
        ConvertedArrayToFlip[0,i]=ArrayToFlip[i,0]
        ConvertedArrayToFlip[1,i]=ArrayToFlip[i,1]
        ReallyBadConversion[i]=(ArrayToFlip[i,0],ArrayToFlip[i,1])
    return ReallyBadConversion


#WARNING! Its important to get Point A and B the correct way round
#if the first result doesnt look good - try switching them
def useF_to_test_point(FundamentalMatix, MatchingPntA,MatchingPntB,Threshold):
    #lets print corresponding point on both image - then test the formula to 
#see if we get ZERO (as x^T*F*x’=(0) ) and can prove we understand the math
#corresponding pair should be shown above - now lets see if the fundamental matrix formula gets us something close to ZERO
#formula = X'T * F * X' = 0
#PointX transform * FundamentalMatrix * COrrespondingPointX=0
#TODO this initialises the test matrices every time so is not efficient
    pointXY=(MatchingPntA)
    PointXY_Prime=(MatchingPntB)
    #i think the XY matrixs have to be in form "X,Y,1".. dunno why though/
    PointXY_Matrix = np.ones(shape=(3))
    PointXY_Prime_Matrix = np.ones(shape=(3))
    PointXY_Matrix[0]=pointXY[0]
    PointXY_Matrix[1]=pointXY[1]
    PointXY_Prime_Matrix[0]=PointXY_Prime[0]
    PointXY_Prime_Matrix[1]=PointXY_Prime[1]
    #transpose the prime point
    ###PointXY_Prime_Matrix_T=PointXY_Prime_Matrix.transpose()#this doesnt seem to be doing anything

    # convert to actual matrices maybe????
    #this step can be done using numpy matrix casting, or just dot products of basic arrays
    np_XYMatrix=np.matrix(PointXY_Matrix)
    np_XYMatrixPrime=np.matrix(PointXY_Prime_Matrix)
    np_FMatrix=np.matrix(FundamentalMatix)
    np_Error=(np_XYMatrixPrime*np_FMatrix)*np_XYMatrix.transpose()#make sure you multiply the matrices in the correct order
    #this should be close to zero according to formula
    return np_Error[0,0]





def TestF_All_Image(InputColourImg_ImageB,PointXY_of_ImageA,FundamentalMatrix):
    #brute force visualise epipolar line
    #in theory if we test a matched point of a pair, then for
    #the paired image test every pixel with the epipolar constraint
    #formula, the error plotted should be an epipolar line
    DrawImage=InputColourImg_ImageB.copy()
    DrawImage[:,:,:]=0
    Xpixel, Ypixel, ColourDim=DrawImage.shape
    #churn through each pixel and get the F - ouch!!
    TempErrorMatrix=np.ndarray((Xpixel,Ypixel),dtype=float)
    Xjump=1
    YJump=1
    #avoid blowing out detail with ceiling
    #0.5= fuzzy line, 0.01 = sharper, 0.001 = well defined line
    ErrorCeiling=Str_Common.FError_Failure
    
    for X in range (0,Xpixel,Xjump):
        for Y in range (0,Ypixel,YJump):
            #X and Y are wrong way around here!!
            Error=abs(useF_to_test_point(FundamentalMatrix,PointXY_of_ImageA,(Y,X),0))
            #DrawImage[X,Y][2]=Error
            #DrawImage[X,Y][1]=Error
            #DrawImage[X,Y][0]=Error
            if Error>ErrorCeiling:
                Error=ErrorCeiling
                
            TempErrorMatrix[X,Y]=Error
        print(str(round(X /Xpixel * 100)) + "%")
    
    #normalise the ndarray of errors
    #bad normalisation code TODO 
    #TempErrorMatrix=TempErrorMatrix+abs(TempErrorMatrix.min())
    #TempErrorMatrix=TempErrorMatrix-TempErrorMatrix.min()
    TempErrorMatrix=TempErrorMatrix/TempErrorMatrix.max()
    TempErrorMatrix=TempErrorMatrix*254
    
    #redraw back over image (this is pretty lazy code) TODO
    for X in range (0,Xpixel,Xjump):
        for Y in range (0,Ypixel,YJump):
            DrawImage[X,Y][2]=int(TempErrorMatrix[X,Y])
            DrawImage[X,Y][1]=int(TempErrorMatrix[X,Y])
            DrawImage[X,Y][0]=int(TempErrorMatrix[X,Y])
    
#    
#    LastColor=DrawImage[Xjump,YJump]
#    for X in range (0,Xpixel,1):
#        for Y in range (0,Ypixel,1):
#            if X%2=0:
#                
#            DrawImage[X,Y][2]=int(TempErrorMatrix[X,Y])
#            DrawImage[X,Y][1]=int(TempErrorMatrix[X,Y])
#            DrawImage[X,Y][0]=int(TempErrorMatrix[X,Y])
#                   
    
#    #lets make a smaller version of the image if we
#    #used STEPS in XY test loops
#    SmallImage=np.ndarray((round(Xpixel/Xjump),round(Ypixel/YJump)),dtype=float)
#    Xpixelsmall, Ypixelsmall=SmallImage.shape
#    for X in range (0,Xpixelsmall-1,1):
#        for Y in range (0,Ypixelsmall-1,1):
#            DrawImage[X,Y][2]=DrawImage[X*Xjump,Y*YJump][2]
#            DrawImage[X,Y][1]=DrawImage[X*Xjump,Y*YJump][1]
#            DrawImage[X,Y][0]=DrawImage[X*Xjump,Y*YJump][0]
#    
#    
    #super-imposed/blend input image with error matrix
    DrawImage=cv2.addWeighted(DrawImage,0.5,InputColourImg_ImageB,0.5,0)
    return(TempErrorMatrix,DrawImage)
    
    
    
def drawEpipolarlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    count=0
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        count=count+1
        if count>30:
            break
        color = tuple(np.random.randint(0,255,3).tolist())
        #color=(0,0,255)
        try:
            (x0,y0) = map(int, [0, -r[2]/r[1] ])
            (x1,y1) = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            
       # try:
            
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,4)
            img1 = cv2.circle(img1,tuple(pt1),15,color,-1)
            img2 = cv2.circle(img2,tuple(pt2),15,color,-1)
        except Exception as e:
            print(e)
            print("drawEpipolarlines exploded")
            #print ((x0,y0),(x1,y1))
   
        
    return img1,img2

def drawEpipolarline_UsingIndex(img1,img2,lines,pts1,pts2, Index):
    #Debug only code!!
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    count=0
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        
        if (abs(count-Index))<2:
            color = tuple(np.random.randint(0,255,3).tolist())
            #color=(0,0,255)
            (x0,y0) = map(int, [0, -r[2]/r[1] ])
            (x1,y1) = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            
            try:
                
                img1 = cv2.line(img1, (x0,y0), (x1,y1), color,4)
                img1 = cv2.circle(img1,tuple(pt1),15,color,-1)
                img2 = cv2.circle(img2,tuple(pt2),15,color,-1)
            except Exception as e:
                print(e)
                print("drawEpipolarlines exploded")
                print ((x0,y0),(x1,y1))
       
        count=count+1   
    return img1,img2

def DisplayWinImage_Simple(l_Text,l_image, Wait=True):
    
    if l_image is None: return
    CopyOfImage=cv2.resize(l_image.copy(),(1000,1000))
    cv2.putText(CopyOfImage, str(l_Text), (10,80), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,0,255),thickness=2)
    while True: 
        cv2.imshow("",CopyOfImage)
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
            #for some reason
            break
        else:
            break
    if Wait==True: time.sleep(0.8)


def DisplayWinImage_Sparse(l_Text,l_image, InputParameters):
    
    global ParamsObj
    global ImageCounter
    if l_image is None: return
    CopyOfImage=cv2.resize(l_image.copy(),(800,800))
    #CopyOfImage=cv2.resize(l_image.copy(),(400,400))
    cv2.putText(CopyOfImage, str(InputParameters.InteralCounter1), (10,50), cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=2)
    cv2.putText(CopyOfImage, str(l_Text), (10,80), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,0,255),thickness=2)
    InputParameters.InteralCounter1=InputParameters.InteralCounter1+1
    Savestring=Paths_Common.FAILS_path +InputParameters.ProcessName + str(InputParameters.InteralCounter1) + ".jpg"
    cv2.imwrite(Savestring,CopyOfImage)
    while True: 
        cv2.imshow(InputParameters.ProcessName,CopyOfImage)
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
            #for some reason
            break
        else:
            break
    if InputParameters.LiveImageWait==True: time.sleep(0.8)
    
def DrawSingleLine(InputImage, GeneralFormOfLine):
    print(GeneralFormOfLine)
    #using ax+by+c=0 from [a,b,c] form returned from cv2.computeCorrespondEpilines
    WorkingImage=InputImage.copy()
    height, width, channels = WorkingImage.shape
    a=GeneralFormOfLine[0]
    b=GeneralFormOfLine[1]
    c=GeneralFormOfLine[2]
    X1=0
    X2=width
    #first point will be far left of image X=0
    Y1= ((a*X1)+c)/(b*-1)
    #second point far right of image X=width
    Y2= ((a*X2)+c)/(b*-1)
    WorkingImage = cv2.line(WorkingImage, (int(X1),int(Y1)), (int(X2),int(Y2)), (100,50,50),4)
    return WorkingImage

def GetMask_FromEpiline(InputImage, GeneralFormOfLine,Width):
    #creates a mask of the epiline
    
    #using ax+by+c=0 from [a,b,c] form returned from cv2.computeCorrespondEpilines
    height, width, channels = InputImage.shape
    ChannelWidth=int(Width/2)
    Offset_frm_Top=0
    Offset_frm_Bottom=height
    Offset_frm_Left=0
    Offset_frm_Right=width
    mask_image = np.zeros((height,width), np.uint8)#blank grayscale image for mask
    a=GeneralFormOfLine[0]
    b=GeneralFormOfLine[1]
    c=GeneralFormOfLine[2]
    #get slope intercept form so we can see if line is vertical or horizontal
    #to stop us blowing anything up with perpendicular lines
    #y = (-a/b)x -  (c/b)
    #mSlope=A/B
    mSlope=-a/b#= tan theta
    #solve for angle - convert from rad to deg
    Theta_LineAngle=(math.atan(mSlope)) * (180/math.pi)
    #if we have a vertical line this can blow up the Y values and give
    #weird results
    #must be a smarter way to handle this
    #TODO
    if abs(Theta_LineAngle)<45:#line is more or less horizontal
        X1=0
        X2=width
        #first point will be far left of image X=0
        Y1= ((a*X1)+c)/(b*-1)
        #second point far right of image X=width
        Y2= ((a*X2)+c)/(b*-1)
    else:#line is closer to being vertical
        Y1=0
        X1=((b*Y1)+c)/-a
        Y2=height
        X2=((b*Y2)+c)/-a
    
    
    Offset_frm_Top=int(min(Y1,Y2)-ChannelWidth)
    Offset_frm_Bottom=int(max(Y1,Y2)+ChannelWidth)
    Offset_frm_Left=int(min(X1,X2)-ChannelWidth)
    Offset_frm_Right=int(max(X1,X2)+ChannelWidth)
    #limit offsets to image boundary
    Offset_frm_Top=max(Offset_frm_Top,0)
    Offset_frm_Bottom=min(Offset_frm_Bottom,height)
    Offset_frm_Left=max(0,Offset_frm_Left)
    Offset_frm_Right=min(Offset_frm_Right,width)
    
    
    X1=int(X1)
    X2=int(X2)
    Y1=int(Y1)
    Y2=int(Y2)
    
    mask_image = cv2.line(mask_image, (int(X1),int(Y1)), (int(X2),int(Y2)), (1),Width)
    
    mask_image_Cropped=mask_image[Offset_frm_Top:Offset_frm_Bottom,Offset_frm_Left:Offset_frm_Right]
    plt.imshow(mask_image_Cropped)
    
    
    return mask_image,mask_image_Cropped
def GetCroppedEpiline_MaskedImage(InputImage, GeneralFormOfLine,Width):
    #returns a cropped version of the input image - 
    #cropped to extremes of epiline
    
    #using ax+by+c=0 from [a,b,c] form returned from cv2.computeCorrespondEpilines
    height, width, channels = InputImage.shape
    ChannelWidth=int(Width/2)
    #OffsetTuple=collections.namedtuple('hai','Offset_frm_Top Offset_frm_Bottom Offset_frm_Left Offset_frm_Right')
    #l_OffsetTuple=OffsetTuple('omg',0,height,0,width)
    
    ImageOffsetsTuple = collections.namedtuple('Animal', 'Offset_frm_Top Offset_frm_Bottom Offset_frm_Left Offset_frm_Right X1 Y1 X2 Y2')
    
    Offset_frm_Top=0
    Offset_frm_Bottom=height
    Offset_frm_Left=0
    Offset_frm_Right=width
    mask_image = np.zeros((height,width), np.uint8)#blank grayscale image for mask
    a=GeneralFormOfLine[0]
    b=GeneralFormOfLine[1]
    c=GeneralFormOfLine[2]
    #get slope intercept form so we can see if line is vertical or horizontal
    #to stop us blowing anything up with perpendicular lines
    #y = (-a/b)x -  (c/b)
    #mSlope=A/B
    mSlope=-a/b#= tan theta
    #solve for angle - convert from rad to deg
    Theta_LineAngle=(math.atan(mSlope)) * (180/math.pi)
    #if we have a vertical line this can blow up the Y values and give
    #weird results
    #must be a smarter way to handle this
    #TODO
    if abs(Theta_LineAngle)<45:#line is more or less horizontal
        X1=0
        X2=width
        #first point will be far left of image X=0
        Y1= ((a*X1)+c)/(b*-1)
        #second point far right of image X=width
        Y2= ((a*X2)+c)/(b*-1)
    else:#line is closer to being vertical
        Y1=0
        X1=((b*Y1)+c)/-a
        Y2=height
        X2=((b*Y2)+c)/-a
    
    
    Offset_frm_Top=int(min(Y1,Y2)-ChannelWidth)
    Offset_frm_Bottom=int(max(Y1,Y2)+ChannelWidth)
    Offset_frm_Left=int(min(X1,X2)-ChannelWidth)
    Offset_frm_Right=int(max(X1,X2)+ChannelWidth)
    #limit offsets to image boundary
    Offset_frm_Top=max(Offset_frm_Top,0)
    Offset_frm_Bottom=min(Offset_frm_Bottom,height)
    Offset_frm_Left=max(0,Offset_frm_Left)
    Offset_frm_Right=min(Offset_frm_Right,width)
    
    
    X1=int(X1)
    X2=int(X2)
    Y1=int(Y1)
    Y2=int(Y2)
    
    ImageOffsets = ImageOffsetsTuple(Offset_frm_Top=Offset_frm_Top, 
                                     Offset_frm_Bottom=Offset_frm_Bottom, 
                                     Offset_frm_Left=Offset_frm_Left,
                                     Offset_frm_Right=Offset_frm_Right,
                                     X1=X1,
                                     Y1=Y1,
                                     X2=X2,
                                     Y2=Y2)
    
    #not that efficient to draw the line on the full-size image
    #TODO fix this
    mask_image = cv2.line(mask_image, (int(X1),int(Y1)), (int(X2),int(Y2)), (1),Width)
    #mask_image = cv2.line(mask_image, (int(X1),int(Y1)), (int(X2),int(Y2)), (0),1)
    mask_image_Cropped=mask_image[Offset_frm_Top:Offset_frm_Bottom,Offset_frm_Left:Offset_frm_Right]
    OriginalImage_Cropped=InputImage[Offset_frm_Top:Offset_frm_Bottom,Offset_frm_Left:Offset_frm_Right]
    OriginalImage_Cropped=SuperImposeMaskOntoImage(OriginalImage_Cropped,mask_image_Cropped)
    
    
    return mask_image,OriginalImage_Cropped,ImageOffsets


def SuperImposeMaskOntoImage(_3ChannelImage, Mask):
    
    # Select image or mask according to condition array
    OutputImage=_3ChannelImage.copy()
    OutputImage[:,:,0]=OutputImage[:,:,0] * Mask
    OutputImage[:,:,1]=OutputImage[:,:,1] * Mask
    OutputImage[:,:,2]=OutputImage[:,:,2] * Mask
    
    return OutputImage

def EnhancedLocal_Matcher(MainImage=None,
                          GeneralFormLine=None,
                          ImageOff_sets=None,
                          ChannelWidth=None):
    TempImage=MainImage.copy()
    
    
    
    #aggregate cost of template around epiline in a custom shape
    #enhanced-local error reduction hopefully
    
    #dont really need this
    #TODO remove full size image line equation
    #####Line is from full sized image not cropped
    a=GeneralFormLine[0]
    b=GeneralFormLine[1]
    c=-GeneralFormLine[2]
    #get slope intercept form so we can see if line is vertical or horizontal
    #to stop us blowing anything up with perpendicular lines
    #y = (-a/b)x -  (c/b)
    #mSlope=A/B
    mSlope=-a/b#= tan theta
    #solve for angle - convert from rad to deg
    Theta_LineAngle=(math.atan(mSlope)) * (180/math.pi)
    ###Cropped Image line
    mCropped=(ImageOff_sets.Y2-ImageOff_sets.Y1)/(ImageOff_sets.X2-ImageOff_sets.X1)
    #y=mx+c
    #y-mX=c
    cCropped=(ImageOff_sets.Y2-ImageOff_sets.Offset_frm_Top)-(mCropped*(ImageOff_sets.X2-ImageOff_sets.Offset_frm_Left))
    
    MaxIndexY=0
    MinIndexY=0
    MaxIndexX=0
    MinIndexX=0
    ScanRange=int(ChannelWidth/2)
    AreaRange=4
    EnergyArray=None
    i=0
    if abs(Theta_LineAngle)<45:#line is more or less horizontal
        Ypixel=0
        EnergyArray=np.zeros(TempImage.shape[1]-ScanRange-ScanRange)
        for ScanXpixel in range (ScanRange,TempImage.shape[1]-ScanRange):
            Ypixel= (mCropped*ScanXpixel)+cCropped
            MeanValue, Empty=SumAreaOfImage(TempImage,int(ScanXpixel),int(Ypixel),AreaRange)
            EnergyArray[i]=MeanValue
            i=i+1
            
            #cv2.circle(TempImage, (int(ScanXpixel), int(Ypixel)), 1, int(100),1)
            
        #min/maxes in X axis
        MaxIndexX=np.argmax(EnergyArray)+ScanRange
        MinIndexX=np.argmin(EnergyArray)+ScanRange
        MaxIndexY=(mCropped*MaxIndexX)+cCropped
        MinIndexY=(mCropped*MinIndexX)+cCropped
        
            
    else:#line is closer to being vertical
        Xpixel=0
        EnergyArray=np.zeros(TempImage.shape[0]-ScanRange-ScanRange)
        for ScanYpixel in range (ScanRange,TempImage.shape[0]-ScanRange):
            #y=mx+c
            #y-c=mx
            #(y-c)/m=x
            Xpixel= (ScanYpixel-cCropped)/mCropped
            MeanValue, Empty=SumAreaOfImage(TempImage,int(Xpixel),int(ScanYpixel),AreaRange)
            EnergyArray[i]=MeanValue
            i=i+1
            #cv2.circle(TempImage, (int(Xpixel), int(ScanYpixel)), 1, 100,1)
     #min/maxes in X axis
        MaxIndexY=np.argmax(EnergyArray)+ScanRange
        MinIndexY=np.argmin(EnergyArray)+ScanRange
        MaxIndexX=(MaxIndexY-cCropped)/mCropped
        MinIndexX=(MinIndexY-cCropped)/mCropped
           
    
    TempImage=cv2.normalize(TempImage,TempImage, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.circle(TempImage, (int(MinIndexX), int(MinIndexY)), 10, 255,2)
        
    
    
    PythonSucksY=np.ndarray(1)
    PythonSucksX=np.ndarray(1)
    PythonSucksY[0]=int(MinIndexY)
    PythonSucksX[0]=int(MinIndexX)
    MinimumCoordinate=(PythonSucksY, PythonSucksX)
    return TempImage,MinimumCoordinate
        

    

def MatchTemplate_GetXYs(TemplateImage,MainImage,Result_size=1,TypeNormalising=Str_Common.EMPTY,threshold=0.8,Coord_4_debug=None,GeneralFormOfLine=None,ImageOff_sets=None,ChannelWidth=None):
    #convolves template over main image to find matches
    #can return top match (no threshold) or all matches above a match threshold
    
    #use opencv template match DFT - works on colour and grayscale
    #TypeNormalising doesnt do anything for now but is there if
    #we need to modify algorithm
    ErrorLog=[]
    ErrorLog.append("MatchTemplate_GetXYs")
    loc=None
    EpilineArea=None
    try:
    
    

    
        
        if True==True:                                                                     
            if TemplateImage is None:
                ErrorLog.append("TemplateImage is None")
                print(ErrorLog)
                return False,None,None,Str_Common.DenseFailsCode_NoImages,None
            if MainImage is None:
                ErrorLog.append("MainImage is None")
                print(ErrorLog)
                return False,None,None,Str_Common.DenseFailsCode_NoImages,None
            
            h, w, channels = TemplateImage.shape
            res=None
            ErrorLog.append("Cropped Mask shape is : " + str(MainImage.shape))
            ErrorLog.append("Template Image shape is : " + str(TemplateImage.shape))
            
            if TypeNormalising==Str_Common.EMPTY:
                ErrorLog.append("Error - normalising method empty in NCC!")
                return False,None,None,Str_Common.DenseFailsCode_NoNorming,None
            if TypeNormalising==Str_Common.NCC_TM_CCOEFF_NORMED:
                ErrorLog.append("if TypeNormalising==Str_Common.NCC_TM_CCOEFF_NORMED")
                res = cv2.matchTemplate(MainImage,TemplateImage,cv2.TM_CCOEFF_NORMED)
            if TypeNormalising==Str_Common.NCC_TM_SQDIFF_NORMED:
                ErrorLog.append("if TypeNormalising==Str_Common.NCC_TM_SQDIFF_NORMED")
                res = cv2.matchTemplate(MainImage,TemplateImage,cv2.TM_SQDIFF_NORMED)
            
            
            
            
            if Result_size<2:#1 result
                #can also use this but the numpy thing below works just as well
                #refusing to work - do it yourself .______.
                #minVal,maxVal,minLoc,maxLoc = cv2.MinMaxLoc(res)
                ErrorLog.append("Result size <2")
                if TypeNormalising==Str_Common.NCC_TM_CCOEFF_NORMED:#Maximum response
                    ErrorLog.append("NCC_TM_CCOEFF_NORMED getting loc")
                    loc = np.where( res >= res.max())#just get highest rated match - but leave in flexibility
                    #check biggest response is greater or equals threshold or throw away
                    ErrorLog.append("loc: " + str(len(loc)))
                    ErrorLog.append("checking threshold")
                    ErrorLog.append("length Loc=" + str(len(loc)))
                    ErrorLog.append("loc[0] " + str(loc[0]))
                    ErrorLog.append("loc[1] " + str(loc[1]))
                    #we dont know if its just one result or a few - so make sure this can handle either case
                    if res[int(loc[0][0]),int(loc[1][0])]<threshold:
                        return False,None,None,Str_Common.DenseFailsCode_NoThreshold,None
                    
                    
                if TypeNormalising==Str_Common.NCC_TM_SQDIFF_NORMED:#Minimum response
                    ErrorLog.append("NCC_TM_SQDIFF_NORMED getting loc")
                    loc = np.where( res <= res.min())#just get highest rated match - but leave in flexibility
                    #check biggest response is greater or equals threshold or throw away
                    ErrorLog.append("loc: " + str(len(loc)))
                    ErrorLog.append("checking threshold")
                    ErrorLog.append("length Loc=" + str(len(loc)))
                    ErrorLog.append("loc[0] " + str(loc[0]))
                    ErrorLog.append("loc[1] " + str(loc[1]))
                    #we dont know if its just one result or a few - so make sure this can handle either case
                    
                    if res[int(loc[0][0]),int(loc[1][0])]>(1-threshold):
                        return False,None,None,Str_Common.DenseFailsCode_NoThreshold,None
                    #EpilineArea,MinCoordinate=EnhancedLocal_Matcher(res,GeneralFormOfLine,ImageOff_sets,ChannelWidth=ChannelWidth)
                    #loc=MinCoordinate
                
                
            else:
                #print("LIES")
                ErrorLog.append("Result size >1 - shouldnt be handling this yet")
                #loc = np.where( (res >= threshold))
                print(ErrorLog)
                return False,None,None,Str_Common.DenseFailsCode_ResSize,None
            
            #if loc is None:
            #    return False,None,None
            
            Ih8Tuples=[]
            ErrorLog.append("for pt in zip(*loc[::-1]):")
            for pt in zip(*loc[::-1]):
                ErrorLog.append("in loop")
                Ih8Tuples.append((int(pt[0] + (w/2)),int(pt[1]+ (h/2))))
                #cv2.rectangle(TempMatchMain_Col, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
                #cv2.circle(TempMatchMain_Col, (int(pt[0] + (w/2)), int(pt[1]+ (h/2))), 3, 255,2)
            
            #debug image
            res=Normalize_and_round(res)
            #print(ErrorLog)
            return True, Ih8Tuples,res,Str_Common.DenseFailsCode_OK,EpilineArea
            
    except Exception as e:
           ErrorLog.append(e)
           ErrorLog.append("At coordinate " + str(Coord_4_debug))
           print(ErrorLog)
           return False,None,None,Str_Common.DenseFailsCode_Except,None
       

def Normalize_and_round(InputImage_Channel):
    Normalised=InputImage_Channel.copy()
    Normalised=cv2.normalize(Normalised,Normalised, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    Normalised=np.round(Normalised)
    return Normalised

def Correct_To_Epiline(X=0, Y=0, Epiline_GeneralForm=None,Threshold=0):
    #Corrects coordinates to nearest position on epiline
#    might not do enough to be worth the cycles!
    a_eq=Epiline_GeneralForm[0]
    b_eq=Epiline_GeneralForm[1]
    c_eq=-Epiline_GeneralForm[2]
    #ax+by=c
    #ax=c-by
    #x=(c-by)/a
    
    #ax+by=c
    #ax-c=-by
    #(ax-c)/-b=y
    
    
    mSlope=-a_eq/b_eq
    mx_plus_C=(mSlope*X)+c_eq
    
    if abs(mx_plus_C-Y)>Threshold:
        #average onto epiline
        Y_new=((a_eq*X)-c_eq)/-b_eq
        X_new= (c_eq-(b_eq*Y))/a_eq
        
        X_av=int((X+X_new)/2)
        Y_av=int((Y+Y_new)/2)
        return X_av,Y_av
    
    return X,Y



def MatchPoints_UsingDenseMethod(ImageAPoints,
                           ImageBTestPoints,
                           ImageA,
                           ImageB,
                           DisplayFunction4Debug,
                           DisplayWindowA_4Debug,
                           DisplayWindowB_4Debug,
                           EpipolarLinesB,
                           InputArgumentObject):
    
    ChannelWidth=InputArgumentObject.DenseNCC_ChannelWidth
    AreaSpan=InputArgumentObject.DenseNCC_TemplateSize
    Threshold=InputArgumentObject.DenseNCC_AcceptThreshold
    
    
    #ImageAPoints is input points from image A - either scanning
    #each pixel or input Feature Pairs for testing
    
    #ImageBTestPoints can be populated to test reconstruction
    #if this is populated - results found using dense recon methods
    #will be tested against these to find any error
    
    #FundamentalMatrix- relation between A and B images
    
    #ImageA stereo origin image (colour)
    #ImageB stereo partner image (colour)
    
    #DisplayFunction4Debug - pass in the debug display window if
    #you want debug data
    
    #DisplayWindowA_4Debug - NAMED WINDOW from stereo image A 
    #DisplayWindowb_4Debug - NAMED WINDOW from stereo image b 
    
    #EpipolarLinesB - Epipolar lines which corresponding with pixels
    #from Image A
    
    #ChannelWidth is width in PIXELS of epiline mask - 14 seems OK
    #AreaSpan is size of template that will be used as convolving kernel

    
    
                       
    FailedDenseCount=0
    TestMode=True
    if ImageAPoints is None:
        raise ValueError("MatchPoints_UsingDenseMethod: No Input ImageAPoints")
    if ImageBTestPoints is None:
        TestMode=False
    if ImageA is None:
        raise ValueError("MatchPoints_UsingDenseMethod: No Input ImageA")
    if ImageB is None:
        raise ValueError("MatchPoints_UsingDenseMethod: No Input ImageB")
    if DisplayFunction4Debug is None:
        TestMode=False
    if DisplayWindowA_4Debug is None:
        TestMode=False
    if DisplayWindowB_4Debug is None:
        TestMode=False
    if EpipolarLinesB is None:
        raise ValueError("MatchPoints_UsingDenseMethod: No Input EpipolarLinesB")
    
    
    if TestMode==True:
        if len(ImageAPoints)!=len(ImageBTestPoints):
            raise ValueError("MatchPoints_UsingDenseMethod: Test Mode:len(ImageAPoints)<>len(ImageBTestPoints) ")
    
    
    if TestMode==True:
        #IF TEST POINTS EXIST (requirements - test points A, B and
        #fundamental matrix)-
        #We can create test images to visualise Dense method results
        imgTestMatchesA=ImageA.copy()
        imgTestMatchesB=ImageB.copy()
        # really cheap way to let us draw colour markers on B&W version of image
        imgTestMatchesA=cv2.cvtColor(imgTestMatchesA,cv2.COLOR_BGR2GRAY)
        imgTestMatchesB=cv2.cvtColor(imgTestMatchesB,cv2.COLOR_BGR2GRAY)
        imgTestMatchesA=cv2.cvtColor(imgTestMatchesA,cv2.COLOR_GRAY2BGR)
        imgTestMatchesB=cv2.cvtColor(imgTestMatchesB,cv2.COLOR_GRAY2BGR)
        for i in range (len(ImageAPoints)):
            FRadius=8
            FThickness=5
            point=(ImageAPoints[i][0],ImageAPoints[i][1])
            pointprime=(ImageBTestPoints[i][0],ImageBTestPoints[i][1])
            cv2.circle(img=imgTestMatchesA,center=point,radius=FRadius,color=Str_Common.Green,thickness=FThickness)
            cv2.circle(img=imgTestMatchesB,center=pointprime,radius=FRadius,color=Str_Common.Green,thickness=FThickness)



    #initialise array to hold dense points found in image B
    DensePointsB=np.zeros(ImageAPoints.shape)
    #initialise array to show what 2 points have passed/failed
    DensePoints_Success=np.zeros(ImageAPoints.shape,dtype=bool)
    FtrPt_V_NCC_Error=[]
    #loop around all feature points and print NCC equivalent to eyeball error
    for i in range (len(ImageAPoints)):
        
        #if i % 200==0:
            #print(round(i/len(ImageAPoints)*100), "%")
        #create mask of image B with the epiline corresponding to point on image A (corresponding
        #epilines are in EpipolarLinesB structure)
        CheckMaskB,CroppedMask,ImageOffsets=GetCroppedEpiline_MaskedImage(ImageB,EpipolarLinesB[i][-1,:],Width=ChannelWidth)
        #this line is done in GetCroppedEpiline_MaskedImage
        #CheckMaskB=SuperImposeMaskOntoImage(ImageB,CheckMaskB)
        #sample an area of Image A around the feature point
        Colour, TemplateImageA=SampleAreaOfImage(ImageA,ImageAPoints[i][0],ImageAPoints[i][1],AreaSpan=AreaSpan)
        
#        DisplayFunction4Debug("Mask Error: ImageB",ImageB,DisplayWindowB_4Debug)
#        DisplayFunction4Debug("Cropped mask too small",TemplateImageA,DisplayWindowA_4Debug)
#        DisplayFunction4Debug("Cropped mask too small",CroppedMask,DisplayWindowB_4Debug)
            
        #make sure cropped mask is big enough to run template through
        if (int(CroppedMask.shape[0])<int(TemplateImageA.shape[0])) or (int(CroppedMask.shape[1])<int(TemplateImageA.shape[1])):
            #print("Bad mask size")
            if TestMode==True:
                DisplayFunction4Debug("Cropped mask too small",TemplateImageA,DisplayWindowA_4Debug)
                DisplayFunction4Debug("Cropped mask too small",CheckMaskB,DisplayWindowB_4Debug)
                BrokenMask=ImageA.copy()
                cv2.circle(BrokenMask, ((ImageAPoints[i][0]),(ImageAPoints[i][1])), 8, (0,0,255),5)
                DisplayFunction4Debug("Cropped mask too small",BrokenMask,DisplayWindowB_4Debug)
            
            continue    
           
        
        
        CCSuccess,MatchPoints,CC_Image,FailCode,DebugImage_Epiline=MatchTemplate_GetXYs(TemplateImageA,
                                CroppedMask,
                                Result_size=1,
                                TypeNormalising =InputArgumentObject.DenseNCC_ResultType,
                                threshold=Threshold,
                                Coord_4_debug=ImageAPoints[i],
                                GeneralFormOfLine=EpipolarLinesB[i][-1,:],
                                ImageOff_sets=ImageOffsets,
                                ChannelWidth=ChannelWidth)
        #make sure no weird NCC found at boundary of image
        if CCSuccess==True:
            #DisplayFunction4Debug("Cropped mask too small",CC_Image,DisplayWindowA_4Debug)
        
            
            MatchPointsX=MatchPoints[0][0]+ImageOffsets.Offset_frm_Left
            MatchPointsY=MatchPoints[0][1]+ImageOffsets.Offset_frm_Top
            #lets correct XY point in case its off the theoretical epiline
            #might give worse results
            #MatchPointsX,MatchPointsY=Correct_To_Epiline(MatchPointsX,MatchPointsY,EpipolarLinesB[i][-1,:],1)
     
            
            
            
            if MatchPointsX < 1:
                CCSuccess=False
                #print("1")
            if MatchPointsY < 1:
                CCSuccess=False
                #print("2")
            if MatchPointsX >= ImageB.shape[1]:
                CCSuccess=False
                #print("3")
            if MatchPointsY >= ImageB.shape[0]:
                CCSuccess=False
                #print("4")
        
        
#        if CCSuccess==False:
#            print("5")
        if CCSuccess==True:
            
            #add offsets if cropping image to speed up NCC
            #MatchPointsX=MatchPoints[0][0]+ImageOffsets.Offset_frm_Left
            #MatchPointsY=MatchPoints[0][1]+ImageOffsets.Offset_frm_Top
            DensePointsB[i]=(MatchPointsX,MatchPointsY)
            DensePoints_Success[i]=1
            if TestMode==True:
                imgTestMatchesB = cv2.line(imgTestMatchesB, ((MatchPointsX),(MatchPointsY)), (ImageBTestPoints[i][0],ImageBTestPoints[i][1]), (200,50,200),2)
                cv2.circle(imgTestMatchesB, ((MatchPointsX),(MatchPointsY)), 2, (0,0,255),2)
                Distance_Err=((MatchPointsX)-ImageBTestPoints[i][0])*((MatchPointsX)-ImageBTestPoints[i][0])
                Distance_Err=Distance_Err+((MatchPointsY)-ImageBTestPoints[i][1])*((MatchPointsY)-ImageBTestPoints[i][1])
                Distance_Err=np.sqrt(Distance_Err)
                FtrPt_V_NCC_Error.append(np.round(Distance_Err,4))
                #debug display images
                CC_Image=cv2.normalize(CC_Image,CC_Image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.circle(CroppedMask, ((MatchPoints[0][0]),(MatchPoints[0][1])), 8, (0,0,255),5)
                
                CC_Image=cv2.cvtColor(CC_Image,cv2.COLOR_GRAY2BGR)
                CC_Image=cv2.resize(CC_Image,(CroppedMask.shape[1],CroppedMask.shape[0]))
                cv2.circle(CC_Image, ((MatchPoints[0][0]),(MatchPoints[0][1])), 8, (0,0,255),2)
                
                StackDebugImages = np.concatenate((CroppedMask, CC_Image), axis=0)
                StackDebugImages=cv2.normalize(StackDebugImages,StackDebugImages, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                
                
                DisplayFunction4Debug("Cropped mask epiline", DebugImage_Epiline, DisplayWindowB_4Debug)
                DisplayFunction4Debug("Cropped mask", CroppedMask, DisplayWindowB_4Debug)
                DisplayFunction4Debug("TemplateImageA imageA", TemplateImageA,DisplayWindowA_4Debug)
                DisplayFunction4Debug("masked epiline and NCC'd epiline", StackDebugImages,DisplayWindowB_4Debug)
        else:
            #set broken points
            #is this a good idea?
            FailedDenseCount=FailedDenseCount+1
            DensePoints_Success[i]=0
            #just set it to something for now so its not empty
            DensePointsB[i]=(0.123456,0.123456)
            #print("Failed at point: " + str(ImageAPoints[i]))
            
            
    
    #after loop leave the final image on the debug windows
    #of test points on A, pair partners on B and the Dense reconstructed A points on B        
    if TestMode==True:
        DisplayFunction4Debug("Pod2Image_col_CompareCC_FM", imgTestMatchesA,DisplayWindowA_4Debug)
        DisplayFunction4Debug("Pod2Image_col_CompareCC_FM", imgTestMatchesB,DisplayWindowB_4Debug)
    
    #print("Outlier Filtered Error " + str(reject_outliers(FtrPt_V_NCC_Error)))
    print("Dense Recon pts removed (errors/tolerance): " + str(np.round(((FailedDenseCount/len(ImageAPoints))*100))) + "% of " + str(len(ImageAPoints)))
    
    
    return FtrPt_V_NCC_Error, DensePointsB,DensePoints_Success

def reject_outliers(data, m = 2.):
    #Not proved to work
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]
def GetCameraCoords_FromProjectionMatrix(ProjectionMatrix,useOpenCVmethod=False):
    
    #two methods to decompose projection matrix
    
    #this returns a 1*3 vector
    #PADS with 1 - for translation
    #padding with zero is for direction
    if useOpenCVmethod==False:
        K, R, t = decomposeP(ProjectionMatrix)
        
        C_Padded = np.zeros(shape=(4,1))
        C_Padded[0]=t[0]
        C_Padded[1]=t[1]
        C_Padded[2]=t[2]
        C_Padded[3]=1
        
        return C_Padded
    else:
    #this returns a 1*4 vector
        Kmat_cv2,Rmat_cv2,C_Or_Tmat_cv2,RotX_cv2,RotY_cv2,RotZ_cv2,eulerAngles_cv2=cv2.decomposeProjectionMatrix(ProjectionMatrix)
        return C_Or_Tmat_cv2

def GetRelativeRotation(Camera1_transform,Camera2_transform):
    #formula may be R12=R1*R2transform
    
    #convert to numpy matrices
    TempCam2=np.matrix(Camera2_transform)
    TempCam1=np.matrix(Camera1_transform)
    #transpose 
    TempCam2_Transform=TempCam2.transpose()
    #multiply
    Result=TempCam1*TempCam2_Transform
    
    return Result


def GetRelativeTranslation(Camera1_translation,Camera2_translation):
    #gets relative vector? translation? between two 3d points
    #should be subtration - but keep an eye on anything coming in thats NORMALISED
    
    
    Result=Camera1_translation-Camera2_translation
    return Result
    

def decomposeP(P):
    M = P[0:3,0:3]
    Q = np.eye(3)[::-1]
    P_b = Q @ M @ M.T @ Q
    K_h = Q @ np.linalg.cholesky(P_b) @ Q
    K = K_h / K_h[2,2]
    A = np.linalg.inv(K) @ M
    l = (1/np.linalg.det(A)) ** (1/3)
    R = l * A
    t = l * np.linalg.inv(K) @ P[0:3,3]
    return K, R, t

def GetUnrealEngine_FriendlyTransforms(intRotationMatrix, intTranslationVector):
    #convert to quarterions for unreal engine to display positions
    qw= (math.sqrt(intRotationMatrix[0,0]+intRotationMatrix[1,1]+intRotationMatrix[2,2]+1))/2
    qx = (intRotationMatrix[2,1] - intRotationMatrix[1,2])/( 4 *qw)
    qy = (intRotationMatrix[0,2] - intRotationMatrix[2,0])/( 4 *qw)
    qz = (intRotationMatrix[1,0] - intRotationMatrix[0,1])/( 4 *qw)
    lstr_Translation=("{" + str(round(float(intTranslationVector[0]),3)) + "," + str(round(float(intTranslationVector[1]),3)) + "," +str(round(float(intTranslationVector[2]),3)) + "},")
    lstr_Rotation=("{" + str(round(qx,2)) + "," + str(round(qy,2)) + "," + str(round(qz,2)) + "," + str(round(qw,2)) + "}")
    return lstr_Translation,lstr_Rotation





def RecomposeProjectionMatrices(ProjectionMatrix_CamA,ProjectionMatrix_CamB):
    #decompose projection matrices and find RELATIVE rotation/translation of
    #camera B to camera A then recompose
    
    #break down Projection matrices into :
    #Camera intrinsics (focal length distortion skew)
    #rotation (relative to origin)
    #translation (relative to origin)
    
    Cam1_Kmat_cv2, Cam1_Rmat_cv2 , Cam1_C_Or_Tmat_cv2 = decomposeP(ProjectionMatrix_CamA)
    Cam2_Kmat_cv2, Cam2_Rmat_cv2 , Cam2_C_Or_Tmat_cv2 = decomposeP(ProjectionMatrix_CamB)
    
    
    #get relative rotation of camera 2 compared to camera 1
    Cam12_RelativeRotation=GetRelativeRotation(Cam2_Rmat_cv2,Cam1_Rmat_cv2)
    Cam12_RelativeRotation=Cam12_RelativeRotation
    
    #now get position of camera 2 relative to camera 1
    #this might be inverted - keep an eye on it
    RelativeTransform=GetRelativeTranslation(Cam1_C_Or_Tmat_cv2,Cam2_C_Or_Tmat_cv2)
    
    #recalculate P for camera A - but set translation and rotation to ZERO
    #so effectively this is now the origin
    Pmatrix_manualA=FindProjectionMatrix_frmStereoCal(CameraMatrix=Cam1_Kmat_cv2,
                                                                   RotationMatrix=np.eye(3, k=0),
                                                                   TranslationVector=np.array([0, 0, 0,1]),
                                                                   AlternativeMethod=True)
    
    #recalculate P for camera B using relative translation and rotation WRT
    #camera A
    Pmatrix_manualB=FindProjectionMatrix_frmStereoCal(CameraMatrix=Cam2_Kmat_cv2,
                                                                   RotationMatrix=Cam12_RelativeRotation,
                                                                   TranslationVector=RelativeTransform,
                                                                   AlternativeMethod=True)
   
    
    
    
    
    intDebug=False
    if intDebug==True:
        ########################
        #check positional stuff#
        ########################
        Cam1_Kmat_cv2, Cam1_Rmat_cv2 , Cam1_C_Or_Tmat_cv2 = decomposeP(Pmatrix_manualA)
        Cam2_Kmat_cv2, Cam2_Rmat_cv2 , Cam2_C_Or_Tmat_cv2 = decomposeP(Pmatrix_manualB)
        
         #convert transform for unreal engine visualisation
        #print("Cam 1")
        lstr_Translation, lstr_Rotation = GetUnrealEngine_FriendlyTransforms(Cam1_Rmat_cv2,Cam1_C_Or_Tmat_cv2)
        #print(lstr_Translation, lstr_Rotation)
        Cam1=lstr_Translation+ lstr_Rotation
        #print("Cam 2")
        lstr_Translation, lstr_Rotation = GetUnrealEngine_FriendlyTransforms(Cam2_Rmat_cv2,Cam2_C_Or_Tmat_cv2)
        #print(lstr_Translation, lstr_Rotation)
        Cam2=lstr_Translation+ lstr_Rotation
        #input("Press Enter to continue...")
        Cam2=Cam2

    return Pmatrix_manualA,Pmatrix_manualB



def CalculateFundamental_FromProjections(ProjectionMatrixA_To_be_Origin,ProjectionMatrixCamB_to_be_Relative):
    #takes two projection matrices
    #matrix A will be zero'd (position and rotation)
    #matrix B will have be recalculated to relative translation and rotation
    #will return the fundamental matrix between cameras - 
    #not anything to do with other cameras in 3D space!
    
    #CUSTOM Fundamental matrix
    #F=[P'C]x   P'P^+
    
    #P'P^+
    #Multiply ProjectionMatrixB by Pseudo-Inverse of ProjectionMatrixA
    ProjectMat_OriginP_PseudInvs=np.linalg.pinv(ProjectionMatrixA_To_be_Origin)
    DoublePs=np.matmul(ProjectionMatrixCamB_to_be_Relative,ProjectMat_OriginP_PseudInvs)
    
    #P'C
    #multiply ProjectionMatrixB by Camera Coordinations (translation) of decomposed
    #ProjectionMatrixA
    C_cameraCoords=GetCameraCoords_FromProjectionMatrix(ProjectionMatrixA_To_be_Origin,useOpenCVmethod=False)
    PC_part=np.matmul(ProjectionMatrixCamB_to_be_Relative,C_cameraCoords)
    
    
    
    #[P'C]x
    #get skew symmetric matrix of this vector
    #if a=(a1,a2,a3)^t is a 3 vector - one can define a corresponding skew symmetric matrix:
    #       [0 -a3 a2]
    #[a]x=  [a3 0 -a1]
    #       [-a2 a1 0]
    ZeroOffset=1
    PC_part_SkewSymmetric= np.zeros(shape=(3,3))
    PC_part_SkewSymmetric[0,1]=-PC_part[3-ZeroOffset]
    PC_part_SkewSymmetric[0,2]=PC_part[2-ZeroOffset]
    PC_part_SkewSymmetric[1,0]=PC_part[3-ZeroOffset]
    PC_part_SkewSymmetric[1,2]=-PC_part[1-ZeroOffset]
    PC_part_SkewSymmetric[2,0]=-PC_part[2-ZeroOffset]
    PC_part_SkewSymmetric[2,1]=PC_part[1-ZeroOffset]
    
    
    
    #F=[P'C]x   P'P^+
    #multiply [P'C]x by P'P^+
    FundamentalFromProjections=np.matmul( PC_part_SkewSymmetric,DoublePs)
    FundamentalMatrix=FundamentalFromProjections
    return FundamentalMatrix




def GetPointsFromGradient(InputImage_GRAYSCALE,LvlDetail=1):
    global DisplayWinImage_Dense
    LvlDetail=int(LvlDetail)
    if LvlDetail<1: LvlDetail=1
    
    #TODO - no check that we are getting grayscale in here
    
    GradientImage_laplacian=InputImage_GRAYSCALE
   # GradientImage_laplacian=cv2.Laplacian(InputImage,cv2.CV_64F)
   # GradientImage_laplacian=_3DVisLabLib.Normalize_and_round(GradientImage_laplacian)
    #invert the calibration image - circle finder needs black dots on white background
    #GradientImage_laplacian=cv2.bitwise_not(GradientImage_laplacian)
    #ret,GradientImage_laplacian = cv2.threshold(GradientImage_laplacian,100,255,cv2.THRESH_BINARY)
    GradientImage_laplacian=cv2.adaptiveThreshold(GradientImage_laplacian,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
#    DisplayWinImage_Dense("AdaptiveThreshold",GradientImage_laplacian,ParamsObj.Dense_AutoF_CameraA_Name)
     
     
    ret,GradientImage_laplacian = cv2.threshold(GradientImage_laplacian,150,255,cv2.THRESH_BINARY)
    
   # DisplayWinImage_Dense("Binary Threshold",GradientImage_laplacian,ParamsObj.Dense_AutoF_CameraA_Name)
    
    GradientImage_laplacian= cv2.medianBlur(GradientImage_laplacian, 3)
    
   # DisplayWinImage_Dense("Median Blur",GradientImage_laplacian,ParamsObj.Dense_AutoF_CameraA_Name)
    
    #plt.imshow(GradientImage_laplacian)
    #cv2.imwrite("testedge.jpg",GradientImage_laplacian)
    
    
    EdgesChosenCheck=InputImage_GRAYSCALE.copy()
    EdgesChosenCheck[:,:]=0
   
    
    pts_edges=[]
    ImageX=GradientImage_laplacian.shape[1]
    ImageY=GradientImage_laplacian.shape[0]
    #TODO can be done faster with a WHERE function
    #seems we need a border or the cross correlation thing will explode
    RangeX= range(10,ImageX-10,1)
    RangeY= range(10,ImageY-10,1)
    DetailCountUp=0
    for Xindex in RangeX:#range (10,ImageX-10,1):
        for Yindex in RangeY:#range (10,ImageY-10,1):
            if GradientImage_laplacian[Yindex,Xindex]<200:
                if DetailCountUp>LvlDetail:
                    EdgesChosenCheck[Yindex,Xindex]=255
                    pts_edges.append((Xindex,Yindex))
                    DetailCountUp=0
                DetailCountUp=DetailCountUp+1
    #print("hay****************")
    #print (len(pts_edges))     
    #DisplayWinImage_Dense(str(len(pts_edges)) + " EdgePoints Chosen Double Check ",EdgesChosenCheck,ParamsObj.Dense_AutoF_CameraA_Name)
    
    return GradientImage_laplacian ,pts_edges



def FindProjectionMatrix_ForCamera(CameraName, CameraDetailFolder):
    #now lets get projection matrices from some kind of repository
    #TEMPORARY SOLUTION
    #use cameraname to find calibrated file in calibs folder and return the
    #projection matrix
    #if its origincamera we want - because we dont have that specific calibration file
    #we will have to do a bit of cheating - as at moment each calibration file
    #hase ORIGIN cam as cam A and Stereo Pair as camera B
    file_list = glob.glob(CameraDetailFolder + "*.hey")
    ProjectionMatrix=None
    
    for CalibFile in file_list:
        if CameraName.lower() in CalibFile.lower():
            #calibration found
            #in this current format - when we load calibration file
            #which is in A vs B form - are we picking the ORIGIN camera
            #(A) or a stereo pair (B)? Remember we have no specific
            #calibration file for the origin - only ORIGIN V PAIR
        
            CalibrationData_reload=pickle.load(open(CalibFile, 'rb'))
            #print(CalibrationData_reload)
            
            if CalibrationData_reload["str_SequenceCamA"].lower()==CameraName.lower():
                #print("First Cam")
                return CalibrationData_reload["Pmatrix_manualA"]
            
            if CalibrationData_reload["str_SequenceCamB"].lower()==CameraName.lower():
                #print("second Cam")
                return CalibrationData_reload["Pmatrix_manualB"]
            
            if ProjectionMatrix is None:
                raise ValueError('FindProjectionMatrix_ForCamera no camera details in calib pair or no calib file : ' + CameraName + " in  " + CameraDetailFolder )
                
    #otherwise nothing has been found
    raise ValueError('FindProjectionMatrix_ForCamera no camera details in calib pair or no calib file : ' + CameraName + " in  " + CameraDetailFolder )
           




def GetScalingRatio_FromMVG_SparseCloud(intData_ScaleMesh_to_realworld,intObj_ReconDetails):
    """Prototype method to disseminate the initialposes.ply file from OpenMVG"""
    
    """#If reconstruction is succesfull and the two forced views are found
    #(pod2primary and pod2secondary) then a file "initialPair.ply" will be 
    #created which is sparse cloud and the TWO views - cloud is in WHITE while
    #the view colours will be in GREEN .30
    
    #lets open the file, find the green views
    #(camera positions) and get the distance"""
    #but we have to find the two camera points and a voxel from the subject, then
    #open CLOUD AND POSES which doesnt have colours to discrimate between camera positions
    #and subject cloud - and refind this triangle
    #from this we can get the final scale
    import plyfile
    from plyfile import PlyData, PlyElement

    
    

    with open(intObj_ReconDetails.File_InitialPair, 'rb') as x:
        intData_ScaleMesh_to_realworld.SparseCloud_InitialPair = PlyData.read(x)
        #first and second elements should always be initial pair of views
        intData_ScaleMesh_to_realworld.InitialPairA=intData_ScaleMesh_to_realworld.SparseCloud_InitialPair.elements[0].data[0]
        intData_ScaleMesh_to_realworld.InitialPairB=intData_ScaleMesh_to_realworld.SparseCloud_InitialPair.elements[0].data[1]
        #3rd element should be first sparse recon voxel
        intData_ScaleMesh_to_realworld.InitialPairFirstVoxel=intData_ScaleMesh_to_realworld.SparseCloud_InitialPair.elements[0].data[2]
        #LengthPointCLoud=len(intData_ScaleMesh_to_realworld.SparseCloud_InitialPair.elements[0].data)-1
        intData_ScaleMesh_to_realworld.InitialPairLastVoxel=intData_ScaleMesh_to_realworld.SparseCloud_InitialPair.elements[0].data[3]
        
        
        #colour first pixel found
        
        #test validity of points
        if (intData_ScaleMesh_to_realworld.InitialPairA[3]!=0 or intData_ScaleMesh_to_realworld.InitialPairA[4]!=255 or intData_ScaleMesh_to_realworld.InitialPairA[5]!=0):
                raise ValueError("Initial Pair .ply element 0 is not green (first camera View)")
        if (intData_ScaleMesh_to_realworld.
            InitialPairB[3]!=0 or intData_ScaleMesh_to_realworld.InitialPairB[4]!=255 or intData_ScaleMesh_to_realworld.InitialPairB[5]!=0):
                raise ValueError("Initial Pair .ply element 1 is not green (second camera View)")
        if (intData_ScaleMesh_to_realworld.InitialPairFirstVoxel[3]!=255 or intData_ScaleMesh_to_realworld.InitialPairFirstVoxel[4]!=255 or intData_ScaleMesh_to_realworld.InitialPairFirstVoxel[5]!=255):
                raise ValueError("Initial Pair .ply element 2 is not white (first subject 3d point)")       
        if (intData_ScaleMesh_to_realworld.InitialPairLastVoxel[3]!=255 or intData_ScaleMesh_to_realworld.InitialPairLastVoxel[4]!=255 or intData_ScaleMesh_to_realworld.InitialPairLastVoxel[5]!=255):
                raise ValueError("Initial Pair .ply element N is not white (Nth subject 3d point)")      
        
        #calculate distances to use for ratio test 
        intData_ScaleMesh_to_realworld.Distances_InitialPosesA_B=GetDistance_XYZs_PlyElements(intData_ScaleMesh_to_realworld.InitialPairA,intData_ScaleMesh_to_realworld.InitialPairB)
        intData_ScaleMesh_to_realworld.Distances_InitialPosesA_1stVoxel=GetDistance_XYZs_PlyElements(intData_ScaleMesh_to_realworld.InitialPairA,intData_ScaleMesh_to_realworld.InitialPairFirstVoxel)
        intData_ScaleMesh_to_realworld.Distances_InitialPosesB_1stVoxel=GetDistance_XYZs_PlyElements(intData_ScaleMesh_to_realworld.InitialPairB,intData_ScaleMesh_to_realworld.InitialPairFirstVoxel)
        intData_ScaleMesh_to_realworld.Distances_InitialPosesA_LastVoxel=GetDistance_XYZs_PlyElements(intData_ScaleMesh_to_realworld.InitialPairA,intData_ScaleMesh_to_realworld.InitialPairLastVoxel)
        intData_ScaleMesh_to_realworld.Distances_InitialPosesB_LastVoxel=GetDistance_XYZs_PlyElements(intData_ScaleMesh_to_realworld.InitialPairB,intData_ScaleMesh_to_realworld.InitialPairLastVoxel)
        intData_ScaleMesh_to_realworld.Distances_InitialPosesFirstVoxel_LastVoxel=GetDistance_XYZs_PlyElements(intData_ScaleMesh_to_realworld.InitialPairFirstVoxel,intData_ScaleMesh_to_realworld.InitialPairLastVoxel)
        
            #try to close the locked file
            #intData_ScaleMesh_to_realworld.SparseCloud_cloud_and_poses
        
    #print("intData_ScaleMesh_to_realworld.InitialPairFirstVoxel",intData_ScaleMesh_to_realworld.InitialPairFirstVoxel)
    print("intData_ScaleMesh_to_realworld.InitialPairA", intData_ScaleMesh_to_realworld.InitialPairA)
    print("intData_ScaleMesh_to_realworld.InitialPairB", intData_ScaleMesh_to_realworld.InitialPairB)
    print("Distances_InitialPosesA_B", intData_ScaleMesh_to_realworld.Distances_InitialPosesA_B)
    #print("Distances_InitialPosesA_1stVoxel", intData_ScaleMesh_to_realworld.Distances_InitialPosesA_1stVoxel)
    #print("Distances_InitialPosesB_1stVoxel", intData_ScaleMesh_to_realworld.Distances_InitialPosesB_1stVoxel)
    #print("Distances_InitialPosesA_LastVoxel", intData_ScaleMesh_to_realworld.Distances_InitialPosesA_LastVoxel)
    #print("Distances_InitialPosesB_LastVoxel", intData_ScaleMesh_to_realworld.Distances_InitialPosesB_LastVoxel)
    #print("Distances_InitialPosesFirstVoxel_LastVoxel", intData_ScaleMesh_to_realworld.Distances_InitialPosesFirstVoxel_LastVoxel)
    
    
    Dict_StoreRatioTest=[]
    FirstVoxelPos=None
    LastVoxelPos=None
    #Now lets open the rescaled point cloud (rscaled by bundler?)
    #we want to find the same camera views as above so we can get the distance and change of scale,
    #but the cameras can be in different indexes- and can be translated anywhere, so we need a 3rd
    #position to make sure we have the right ones. We are assuming the first 3d point of the subject
    #is always after the camera positions, that way for each 3 sets (A,B,point) we get the
    #distnace ratio and see if it matches the InitialPair.ply results
    with open(intObj_ReconDetails.File_cloud_and_poses, 'rb') as f:
        intData_ScaleMesh_to_realworld.SparseCloud_cloud_and_poses = PlyData.read(f)
        #lets try and find the first cloud voxels (white colour)
        #assuming its first element AFTER all views (which could be less than number of images if
        #some have failed stereoscopy)
        PlyElementX=0
        StartRange=0
        FirstCloudElement=0#in cloud and poses, what element does cloud start (view voxels always first)
        Temp_ViewApos=None
        Temp_ViewBpos=None
        
        Temp_DistanceA_B=None
        Temp_DistanceA_FirstVoxel=None
        Temp_DistanceB_FirstVoxel=None
        Temp_DistanceA_LastVoxel=None
        Temp_DistanceB_LastVoxel=None
        Temp_DistanceFirstVoxel_LastVoxel=None
        Temp_RatioA_B=None
        Temp_RatioA_FirstVoxel=None
        Temp_RatioB_FirstVoxel=None
        Temp_RatioA_LastVoxel=None
        Temp_RatioB_LastVoxel=None
        Temp_RatioFirstVoxel_LastVoxel=None
        EndRange=intObj_ReconDetails.NumberOfViews+3
        for PlyElementX in range (StartRange,EndRange):
            
           
            
            FirstCloudElement=FirstCloudElement+1
            TestPly_Element=intData_ScaleMesh_to_realworld.SparseCloud_cloud_and_poses.elements[0].data[PlyElementX]
            
            #load up camera views
            if (TestPly_Element[3]==0 and TestPly_Element[4]==255 and TestPly_Element[5]==0):
                TempDistance=GetDistance_XYZs_PlyElements(TestPly_Element,(0, 0, 0, 0, 255, 0))
                intData_ScaleMesh_to_realworld.Dict_CameraViews.append(intData_ScaleMesh_to_realworld.CameraViewsTuple(TestPly_Element,TempDistance))
            
            
            #loop through elements until finding first white (subject cloud) voxel
            if (TestPly_Element[3]==255 and TestPly_Element[4]==255 and TestPly_Element[5]==255):
                #get first subject pixel
                FirstVoxelPos=TestPly_Element
                #get 2nd subject pixel
                LastVoxelPos=intData_ScaleMesh_to_realworld.SparseCloud_cloud_and_poses.elements[0].data[PlyElementX+1]
                print("First Cloud voxel found at pos", FirstCloudElement)
                
                #print("FirstVoxelPos",FirstVoxelPos)
                #print("LastVoxelPos",LastVoxelPos)
                break
        if PlyElementX>=(EndRange-1):
            raise ValueError("Could not find first white voxel in ",intObj_ReconDetails.File_cloud_and_poses )
        
        if not (FirstVoxelPos[3]==255 and FirstVoxelPos[4]==255 and FirstVoxelPos[5]==255):
            raise ValueError("FirstVoxelPos wrong colour")
        if not (LastVoxelPos[3]==255 and FirstVoxelPos[4]==255 and FirstVoxelPos[5]==255):
            raise ValueError("LastVoxelPos wrong colour")
        intData_ScaleMesh_to_realworld.Distances_CloudPosesFirstVoxel_LastVoxel=GetDistance_XYZs_PlyElements(FirstVoxelPos,LastVoxelPos)
       
        
        
        #find index of minimum distance to zero - 
        #lets assume this is adjusted POINT A 
        minIndex = [num[1] for num in intData_ScaleMesh_to_realworld.Dict_CameraViews].index(min([num[1] for num in intData_ScaleMesh_to_realworld.Dict_CameraViews]))
        intData_ScaleMesh_to_realworld.CloudPosesA=intData_ScaleMesh_to_realworld.SparseCloud_cloud_and_poses.elements[0].data[minIndex]
        print("Minimum POINT A",intData_ScaleMesh_to_realworld.CloudPosesA)
        #print([num[1] for num in intData_ScaleMesh_to_realworld.Dict_CameraViews])
        #print("found at ", minIndex)
        
        
        
        
        
        #test each view against the others and see if scaling fits original A/B/Voxel
        #this repeats tests so can be made more efficient if we have to do the
        #entire cloud
        #TODO
       
        #top loop goes through every element in plyfile (including camera views)
        for PlyElementX2 in range (0,intData_ScaleMesh_to_realworld.SparseCloud_cloud_and_poses.elements[0][:].size):
            TestPly_Element=intData_ScaleMesh_to_realworld.SparseCloud_cloud_and_poses.elements[0].data[PlyElementX2]
            #checks this element is not camera view (by colour)
            if (TestPly_Element[3]==255 and TestPly_Element[4]==255 and TestPly_Element[5]==255):
                #get first pixel of sparse reconstructed subject
                FirstVoxelPos=TestPly_Element
                #loop through camera views for VIEW A
                for ViewA_index in range (0,intObj_ReconDetails.NumberOfViews+1):#make sure not stopping too early
                    Temp_RatiosVariance=None
                    Temp_ViewApos=None
                    Temp_ViewApos=intData_ScaleMesh_to_realworld.SparseCloud_cloud_and_poses.elements[0].data[ViewA_index]
                    #Temp_ViewApos=intData_ScaleMesh_to_realworld.CloudPosesA#this is minimum distance to zero POINT A 
                    
                    #check VIEW A is valid camera view (by colour (green))
                    if (Temp_ViewApos[3]==0 and Temp_ViewApos[4]==255 and Temp_ViewApos[5]==0):
                        #get VIEW B - avoiding already-done combos
                        for ViewB_index in range (0,intObj_ReconDetails.NumberOfViews+1):
                             Temp_ViewBpos=None
                             Temp_ViewBpos=intData_ScaleMesh_to_realworld.SparseCloud_cloud_and_poses.elements[0].data[ViewB_index]   
                             #Temp_ViewBpos=intData_ScaleMesh_to_realworld.SparseCloud_cloud_and_poses.elements[0].data[2]   
                             
                             #check VIEW B is valid camera view (by colour (green))
                             if (Temp_ViewBpos[3]==0 and Temp_ViewBpos[4]==255 and Temp_ViewBpos[5]==0):
                                 if (Temp_ViewApos is not None) and (Temp_ViewBpos is not None)and (FirstVoxelPos is not None) :
                                     #we have a valid points of a triangle now
                                     #lets check ratios
                                     Temp_DistanceA_B= GetDistance_XYZs_PlyElements(Temp_ViewApos,Temp_ViewBpos)
                                     Temp_DistanceA_FirstVoxel=GetDistance_XYZs_PlyElements(Temp_ViewApos,FirstVoxelPos)
                                     Temp_DistanceB_FirstVoxel=GetDistance_XYZs_PlyElements(Temp_ViewBpos,FirstVoxelPos)
                                     #Temp_DistanceA_LastVoxel=GetDistance_XYZs_PlyElements(Temp_ViewApos,LastVoxelPos)
                                     #Temp_DistanceB_LastVoxel=GetDistance_XYZs_PlyElements(Temp_ViewBpos,LastVoxelPos)
                                     #Temp_DistanceFirstVoxel_LastVoxel=GetDistance_XYZs_PlyElements(FirstVoxelPos,LastVoxelPos)
                                     
                                     
                                     #make sure no duplicate errors when grabbing points
                                     if abs(Temp_DistanceA_B)>0.00001 and abs(Temp_DistanceA_FirstVoxel)>0.00001 and abs(Temp_DistanceB_FirstVoxel)>0.00001 :
                                         
                                         Temp_RatioA_B=round(intData_ScaleMesh_to_realworld.Distances_InitialPosesA_B/Temp_DistanceA_B,3)
                                         Temp_RatioA_FirstVoxel=round(intData_ScaleMesh_to_realworld.Distances_InitialPosesA_1stVoxel/Temp_DistanceA_FirstVoxel,3)
                                         Temp_RatioB_FirstVoxel=round(intData_ScaleMesh_to_realworld.Distances_InitialPosesB_1stVoxel/Temp_DistanceB_FirstVoxel,3)
                                         #Temp_RatioA_LastVoxel=intData_ScaleMesh_to_realworld.Distances_InitialPosesA_1stVoxel/Temp_DistanceA_LastVoxel
                                         #Temp_RatioB_LastVoxel=intData_ScaleMesh_to_realworld.Distances_InitialPosesB_1stVoxel/Temp_DistanceB_LastVoxel
                                         #Temp_RatioFirstVoxel_LastVoxel=intData_ScaleMesh_to_realworld.Distances_InitialPosesFirstVoxel_LastVoxel/Temp_DistanceFirstVoxel_LastVoxel
                                         Temp_RatiosVariance=round(np.var([Temp_RatioA_B,Temp_RatioA_FirstVoxel,Temp_RatioB_FirstVoxel]),8)
                                         TempString="rAB " + str(Temp_RatioA_B) + " rA_1stV " + str(Temp_RatioA_FirstVoxel) + " rB_1stV " + str(Temp_RatioB_FirstVoxel) + " pos: " +str (PlyElementX2)
                                         #not sure why this breaks after moving here
                                         Dict_StoreRatioTest.append(intData_ScaleMesh_to_realworld.RatioTuple(Temp_RatiosVariance,copy(Temp_ViewApos),copy(Temp_ViewBpos),copy(FirstVoxelPos),TempString))
                                     else:
                                         pass
                                       # print("zero found")
    #                            
    #should have a list of all the possible combinations of the N viewpoints in CloudAndPoses.ply file
    
    #lets find element with lowest variance of ratios (all distances divided by initial distances have least error)
    #can't figure out Python method to get a maximum from a list of tuples so do it hard way
        #f.close()
        
        
    TempMinVariance=999999
    Variances=[]
    TempStoreIndex=0
    for GetMin in range (0, len(Dict_StoreRatioTest)):
        Variances.append(Dict_StoreRatioTest[GetMin][0])
        if Dict_StoreRatioTest[GetMin][0]<TempMinVariance:
            TempMinVariance=Dict_StoreRatioTest[GetMin][0]
            TempStoreIndex=GetMin
    Variances.sort(key=float)
    
              
    intData_ScaleMesh_to_realworld.Distances_CloudPosesA_B=GetDistance_XYZs_PlyElements(Dict_StoreRatioTest[TempStoreIndex][1],Dict_StoreRatioTest[TempStoreIndex][2])
    intData_ScaleMesh_to_realworld.Distances_CloudPosesA_1stVoxel=GetDistance_XYZs_PlyElements(Dict_StoreRatioTest[TempStoreIndex][1],FirstVoxelPos)
    intData_ScaleMesh_to_realworld.Distances_CloudPosesB_1stVoxel=GetDistance_XYZs_PlyElements(Dict_StoreRatioTest[TempStoreIndex][2],FirstVoxelPos)
    #intData_ScaleMesh_to_realworld.Distances_CloudPosesA_LastVoxel=GetDistance_XYZs_PlyElements(Dict_StoreRatioTest[TempStoreIndex][1],LastVoxelPos)
    #intData_ScaleMesh_to_realworld.Distances_CloudPosesB_LastVoxel=GetDistance_XYZs_PlyElements(Dict_StoreRatioTest[TempStoreIndex][2],LastVoxelPos)
           
    #print("Cloudposes1stVoxel",FirstVoxelPos)
    #print("Cloudposes1stVoxel",LastVoxelPos)
    print("Point Combination Found", Dict_StoreRatioTest[TempStoreIndex])
    print("Distances_CloudPosesA_B", intData_ScaleMesh_to_realworld.Distances_CloudPosesA_B)
    print("Distances_CloudPosesA_1stVoxel", intData_ScaleMesh_to_realworld.Distances_CloudPosesA_1stVoxel)
    print("Distances_CloudPosesB_1stVoxel", intData_ScaleMesh_to_realworld.Distances_CloudPosesB_1stVoxel)
    #print("Distances_CloudPosesA_LastVoxel", intData_ScaleMesh_to_realworld.Distances_CloudPosesA_LastVoxel)
    #print("Distances_CloudPosesB_LastVoxel", intData_ScaleMesh_to_realworld.Distances_CloudPosesB_LastVoxel)
    #print("Distances_CloudPosesFirstVoxel_LastVoxel", intData_ScaleMesh_to_realworld.Distances_CloudPosesFirstVoxel_LastVoxel)
    
    print("cloud poses A",Dict_StoreRatioTest[TempStoreIndex][1])
    print("cloud poses B",Dict_StoreRatioTest[TempStoreIndex][2])
    
    #print(Dict_StoreRatioTest)
    #'cameras primary 1 secondary 1 measured at ~16.7mm
    #if initial camera poses pair is 1 then we don't need to use a ratio -
    #but check it mannually
    
    return intData_ScaleMesh_to_realworld.Distances_CloudPosesA_B

def GetDistance_XYZs_PlyElements(Ply_Element1, Ply_Element2):
    return round(math.sqrt(((Ply_Element1[0]-Ply_Element2[0])**2)+((Ply_Element1[1]-Ply_Element2[1])**2)+((Ply_Element1[2]-Ply_Element2[2])**2)),3)
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class ReconstructionDetails:
    """Hold working details like filenames"""
    def __init__(self,Folder_ReconRoot="",
                            Str_SpecifiedViewA="",
                            Str_SpecifiedViewB="",
                            MeshLabServerEXE="",
                            NumberOfViews=0):
        self.Folder_ReconRoot=Folder_ReconRoot
        self.InitialImageFolder="/InitialImages"
        self.Str_ViewA=""
        self.Str_ViewB=""
        self.Str_SpecifiedViewA=Str_SpecifiedViewA
        self.Str_SpecifiedViewB=Str_SpecifiedViewB
        self.MVS_Output="/mvs_dir"
        self.SensorWidthDBFileLocation="/sensor_width_database/sensor_width_camera_database.txt"
        self.Folder_Output_Matches="/matches"
        self.File_SFmJson="/sfm_data.json"
        self.Folder_IncrmtRecon="/out_Incremental_Reconstruction"
        self.File_SFmData="/sfm_data.bin"
        self.File_ReconModel="/model.mvs"
        self.NumberOfViews=NumberOfViews
        self.MeshLabScriptDummyTxt="dummytexthere"
        self.File_FinalMesh="/model_dense_mesh_refine_texture.ply"
        self.Sfm_data_Exported="sfm_data_Extrinsics_Views.json"
        self.Sfm_data_Exported_InitPair_Distance=None
        self.ScaledMeshFileName="/ScaledMesh_mm.ply"
        self.MeshLabServerEXE=MeshLabServerEXE
        self.ImageSet_Dimensions=None
        self.ImageSet_EXIF_Make=None
        self.ImageSet_EXIF_Model=None
        self.ImageSet_ApproximatedFocalLength=""
        #self.ImageSet_PrinciplePointParamater=""
        #this script is saved as meshscalescript.mlx from meshlab when we save scaling process
        self.MeshlabScript_Name="/MeshLab_Scale_Script.mlx"
        self.MeshLabScript="""<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Matrix: Set from translation/rotation/scale">
  <Param type="RichFloat" value="0" name="translationX" description="X Translation" tooltip="Translation factor on X axis"/>
  <Param type="RichFloat" value="0" name="translationY" description="Y Translation" tooltip="Translation factor on Y axis"/>
  <Param type="RichFloat" value="0" name="translationZ" description="Z Translation" tooltip="Translation factor on Z axis"/>
  <Param type="RichFloat" value="0" name="rotationX" description="X Rotation" tooltip="Rotation angle on X axis"/>
  <Param type="RichFloat" value="0" name="rotationY" description="Y Rotation" tooltip="Rotation angle on Y axis"/>
  <Param type="RichFloat" value="0" name="rotationZ" description="Z Rotation" tooltip="Rotation angle on Z axis"/>
  <Param type="RichFloat" value=""" + self.MeshLabScriptDummyTxt + """ name="scaleX" description="X Scale" tooltip="Scaling factor on X axis"/>
  <Param type="RichFloat" value=""" + self.MeshLabScriptDummyTxt + """ name="scaleY" description="Y Scale" tooltip="Scaling factor on Y axis"/>
  <Param type="RichFloat" value=""" + self.MeshLabScriptDummyTxt + """ name="scaleZ" description="Z Scale" tooltip="Scaling factor on Z axis"/>
  <Param type="RichBool" value="false" name="compose" description="Compose with current" tooltip="If selected, the new matrix will be composed with the current one (matrix=new*old)"/>
  <Param type="RichBool" value="true" name="Freeze" description="Freeze Matrix" tooltip="The transformation is explicitly applied, and the vertex coordinates are actually changed"/>
  <Param type="RichBool" value="false" name="allLayers" description="Apply to all visible Layers" tooltip="If selected the filter will be applied to all visible mesh layers"/>
 </filter>
</FilterScript>
"""
        #sequentially dependant
        self.Folder_OpenMVS=self.Folder_ReconRoot + '/Resources/SFM_InstallationFiles/OpenMVS/openMVS_sample-0.7a'
        self.Folder_OpenMVG=self.Folder_ReconRoot + '/Resources/SFM_InstallationFiles/OpenMVG/ReleaseV1.6.Halibut.WindowsBinaries_VS2017'
        self.UserConfigFile=self.Folder_ReconRoot +"/Resources/UserConfig/Retopology_Config.txt"
        self.Folder_WorkingDirectoryOutput=self.Folder_ReconRoot + '/SFM_Output'
        self.Folder_WorkingDirectoryInput=self.Folder_ReconRoot + '/InputImages'
        self.File_InitialPair=self.Folder_WorkingDirectoryOutput + self.Folder_IncrmtRecon + "/initialPair.ply"
        self.File_cloud_and_poses=self.Folder_WorkingDirectoryOutput + self.Folder_IncrmtRecon + "/cloud_and_poses.ply"


class Data_ScaleMesh:
        """Hold working details associated with rescaling a mesh to real-world"""
        def __init__(self,CameraPair_Distance_mm=1):
            self.InitialPair_mm=CameraPair_Distance_mm#distance between chosen initial camera pairs in mm
            self.InitialPairA=None
            self.InitialPairB=None
            self.InitialPairFirstVoxel=None
            self.InitialPairLastVoxel=None
            
            self.CloudPosesA=None
            self.CloudPosesB=None
            self.SparseCloud_InitialPair=None
            self.SparseCloud_cloud_and_poses=None
            self.CloudPoses_unitDistance=None
            self.InitialPair_unitDistance=None
            
            self.Distances_InitialPosesA_B=0.0
            self.Distances_InitialPosesA_1stVoxel=0.0
            self.Distances_InitialPosesB_1stVoxel=0.0
            self.Distances_InitialPosesA_LastVoxel=0.0
            self.Distances_InitialPosesB_LastVoxel=0.0
            self.Distances_InitialPosesFirstVoxel_LastVoxel=None
            
            self.Distances_CloudPosesA_B=0.0
            self.Distances_CloudPosesA_1stVoxel=0.0
            self.Distances_CloudPosesB_1stVoxel=0.0
            self.Distances_CloudPosesA_LastVoxel=0.0
            self.Distances_CloudPosesB_LastVoxel=0.0
            self.Distances_CloudPosesFirstVoxel_LastVoxel=None
            
            
            
            self.Dict_CameraViews=[]
            self.Ratio_InitialPose_A_B_1stVoxel=0.0
            self.CloudPosesFirstVoxel=None
            self.RatioTuple=namedtuple('RatioTuple', ['DistanceVariance', 'ViewA', 'ViewB','FirstVoxel','notes'])                
            self.CameraViewsTuple=namedtuple('CameraViewsTuple', ['PlyElement','DistanceToZero'])                
            


def ExtractSfm_Details_From_Sfm_datafile(Obj_ReconDetails_internal):
    """this will open the extracted sfm_data.bin details from MVG"""
   
    Temp_Filename=""
    Temp_IdPose=""
    Temp_IdView=""
    Temp_InitialImg_A_Pose=None
    Temp_InitialImg_B_Pose=None
    Temp_ViewPosition=None
    Temp_ViewRotation=None
    Temp_Distance=None
    FileName=Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_IncrmtRecon + "/" + Obj_ReconDetails_internal.Sfm_data_Exported
    print("Getting Scaling Details from " + FileName)
    with open(FileName) as json_file:
        sfm_Data_file=json.load(json_file)
        for viewitem in sfm_Data_file['views']:
            Temp_Filename=viewitem['value']['ptr_wrapper']['data']['filename']
            #chop out ".extension"
            #Temp_Filename=Temp_Filename[0:(Temp_Filename.find('.'))]
            Temp_IdPose=viewitem['value']['ptr_wrapper']['data']['id_pose']
            IdView=viewitem['value']['ptr_wrapper']['data']['id_view']
            Temp_position=sfm_Data_file['extrinsics'][int(Temp_IdPose)]['value']['center']
            #test if image A or B
            if Obj_ReconDetails_internal.Str_SpecifiedViewA.upper() in Temp_Filename.upper():
                Temp_InitialImg_A_Pose=Temp_position
                print("found view A " , Temp_Filename)
            elif Obj_ReconDetails_internal.Str_SpecifiedViewB.upper()in Temp_Filename.upper():
                Temp_InitialImg_B_Pose = Temp_position
                print("found view B ",Temp_Filename )
        
        if (Temp_InitialImg_A_Pose is not None) and (Temp_InitialImg_B_Pose is not None):
            Temp_Distance=GetDistance_XYZs_PlyElements(
                                         (Temp_InitialImg_A_Pose[0],
                                          Temp_InitialImg_A_Pose[1],
                                          Temp_InitialImg_A_Pose[2],
                                           0,0,0),
                                          (Temp_InitialImg_B_Pose[0], 
                                           Temp_InitialImg_B_Pose[1], 
                                           Temp_InitialImg_B_Pose[2],
                                           0, 0, 0))

        
            print("Sfm_data.bin initial pair ratio from unity", Temp_Distance)
        return Temp_Distance

def GetList_Of_ImagesInfolder(FolderPath, ImageTypes=(".jPg", ".Png",".gif")):
    
    Image_FileNames=[]
    
    Src_files=os.listdir(FolderPath)
    
    #list comprehension [function-of-item for item in some-list
    ImageTypes_ForceLower=[x.lower()  for x in ImageTypes]
    ImageTypes_ForceLower_Tuple=tuple(ImageTypes_ForceLower)
    
    for filename in Src_files:
    #if a known image filetype - copy file
        if str.endswith(str.lower(filename),ImageTypes_ForceLower_Tuple):
            Image_FileNames.append(filename)
    
    return Image_FileNames

def GetList_Of_ImagesInList(ListOfFiles, ImageTypes=(".jPg", ".Png",".gif")):
    
    Image_FileNames=[]
    
    #list comprehension [function-of-item for item in some-list
    ImageTypes_ForceLower=[x.lower()  for x in ImageTypes]
    ImageTypes_ForceLower_Tuple=tuple(ImageTypes_ForceLower)
    
    for filename in ListOfFiles:
    #if a known image filetype - copy file
        if str.endswith(str.lower(filename),ImageTypes_ForceLower_Tuple):
            Image_FileNames.append(filename)
    
    return Image_FileNames
            
            
def PrepareFolderStructure(Obj_ReconDetails_internal,AskBeforeDelete=True):
    """delete output folder and copy images from input folder"""
    #set up strings
    Is_InitialStereoPairValid=False
    ConfirmString="y"
    UserConfirm_Deltree="n"
    ImageTypes=(".jPg", ".Png",".gif")
    #list comprehension [function-of-item for item in some-list
    ImageTypes_ForceLower=[x.lower()  for x in ImageTypes]
    ImageTypes_ForceLower_Tuple=tuple(ImageTypes_ForceLower)
    #clean full filename and path of initial stereo pair (if using)
    Obj_ReconDetails_internal.Str_ViewA=Str_Common.EMPTY
    Obj_ReconDetails_internal.Str_ViewB=Str_Common.EMPTY
    def test_String_isValid(InputString):
        """Checks a string is valid - if not returns boolean"""
        if ("".__eq__(InputString)==False):
            return True
        return False
    
    
    if test_String_isValid(Obj_ReconDetails_internal.Str_SpecifiedViewA) != test_String_isValid(Obj_ReconDetails_internal.Str_SpecifiedViewB):
       raise Exception("Error, initial stereo image pairs - only one pair is valid string. Set both to NONE or update strings") 
    
    #are input initial stereo image pair strings valid?
    if test_String_isValid(Obj_ReconDetails_internal.Str_SpecifiedViewA) and test_String_isValid(Obj_ReconDetails_internal.Str_SpecifiedViewB):
        Is_InitialStereoPairValid=True
    else:
        Is_InitialStereoPairValid=False
    
    
    if AskBeforeDelete==True: UserConfirm_Deltree=input("Type " + ConfirmString + " to delete folder " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput)
    if (str.lower(UserConfirm_Deltree)==str.lower(ConfirmString)) or AskBeforeDelete==False:
        Deltree(Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput) 
    #create new working directory
    os.mkdir(Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput)
    os.mkdir(Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output)
    #we need to copy images from input directory to working directory
    Src_files=os.listdir(Obj_ReconDetails_internal.Folder_WorkingDirectoryInput)
    #now try and find any known camera pairs
    for filename in Src_files:
        #if a known image filetype - copy file
        if str.endswith(str.lower(filename),ImageTypes_ForceLower_Tuple):
            shutil.copy(Obj_ReconDetails_internal.Folder_WorkingDirectoryInput + "/" + filename, Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput)
        
        
        if Is_InitialStereoPairValid==True:
            #check for initial stereo pair if known
            if str.endswith(str.lower(filename),str.lower(Obj_ReconDetails_internal.Str_SpecifiedViewA)):
                
                if Obj_ReconDetails_internal.Str_ViewA==Str_Common.EMPTY:
                    Obj_ReconDetails_internal.Str_ViewA=filename
                else:
                    raise Exception("Initial Stereo Pair Image A Error - more than 1 instance of specified suffix found!" + Obj_ReconDetails_internal.Str_SpecifiedViewA)
                
            elif str.endswith(str.lower(filename),str.lower(Obj_ReconDetails_internal.Str_SpecifiedViewB)):
                
                if Obj_ReconDetails_internal.Str_ViewB==Str_Common.EMPTY:
                    Obj_ReconDetails_internal.Str_ViewB=filename
                else:
                    raise Exception("Initial Stereo Pair Image B Error - more than 1 instance of specified suffix found!" + Obj_ReconDetails_internal.Str_SpecifiedViewB)
               
                
    if Is_InitialStereoPairValid==True:           
        if Obj_ReconDetails_internal.Str_ViewA==Str_Common.EMPTY:
            raise Exception("Initial Stereo Pair A error - could not find " + Obj_ReconDetails_internal.Str_SpecifiedViewA)
        if Obj_ReconDetails_internal.Str_ViewB==Str_Common.EMPTY:
            raise Exception("Initial Stereo Pair B error - could not find " + Obj_ReconDetails_internal.Str_SpecifiedViewB)
              
        
    #need to change current working directory or mvs/mvg dumps dmap files
    #which if not tidied up can corrupt next reconstruction
    os.chdir(Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput)

      
def get_exitcode_stdout_stderr(int_cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    #example of use:
    
    -arg_cmds=None
    -arg_cmds=["notepad.exe",
          "C:/UDemyOpenCV1/CUSTOM/SFM_Libs/OutputImages/sss.txt"]
    -exitcode, out, err = get_exitcode_stdout_stderr(arg_cmds)
    """
    start = time.time()
    #int_cmd[0]="\"" + int_cmd[0] + "\""#have to wrap in quotes in case we have a space in the filepath
    print("Executing External cmd", int_cmd[0])
    #doesnt work so well with a list - lets combine all arguments into a string
    CombineToString=" ".join(int_cmd)
    #CombineToString=CombineToString.replace("//","/")
    print("String sent to external process : " + CombineToString)
    proc = Popen(CombineToString, stdout=PIPE, stderr=PIPE,shell=True)
    out, err = proc.communicate()
    exitcode = proc.returncode
    #
    end = time.time()
    print("time ", round(end-start), " seconds")
    return exitcode, out, err



    
def ExecuteExternalReconLibraries(Obj_ReconDetails_internal): 
    def build_Executable_Combo(Input_List_Of_Arguments, MasterList):
            MasterList.append(Input_List_Of_Arguments)
            return MasterList   
    List_of_arg_cmds=[]
    
    #build executable and argument combos here
    
    
    #create sfm_data.json file - moved image dataset stuff
    #[-i|–imageDirectory]
    #[-d|–sensorWidthDatabase]
    #[-o|–outputDirectory] 
    #[-g|–group_camera_model]
    # – 0-> each view have it’s own camera intrinsic parameters
    #If synthetic images/no exif data - force a focal length- use: 1.2 * max(image_width, image_height) - EXAMPLE - " -f 6220.8" 
    #this is handled in ImageSet_ApproximatedFocalLength, which will be empty if no approximated focla length is needed (sensor data exists in EXIF and database)
    List_of_arg_cmds=build_Executable_Combo(
            [Obj_ReconDetails_internal.Folder_OpenMVG+"/openMVG_main_SfMInit_ImageListing ",
              " -i " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput,
              " -d " + Obj_ReconDetails_internal.Folder_OpenMVG + Obj_ReconDetails_internal.SensorWidthDBFileLocation,
              " -o " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_Output_Matches,
              " -g 0",Obj_ReconDetails_internal.ImageSet_ApproximatedFocalLength],
             List_of_arg_cmds)
    
    
    #Compute image description for a given sfm_data.json ﬁle.
    # For each view it compute the image description (local regions) and store them on disk
    #creates image_discriber.json
    # [-i|–input_ﬁle] – a SfM_Data ﬁle 
    # [-o|–outdirpath] – path were image description will be stored 
    # [-p|–describerPreset] – Used to control the Image_describer conﬁguration: * NORMAL, * HIGH, * ULTRA: !!Can be time consuming!! 
    List_of_arg_cmds=build_Executable_Combo(
            [Obj_ReconDetails_internal.Folder_OpenMVG+"/openMVG_main_ComputeFeatures ",
              " -p HIGH",
              " -i " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_Output_Matches + Obj_ReconDetails_internal.File_SFmJson,
              " -o " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_Output_Matches],
             List_of_arg_cmds)
    
    
    #computes images that have a visual overlap. Using image descriptions computed by openMVG_main_ComputeFeatures,
    #establishes the corresponding photometric matches and refines
    #the resulting correspondences using some robust geometric ﬁlters.
    #"Neato" fails here no matter what
    #creates *matches file
    
    # [-i|–input_ﬁle] – a SfM_Data ﬁle 
    # [-o|–outdirpath] – path were putative and geometric matches will be stored
    # [-r|-ratio] – (NearestNeighbordistanceratio,defaultvalueissetto0.8). Using 0.6 is more restrictive => provides less false positive. 
    List_of_arg_cmds=build_Executable_Combo(
            [Obj_ReconDetails_internal.Folder_OpenMVG+"/openMVG_main_ComputeMatches ",
            " -r .8",
            " -i " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_Output_Matches +Obj_ReconDetails_internal.File_SFmJson,
            " -o " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_Output_Matches],
             List_of_arg_cmds)
    
    #input of camera calibration and pairwise point correspondences
    #outs a 3D pointcloud with camera poses..
    #bundle adjustment and levenberg-marquadt steps (?) also used to correct drifting
    #for the succesive triangulated points 
    #creates Resection.ply files, cloud_and_poses.ply, initialPair.ply, sfm_data.bin in IncrementalReconstruction folder
    
    #sfm_data.bin has the views, extrinsics, instrinsics, landmarks (?), linked images
    # [-i|–input_ﬁle] – a SfM_Data ﬁle
    # [-m|–matchdir] – path were geometric matches were stored 
    # [-o|–outdir] – path where the output data will be stored 
    # [-a|–initialPairA NAME] – the ﬁlename image to use (i.e. 100_7001.JPG
    # [-b|–initialPairB NAME] – the ﬁlename image to use (i.e. 100_7002.JPG)
    
    #-f here is used to disable PRINCIPLE POINT - principle point was creating complexity when importing
    #the camera pose to 3D software - as the shifting of the camera frustrum on top of a camera rotation
    #was making reprojection of DLIB points much more difficult
    
    #" -f ADJUST_DISTORTION " gives good results
    #" -f ADJUST_FOCAL_LENGTH|ADJUST_DISTORTION " doesnt seem to work! this is the ideal setting    
    #" -f ADJUST_FOCAL_LENGTH " seems to be best we can do with the options if we can't do more than one - keep an eye on mesh quality
    #
    #set default principle point deactivated
    RefineInstrinsics=" -f \"ADJUST_FOCAL_LENGTH|ADJUST_DISTORTION\" "

    #get user configuration options 
    Success,EnumOfType=GetUser_Parameter_For_Feature(Obj_ReconDetails_internal.UserConfigFile, PrinciplePoint_state, "@")
    if Success==True:
        #if managed to find config file and parse the user config switch - set the principle point accordingly
        PrinciplePoint_used=EnumOfType
        if PrinciplePoint_used==PrinciplePoint_state.TRUE:
            RefineInstrinsics=""
            print("setting principle point ACTIVE according to user settings in config file")
        else:
            RefineInstrinsics=" -f \"ADJUST_FOCAL_LENGTH|ADJUST_DISTORTION\" "
            print("setting principle point DISABLED according to user settings in config file")
    
     

    #RefineInstrinsics=""
    print("openMVG_main_IncrementalSfM: PRINCIPLE POINT state - RefineInstrinsics : ",RefineInstrinsics)
    
    
    if (Obj_ReconDetails_internal.Str_ViewA=="" or Obj_ReconDetails_internal.Str_ViewB=="") or (Obj_ReconDetails_internal.Str_ViewA==Str_Common.EMPTY or  Obj_ReconDetails_internal.Str_ViewB==Str_Common.EMPTY) or (Obj_ReconDetails_internal.Str_ViewA is None or Obj_ReconDetails_internal.Str_ViewB is None):
         List_of_arg_cmds=build_Executable_Combo(
                [Obj_ReconDetails_internal.Folder_OpenMVG+"/openMVG_main_IncrementalSfM ",
                 " -i " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_Output_Matches +Obj_ReconDetails_internal.File_SFmJson,
                 " -m " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_Output_Matches,
                 " -o " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_IncrmtRecon,
                 RefineInstrinsics
                 ],
                 List_of_arg_cmds)
         
    else:
        List_of_arg_cmds=build_Executable_Combo(
                [Obj_ReconDetails_internal.Folder_OpenMVG+"/openMVG_main_IncrementalSfM ",
                 " -i " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_Output_Matches +Obj_ReconDetails_internal.File_SFmJson,
                 " -m " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_Output_Matches,
                 " -o " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_IncrmtRecon,
                 " -a " + Obj_ReconDetails_internal.Str_ViewA,#forcing our choices of initial cameras here
                 " -b " + Obj_ReconDetails_internal.Str_ViewB,
                 RefineInstrinsics
                 ],
                 List_of_arg_cmds)
        


    #computes corresponding features
    #robustly triangulate according to known poses and camera instrinsics
    #not sure why we need this step!!
    # [-i|–input_ﬁle] – a SfM_Data ﬁle with valid intrinsics and poses and optional structure 
    # [-m|–matchdir] – path were image descriptions were stored
    # [-o|–outdir] – path where the updated scene data will be stored 
    # [-f|–match_ﬁle] – path to a matches ﬁle (pairs of the match ﬁles will be listed and used) 
    # List_of_arg_cmds=build_Executable_Combo(
    #         [Obj_ReconDetails_internal.Folder_OpenMVG + "/openMVG_main_ComputeStructureFromKnownPoses ",
    #          " -i " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_IncrmtRecon + Obj_ReconDetails_internal.File_SFmData,
    #          " -m " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_Output_Matches,
    #          " -f " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_Output_Matches +"/matches.f.bin ",
    #          " -o " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput,
    #          Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_IncrmtRecon + "/robust.bin "
    #          ],
    #          List_of_arg_cmds)
    
    
    
    ##"Cannot find .ply here no matter what - doesnt exist at this stage - maybe error in source batch
    ##file - doesnt effect results
    #List_of_arg_cmds=build_Executable_Combo(
    #        [Folder_OpenMVG + "/openMVG_main_ComputeSfM_DataColor ",
    #        "-i " + Folder_WorkingDirectoryOutput +Folder_IncrmtRecon +"/robust_colorized.ply"
    #        ],
    #         List_of_arg_cmds)

    
        #testing - export camera frustums
    #[-i|–--input_file] path to a SfM_Data scene
    #[-o|–-output_file] PLY file to store the camera frustums as triangle meshes.
    List_of_arg_cmds=build_Executable_Combo(
            [Obj_ReconDetails_internal.Folder_OpenMVG + "/openMVG_main_ExportCameraFrustums ",
              " -i " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_IncrmtRecon + Obj_ReconDetails_internal.File_SFmData,
              " -o " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output + "/CameraFrustrums.ply "
              ],
              List_of_arg_cmds)


    #we can export data from the sfm_data.bin file (contains all sfm data like poses etc)
    #output file should allow us to match pose with image file in Cloud and Poses . ply file
    
    #optional extras -V for views, -I for Intrinsics, -E for extrinsics, S for structure,  C control points
    #or leave out all options for everything
    List_of_arg_cmds=build_Executable_Combo(
            [Obj_ReconDetails_internal.Folder_OpenMVG + "/openMVG_main_ConvertSfM_DataFormat ",
             " -i " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_IncrmtRecon + "/" + Obj_ReconDetails_internal.File_SFmData,
             " -o " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output + "/" + Obj_ReconDetails_internal.Sfm_data_Exported + " ",
             " -V ",
             " -I ",
             " -E "
             ],
                 List_of_arg_cmds)
        
#Now dense reconstruction/meshing

    #debug


    #creates Model.mvs and copied image files
    List_of_arg_cmds=build_Executable_Combo(
            [Obj_ReconDetails_internal.Folder_OpenMVG + "/openMVG_main_openMVG2openMVS ",
             "-i " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.Folder_IncrmtRecon + Obj_ReconDetails_internal.File_SFmData,
             "-o " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output +Obj_ReconDetails_internal.File_ReconModel, 
             "-d " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output],
             List_of_arg_cmds)
    
    #creates model_dense.mvs and model_dense.ply
    List_of_arg_cmds=build_Executable_Combo(
            [Obj_ReconDetails_internal.Folder_OpenMVS + "/DensifyPointCloud.exe ",
             "-i " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output+Obj_ReconDetails_internal.File_ReconModel],
             List_of_arg_cmds)
    
    List_of_arg_cmds=build_Executable_Combo(
            [Obj_ReconDetails_internal.Folder_OpenMVS + "/ReconstructMesh.exe  ",
             Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output+"/model_dense.mvs "],
             List_of_arg_cmds)
    
    List_of_arg_cmds=build_Executable_Combo(
            [Obj_ReconDetails_internal.Folder_OpenMVS + "/RefineMesh.exe  ",
             "--resolution-level 2 "  + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output+"/model_dense_mesh.mvs"],
             List_of_arg_cmds)
    
    List_of_arg_cmds=build_Executable_Combo(
            [Obj_ReconDetails_internal.Folder_OpenMVS + "/TextureMesh.exe  ",
             Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output+"/model_dense_mesh_refine.mvs"],
             List_of_arg_cmds)
    
    start_total = time.time()
    for ArgList in List_of_arg_cmds:
        print("----")
        exitcode, out, err = get_exitcode_stdout_stderr(ArgList)
        
        #input("Press Enter, have completed " + ArgList[0])
        #TODO check that "0" is good exit good for all external processes
        if (str(exitcode)!="0"):
            #default failure
            logging.exception("Error running " + ArgList[0] + "err=" + str(err),exc_info=True)#full stack trace
            print("")
            if "Perform incremental SfM" in str(out):
                print("this error may be related to forcing two initial cameras - try disabling this option and run again")
            if "Please consider add your camera model and sensor width" in str(err):
                print("fear me")
            raise Exception("Error running " + " arglist: \n "  + str(ArgList) + "\n err: \n" + str(err))

        print("Return Err from Command: " + str(err))
        print("   Exit code: "  + str(exitcode))
        if "Please consider add your camera model and sensor width" in str(err):
            print("Warning: Adding camera sensor detail very flakey.. Examine EXIF image data to get vender and model.. ")
            print("try removing vender or checking capitalisation or adding multiple instances eg Samsung SM-A415F;6.4 samsung SM-A415F;6.4 SM-A415F;6.4")
            print("Database file is: " + Obj_ReconDetails_internal.Folder_OpenMVG + Obj_ReconDetails_internal.SensorWidthDBFileLocation)
        print("----")
    end_total = time.time()
    print("total time ", round(end_total-start_total), " seconds")


def ExecuteScaling(Obj_ReconDetails_internal,ScaleFactor):
    def build_Executable_Combo(Input_List_Of_Arguments, MasterList):
            MasterList.append(Input_List_Of_Arguments)
            return MasterList   
    List_of_arg_cmds=[]
    
    #create script file
    TempScript=Obj_ReconDetails_internal.MeshLabScript
    TempScript=TempScript.replace(Obj_ReconDetails_internal.MeshLabScriptDummyTxt,'"' + str(round(ScaleFactor,1)) + '"')
    file = open(Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output+Obj_ReconDetails_internal.MeshlabScript_Name,'w')
    file.write(TempScript)
    file.close() 

    #now lets use meshlabserver to resize the final mesh by this scaling factor
    List_of_arg_cmds=build_Executable_Combo(
                [Obj_ReconDetails_internal.MeshLabServerEXE,
                 " -i " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output+ Obj_ReconDetails_internal.File_FinalMesh,
                 " -o " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output+ Obj_ReconDetails_internal.ScaledMeshFileName + " -m wt",#need to export wedge texture coords as well
                 " -s " + Obj_ReconDetails_internal.Folder_WorkingDirectoryOutput + Obj_ReconDetails_internal.MVS_Output+ Obj_ReconDetails_internal.MeshlabScript_Name],
                List_of_arg_cmds)
    for ArgList in List_of_arg_cmds:
        exitcode, out, err = get_exitcode_stdout_stderr(ArgList)
        print("----")
        print(err)





def ReconstructImages_ToMesh(
                            Folder_ReconRoot="",#root working folder
                            Str_SpecifiedViewA="",#Optional, If camera distances are known
                            Str_SpecifiedViewB="",#Optional, If camera distances are known
                            MeshLabServerEXE="",
                            NumberOfViews=0,#Optional if known 
                            Try_to_Scale=False,#Optional if known 
                            CameraPair_Distance_mm=1,#distance between cameras if known
                            AskBeforeDeleteFolders=True):
    #TODO this is not good coding practise to put the import here
    #currently necessary as using a lot of different environments
    import exifread
    from PIL import Image, ExifTags

    #test Inputs
    test_String(Folder_ReconRoot)
    test_Path(Folder_ReconRoot)
    Folder_ReconRoot=Correct_FilePath_Slashes(Folder_ReconRoot)
    
    # need to check if server exists if scaling 
    if Try_to_Scale==True:
        test_String(MeshLabServerEXE)
        MeshLabServerEXE=Correct_FilePath_Slashes(MeshLabServerEXE)
        test_file(MeshLabServerEXE)


    #initalise working details
    Obj_ReconDetails= ReconstructionDetails(
                                Folder_ReconRoot,
                                Str_SpecifiedViewA,
                                Str_SpecifiedViewB,
                                MeshLabServerEXE,
                                NumberOfViews)
    #initialise scaling working details
    Data_ScaleMesh_to_realworld=Data_ScaleMesh(CameraPair_Distance_mm)








    #Detect if images are synthetic or do not come with sensor information in the EXIF, and create approximation of focal length instead
    #Possibly here would be a good place to check if the sensor information is included in the sensor database, as that is a point of failure
    
    #initial image check format
    CheckHomogenousImageFormat_Size=Str_Common.EMPTY
    CheckHomogenousImageFormat_make=Str_Common.EMPTY
    CheckHomogenousImageFormat_model=Str_Common.EMPTY
    

    #generate file string
    InputImg_Folder=Obj_ReconDetails.Folder_WorkingDirectoryInput +"/"
    #get all valid images from input folder
    List_Files_in_InitalImages_folder =GetList_Of_ImagesInfolder(InputImg_Folder)
    #roll thorugh images and check make/model and dimension data, ensuring all images are same format
    #if images do not have make/model OR make/model is not in sensor database, we
    #need to detect this and generate an approximation of focal lengths using the dimension data
    for Indexer, ImageFile in enumerate(List_Files_in_InitalImages_folder):
        #subsequent images check format
        CheckHomogenousImageFormat_Size_Checker=Str_Common.EMPTY
        CheckHomogenousImageFormat_make_Checker=Str_Common.EMPTY
        CheckHomogenousImageFormat_model_Checker=Str_Common.EMPTY
        # Read image with EXIF data using PIL/Pillow
        f = open(InputImg_Folder + ImageFile, 'rb')
        tags = exifread.process_file(f)
        #read dimension data - use a different system that won't break if no EXIF data
        img = Image.open((InputImg_Folder + ImageFile))

        #get first item and use to check all images are same format
        if (Indexer==0):
            try:
                CheckHomogenousImageFormat_make=tags["Image Make"].values
                CheckHomogenousImageFormat_model=tags["Image Model"].values
            except:
                #no exif data exists 
                pass

            try:
                CheckHomogenousImageFormat_Size=img.size
            except:
                #no image size data exists - this is major error
                raise Exception("No image dimensions exist for file, ",InputImg_Folder + ImageFile)
            print("Check input images EXIF for format homogenity: ", ImageFile,CheckHomogenousImageFormat_make,CheckHomogenousImageFormat_model,CheckHomogenousImageFormat_Size)
            
        else:
            #check following images to test format homogenousity
            try:
                CheckHomogenousImageFormat_make_Checker=tags["Image Make"].values
                CheckHomogenousImageFormat_model_Checker=tags["Image Model"].values
            except:
                #no exif data exists 
                pass
            
            try:
                CheckHomogenousImageFormat_Size_Checker=img.size
            except:
                #no image size data exists - this is major error
                raise Exception("No image dimensions exist for file, ",InputImg_Folder + ImageFile)
            
            print("Check input images EXIF for format homogenity: ", ImageFile,CheckHomogenousImageFormat_make_Checker,CheckHomogenousImageFormat_model_Checker,CheckHomogenousImageFormat_Size_Checker)
            
            #check current image in folder matches first image
            if (CheckHomogenousImageFormat_make_Checker!=CheckHomogenousImageFormat_make) or (CheckHomogenousImageFormat_model_Checker!=CheckHomogenousImageFormat_model) or (CheckHomogenousImageFormat_Size !=CheckHomogenousImageFormat_Size_Checker):
                print("exif data image comparison a/b a/b a/b a/b")
                print(CheckHomogenousImageFormat_make_Checker,CheckHomogenousImageFormat_make,CheckHomogenousImageFormat_model_Checker,CheckHomogenousImageFormat_model,CheckHomogenousImageFormat_Size,CheckHomogenousImageFormat_Size_Checker )
                raise Exception("Please homogenise: Potential image formats non-homogenous in input folder, checking dimensions, Make and Model ",InputImg_Folder )
            


    #If synthetic images/no exif data - force a focal length- use: 1.2 * max(image_width, image_height) - EXAMPLE - " -f 6220.8"
    #set data object image parameters
    Obj_ReconDetails.ImageSet_Dimensions=CheckHomogenousImageFormat_Size
    Obj_ReconDetails.ImageSet_EXIF_Make=CheckHomogenousImageFormat_make
    Obj_ReconDetails.ImageSet_EXIF_Model=CheckHomogenousImageFormat_model
    #if no exif make/model - we have to approximate the focal length
    if Obj_ReconDetails.ImageSet_EXIF_Make==Str_Common.EMPTY:
        print("No EXIF data found - generating focal length from approximation formula 1.2 * max(image_width, image_height), generating input parameter string")
        FL=max(Obj_ReconDetails.ImageSet_Dimensions[0],Obj_ReconDetails.ImageSet_Dimensions[1])*1.2
        Obj_ReconDetails.ImageSet_ApproximatedFocalLength=str(" -f " + str(FL))
        print(Obj_ReconDetails.ImageSet_ApproximatedFocalLength)
    
    

    #check initial image folder
    InitialImg_Folder=Obj_ReconDetails.Folder_ReconRoot +Obj_ReconDetails.InitialImageFolder +"/"
    print("Checking Initial Image pair folder (populate with 2 initial images if required)", InitialImg_Folder)
    #process has optional initial image mode - user can dictate this by leaving a copy of initial images in a folder
    #check user hasnt deleted them in INPUT folder by checking for existence and copying them back in
    #check folder exists
    if os.path.exists(InitialImg_Folder):
        #get list of images in folder
        List_Files_in_InitalImages_folder = GetList_Of_ImagesInfolder(InitialImg_Folder)

        if len(List_Files_in_InitalImages_folder) == 2:
            #folder is populated with two files (should be images)
            #we should check user hasnt accidently "cut" the files and thus removed them from the image pool in INPUTIMAGES folder,
            #or left them there from a previous process
            for InitialImageFile in List_Files_in_InitalImages_folder:
                #files are filename format without path
                if not os.path.exists(Obj_ReconDetails.Folder_WorkingDirectoryInput + "/" + InitialImageFile):
                    raise Exception("Initial User file ",InitialImg_Folder+InitialImageFile , " not found in INPUTIMAGES folder! ",Obj_ReconDetails.Folder_WorkingDirectoryInput, ", either previous session contamination or cut and paste error")

            #now populate initial image A and B with images from list
            #make user aware that there is an order which could ultimately effect pose of output
            Obj_ReconDetails.Str_SpecifiedViewA=List_Files_in_InitalImages_folder[0]
            print("Initial Image A set to ", List_Files_in_InitalImages_folder[0])
            Obj_ReconDetails.Str_SpecifiedViewB = List_Files_in_InitalImages_folder[1]
            print("Initial Image B set to ", List_Files_in_InitalImages_folder[1])
            
        else:
            #dont do anything - folder exists but user has not populated it
            print("Initial Images <> 2, no initial images set for sessions")


    #folder does not exist - autogenerate for next time
    else:
        print("Initial image folder does not exist - creating for next session")
        os.makedirs(InitialImg_Folder)


    
    #delete output folder
    #copy images from input folder to output folder
    #locate origin camera pairs if enabled
    PrepareFolderStructure(Obj_ReconDetails,AskBeforeDeleteFolders)
    
    #reconstruct textured mesh from images - results in Output Folder
    ExecuteExternalReconLibraries(Obj_ReconDetails)
    
    if Try_to_Scale==True:
        #experimental alternative method to get scaling factor
        TemporaryDistance=ExtractSfm_Details_From_Sfm_datafile(Obj_ReconDetails)
        if (TemporaryDistance is not None):
            Obj_ReconDetails.Sfm_data_Exported_InitPair_Distance=TemporaryDistance
            ScaleFactor=Data_ScaleMesh_to_realworld.InitialPair_mm/Obj_ReconDetails.Sfm_data_Exported_InitPair_Distance
            ExecuteScaling(Obj_ReconDetails,ScaleFactor)
            print("scaling factor",round(ScaleFactor,1) )
        
    #set default folder as otherwise will lock OUTPUT folder..
    #TODO this should be somewhere more sensible
    os.chdir(Folder_ReconRoot)

    #create report dictionary for next process
    ReconstructionReport={}
    ReconstructionReport["InputFolder"]= Correct_FilePath_ForwardSlashes(str(Obj_ReconDetails.Folder_WorkingDirectoryInput))#add input directory of images
    ReconstructionReport["OutputFolder"]=Correct_FilePath_ForwardSlashes(str(Obj_ReconDetails.Folder_WorkingDirectoryOutput + Obj_ReconDetails.MVS_Output))#add output directory for all sfm output
    ReconstructionReport["InitialImages"]=[Obj_ReconDetails.Str_SpecifiedViewA, Obj_ReconDetails.Str_SpecifiedViewA]#what two initial imgae pair have been used 
    JSON_Save(Obj_ReconDetails.Folder_WorkingDirectoryOutput + Obj_ReconDetails.MVS_Output + '/SFM_FileDetails.json', ReconstructionReport)#save to output folder
    JSON_Save(Obj_ReconDetails.Folder_ReconRoot + '/SFM_FileDetails.json', ReconstructionReport)#save to location process was run - so next process can pick up 
    print("Output folder at : ", ReconstructionReport["OutputFolder"])
    return ReconstructionReport
        
def Correct_FilePath_ForwardSlashes(InputPath):
    ModifiedStr=InputPath.replace("//","/")
    return ModifiedStr

def Correct_FilePath_Slashes(InputPath):
    ModifiedStr=InputPath.replace("\\","/")
    return ModifiedStr
def Correct_filename_leadSlash(InputString):
    """corrects slash formatting at start of filename (without path)"""
    StringList=list(InputString)
    if StringList[0]=="\\":#check for SINGLE backslash
        StringList[0]="/"
    if StringList[0]!="/":
        StringList.insert(0,"/")
    return "".join(StringList)
def test_file(InputFile):
    """Checks a file is valid - if not asserts an error"""
    assert(os.path.isfile(InputFile) ),"File is not found " + str(InputFile)
def test_String(InputString):
    """Checks a string is valid - if not asserts an error"""
    assert("".__eq__(InputString)==False),"String is not valid "
def test_Path(InputPath):
    """Checks a PATH is valid - if not asserts an error"""
    assert(os.path.isdir(InputPath) ),"Path is not found " + str(InputPath)
    
    
    
    
class StopWatch:
    def __init__(self):
        self.start()
    def start(self):
        self._startTime = time.time()
    def getStartTime(self):
        return self._startTime
    def elapsed(self, prec=3):
        prec = 3 if prec is None or not isinstance(prec, (int, long)) else prec
        diff= time.time() - self._startTime
        return round(diff, prec)
def round(n, p=0):
    m = 10 ** p
    return math.floor(n * m + 0.5) / m    

def GetFacePoses(FaceLandMark_Returns_Object : NamedTuples_Common.FaceLandMarkReturns,FaceLandMarks_Template3DPositions, FacePoseImportingSoftware=None):
    
    #TODO this is a bit backwards
    #should be LIST of TUPLES rather than broke up in this manner
    FacePoseResults=[]
    #input structure is a list of facelandmarks (list per face found)
    #and input image (single image for multiple faces)
    for Indexer in range(FaceLandMark_Returns_Object.FaceCount):
        SingleFaceLandmarkResult=NamedTuples_Common.FaceLandMarkReturns(1,
                                                                FaceLandMark_Returns_Object.OriginalImage,
                                                                FaceLandMark_Returns_Object.ImageWithLandMarks,
                                                                FaceLandMark_Returns_Object.ListOfFaceLandMarks[Indexer])
        #send single face landmark result to pose solver
        SingleFacePoseResult=GetFacePose_Transform_Single(SingleFaceLandmarkResult,FaceLandMarks_Template3DPositions,FacePoseImportingSoftware)
        #add results to list for return object
        FacePoseResults.append(SingleFacePoseResult)
    
    #return LIST of NAMED TUPLES, each tuple a collection of results for each face
    return FacePoseResults
 

def Convert_solvepnp_translation(InputSoftwareName_fromStrLib,tvec_fromSolvepnp,rvec_fromSolvePnp):
    """convert compact translation vector from Solvepnp to Importing software (even Opencv)
    return XYZ array of Camera POSITION in Importing softwares' particular coordinate system
    and handy string for debugging"""
    if InputSoftwareName_fromStrLib==Str_Common.Softwares3D.UnrealEngine:
        
        #rvec from solvepnp is in "compact rodrigues form" - we need to convert to rotation matrix
        rotM=cv2.Rodrigues(rvec_fromSolvePnp)[0]#two results from this -[0] is the rotation matrix
        #get camera position in world coordinate system
        cameraPosition= - np.matrix(rotM).T * np.matrix(tvec_fromSolvepnp)
        #make new tvec array
        ReturnXYZ= np.zeros(3)
        ReturnXYZ.reshape(3,1)
        #now convert axis to UNREAL ENGINE coordinate system:
        ReturnXYZ[0]=(round(-cameraPosition[2],2))#UE X = -Opencv Z
        ReturnXYZ[1]=(round(-cameraPosition[0],2))#UE Y = -Opencv X
        ReturnXYZ[2]=(round(cameraPosition[1],2))#UE Z = Opencv Y
        #generate handy string for printing on debug images
        CameraPositionStr="UE cam pos X=" + str(ReturnXYZ[0]) + " Y=" + str(ReturnXYZ[1]) + " z=" + str(ReturnXYZ[2])
        return ReturnXYZ,CameraPositionStr
    
    if InputSoftwareName_fromStrLib==Str_Common.Softwares3D.Blender:
        
        #rvec from solvepnp is in "compact rodrigues form" - we need to convert to rotation matrix
        rotM=cv2.Rodrigues(rvec_fromSolvePnp)[0]#two results from this -[0] is the rotation matrix
        #get camera position in world coordinate system
        cameraPosition= - np.matrix(rotM).T * np.matrix(tvec_fromSolvepnp)#transform of rodrigues rot matrix * tvec, then inversed
        #make new tvec array
        ReturnXYZ= np.zeros(3)
        ReturnXYZ.reshape(3,1)
        ReturnXYZ[0]=(round(cameraPosition[0],2))
        ReturnXYZ[1]=(round(cameraPosition[2],2))*-1
        ReturnXYZ[2]=(round(-cameraPosition[1],2))
        #generate handy string for printing on debug images
        CameraPositionStr="UNTESTED Blender cam pos X=" + str(ReturnXYZ[0]) + " Y=" + str(ReturnXYZ[1]) + " z=" + str(ReturnXYZ[2])
        return ReturnXYZ,str(CameraPositionStr)

    if InputSoftwareName_fromStrLib == Str_Common.Softwares3D.Maya:
        # rvec from solvepnp is in "compact rodrigues form" - we need to convert to rotation matrix
        rotM = cv2.Rodrigues(rvec_fromSolvePnp)[0]  # two results from this -[0] is the rotation matrix
        # get camera position in world coordinate system
        cameraPosition = - np.matrix(rotM).T * np.matrix(
            tvec_fromSolvepnp)  # transform of rodrigues rot matrix * tvec, then inversed
        # make new tvec array
        ReturnXYZ = np.zeros(3)
        ReturnXYZ.reshape(3, 1)
        ReturnXYZ[0] = (round(-cameraPosition[0], 2))
        ReturnXYZ[1] = (round(cameraPosition[1], 2))
        ReturnXYZ[2] = (round(cameraPosition[2], 2))
        # generate handy string for printing on debug images
        CameraPositionStr = "MAYA cam pos X=" + str(ReturnXYZ[0]) + " Y=" + str(ReturnXYZ[1]) + " z=" + str(
            ReturnXYZ[2])
        return ReturnXYZ, str(CameraPositionStr)
    if InputSoftwareName_fromStrLib==Str_Common.Softwares3D.OpenCV:
        
        #rvec from solvepnp is in "compact rodrigues form" - we need to convert to rotation matrix
        rotM=cv2.Rodrigues(rvec_fromSolvePnp)[0]#two results from this -[0] is the rotation matrix
        #get camera position in world coordinate system
        cameraPosition= - np.matrix(rotM).T * np.matrix(tvec_fromSolvepnp)
        #make new tvec array
        ReturnXYZ= np.zeros(3)
        ReturnXYZ.reshape(3,1)
        #now convert axis to UNREAL ENGINE coordinate system:
        ReturnXYZ[0]=(round(cameraPosition[0],2))
        ReturnXYZ[1]=(round(cameraPosition[1],2))
        ReturnXYZ[2]=(round(cameraPosition[2],2))
        #generate handy string for printing on debug images
        CameraPositionStr="OpenCV cam pos X=" + str(ReturnXYZ[0]) + " Y=" + str(ReturnXYZ[1]) + " z=" + str(ReturnXYZ[2])
        return ReturnXYZ,str(CameraPositionStr)
    
    #default failure
    logging.exception("Convert_solvepnp_translation, error with Input software name, string :" + str(InputSoftwareName_fromStrLib) + " not matched",exc_info=True)#full stack trace
    raise Exception("Convert_solvepnp_translation, error with Input software name, string :" + str(InputSoftwareName_fromStrLib) + " not matched")
   
    
def Convert_solvepnp_rotation(InputSoftwareName_fromStrLib,tvec_fromSolvepnp,rvec_fromSolvePnp):
    """convert coordinate systems from solvepnp to importing software, 
    return XYZ array of Camera ROTATION in Importing softwares' particular coordinate system
    and handy string for debugging"""
    if InputSoftwareName_fromStrLib==Str_Common.Softwares3D.UnrealEngine:
        
        def AngleConverter(Element):
            """Angle conversion particular to Unreal Engine"""
            ReturnElement=round(Element*Numbers_Common.DegreePerRadian,1)#convert from radians
            return ReturnElement % 360 #modulo 360 
            
        ReturnXYZ= np.zeros(3)
        ReturnXYZ.reshape(3,1)
        #now convert axis to UNREAL ENGINE coordinate system:
        
        ReturnXYZ[0]=AngleConverter(rvec_fromSolvePnp[2])
        ReturnXYZ[1]=AngleConverter(rvec_fromSolvePnp[0])
        ReturnXYZ[2]=AngleConverter(rvec_fromSolvePnp[1])
        #generate handy string for printing on debug images
        CameraRotationStr="UE cam rot X=" + str(ReturnXYZ[0]) + " Y=" + str(ReturnXYZ[1]) + " z=" + str(ReturnXYZ[2])
        return ReturnXYZ,str(CameraRotationStr)
    
    
    if InputSoftwareName_fromStrLib==Str_Common.Softwares3D.Maya:
        
        def AngleConverter(Element):
            """Angle conversion particular to Maya """
            ReturnElement=round(Element*Numbers_Common.DegreePerRadian,1)#convert from radians
            if ReturnElement<0:
                ReturnElement=360-ReturnElement
            return ReturnElement % 360 #modulo 360 
            
        ReturnXYZ= np.zeros(3)
        ReturnXYZ.reshape(3,1)
        #now convert axis to MAYA coordinate system:
        
        ReturnXYZ[0]=AngleConverter(rvec_fromSolvePnp[0])
        ReturnXYZ[1]=-AngleConverter(rvec_fromSolvePnp[1])
        ReturnXYZ[2]=-AngleConverter(rvec_fromSolvePnp[2])
        #generate handy string for printing on debug images
        CameraRotationStr="Maya cam rot X=" + str(ReturnXYZ[0]) + " Y=" + str(ReturnXYZ[1]) + " z=" + str(ReturnXYZ[2])
        return ReturnXYZ,str(CameraRotationStr)

    if InputSoftwareName_fromStrLib == Str_Common.Softwares3D.Blender:

        def AngleConverter(Element):
            """Angle conversion particular to Maya first """
            ReturnElement = round(Element * Numbers_Common.DegreePerRadian, 1)  # convert from radians
            return ReturnElement % 360 #modulo 360 

        ReturnXYZ = np.zeros(3)
        ReturnXYZ.reshape(3, 1)


        ReturnXYZ[0]=(round(-AngleConverter(rvec_fromSolvePnp[0]),2))
        ReturnXYZ[1]=(round(-AngleConverter(rvec_fromSolvePnp[2]),2))
        ReturnXYZ[2]=(round(AngleConverter(rvec_fromSolvePnp[1]),2))

        # # now convert axis to MAYA coordinate system:

        # ReturnXYZ[0] = AngleConverter(rvec_fromSolvePnp[0])
        # ReturnXYZ[1] = -AngleConverter(rvec_fromSolvePnp[1])
        # ReturnXYZ[2] = -AngleConverter(rvec_fromSolvePnp[2])

        # #now convert to Blender from Maya
        # #posX = lightObj.matrix_world[0][3]
        # #posY = lightObj.matrix_world[2][3]
        # #posZ = -lightObj.matrix_world[1][3]  # note the negative

        # ReturnXYZb = np.zeros(3)
        # ReturnXYZb.reshape(3, 1)

        # ReturnXYZb[0] = -AngleConverter(rvec_fromSolvePnp[0])
        # ReturnXYZb[1] = -AngleConverter(rvec_fromSolvePnp[1])
        # ReturnXYZb[2] = -AngleConverter(rvec_fromSolvePnp[2])

        # generate handy string for printing on debug images
        CameraRotationStr = " UNTESTED Blender cam rot X=" + str(ReturnXYZ[0]) + " Y=" + str(ReturnXYZ[1]) + " z=" + str(
            ReturnXYZ[2])
        return ReturnXYZ, str(CameraRotationStr)



    if InputSoftwareName_fromStrLib==Str_Common.Softwares3D.OpenCV:
        logging.warning("Convert_solvepnp_rotation Converting angle to OPENCV not tested yet")
        def AngleConverter(Element):
            """Angle conversion particular to Opencv """
            ReturnElement=round(Element*Numbers_Common.DegreePerRadian,1)#convert from radians
            return ReturnElement % 360 #modulo 360 
            
        ReturnXYZ= np.zeros(3)
        ReturnXYZ.reshape(3,1)
        #now convert axis to MAYA coordinate system:
        
        ReturnXYZ[0]=AngleConverter(rvec_fromSolvePnp[0])
        ReturnXYZ[1]=AngleConverter(rvec_fromSolvePnp[1])
        ReturnXYZ[2]=AngleConverter(rvec_fromSolvePnp[2])
        #generate handy string for printing on debug images
        CameraRotationStr="OpenCV cam rot X=" + str(ReturnXYZ[0]) + " Y=" + str(ReturnXYZ[1]) + " z=" + str(ReturnXYZ[2])
        return ReturnXYZ,str(CameraRotationStr)    #default failure
   
    logging.exception("Convert_solvepnp_rotation, error with Input software name, string :" + str(InputSoftwareName_fromStrLib) + " not matched",exc_info=True)#full stack trace
    raise Exception("Convert_solvepnp_rotation, error with Input software name, string :" + str(InputSoftwareName_fromStrLib) + " not matched")



def isRotationMatrix(R):
    """# Checks if a matrix is a valid rotation matrix."""
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 

def rotationMatrixToEulerAngles(R):
    """ Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped )."""
    if (isRotationMatrix(R)) == False:
        raise Exception("rotationMatrixToEulerAngles input not a rotation matrix - fail")
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])
def GetFacePose_Transform_Single(FaceLandMark_Returns_Object : NamedTuples_Common.FaceLandMarkReturns,FaceLandMarks_Template3DPositions,FacePoseImportingSoftware=None):
    """handles single return from 3dvislab DLIB FaceLandmarks function
    and calculates 3d transform of face and camera details
    using corresponding static 3d points"""
    if FaceLandMark_Returns_Object.FaceCount <1 :
        print("GetFacePose_Transform_Single FAIL no data to analyse")
        return NamedTuples_Common.FacePoseReturns(0,FaceLandMark_Returns_Object.OriginalImage,None,None,None,None,None,None)
    NotesString=""
    NotesString=NotesString+ "Importing3DSofware=" + str(FacePoseImportingSoftware) + " "
    #scan through 3d points reference object - because we might have changed what reference
    #positions are used from the 68 points available
    
    #Reference points from 3d template are indexed correctly - but
    #2d landmark points returned by DLIB are 0 indexed
    IndexingOffset=1
    _2d_pts_FaceLandmarks=[]
    _3D_pts_FaceLandmarks=[]
    _3D_points_Ref_dict=FaceLandMarks_Template3DPositions 
    
    #initialise containers
    objectPoints_3D = np.zeros((len(_3D_points_Ref_dict),3,1))#create solvepnp compatible array
    imagePoints_2D = np.zeros((len(_3D_points_Ref_dict),2,1))#create solvepnp compatible array
    
    #get image details
    ImgWidthIndex=1
    ImgHeightIndex=0
    ImgWidth=FaceLandMark_Returns_Object.OriginalImage.shape[ImgWidthIndex]
    ImgHeight=FaceLandMark_Returns_Object.OriginalImage.shape[ImgHeightIndex]
    ImgCentre_X=int(FaceLandMark_Returns_Object.OriginalImage.shape[ImgWidthIndex]/2)
    ImgCentre_Y=int(FaceLandMark_Returns_Object.OriginalImage.shape[ImgHeightIndex]/2)
    
    #approximate CAMERA MATRIX (intrinsics)
    cameraMatrix = np.eye(3)#approximate center by centre of image (pixels) and approximate focal length by width image (pixels)
    cameraMatrix[0,0]=ImgWidth#approximate fx
    cameraMatrix[1,1]=ImgWidth#approximate fy
    cameraMatrix[0,2]=ImgCentre_X#approximate image cx
    cameraMatrix[1,2]=ImgCentre_Y#approximate image cx
    NotesString=NotesString+ "CamMatrix: " + str(cameraMatrix)
    #approximate distortion matrix
    distCoeffs = np.zeros((5,1))    
    
    #-------------
    #BUILD matching length containers of corresponding 3D and 2D landmark points
    #-------------
    #Input 3D static points (landmarks on 3D model)
    #dictate what dynamic 2D points we will use
    #_3D_points_Ref_dict is pre-configured dictionary of static 3D face landmarks
    #in form [key=FaceLandmark DLIB index] [x,y,z] ["description"]
    #generally DLIB will find all 68 2D points, but the 3D library might have
    #only 5 basic points that the user has pre-configured 
    for LandmarkRef in _3D_points_Ref_dict:
        #loop through static 3D points - for each DLIB Index found - grab corresponding Index
        #of dynamic 2d points found by face landmark detection system
        try:
            #INDEX (key) of static 3d reference points
            #use index to populate 2d and 3d arrays with corresponding points
            Temp2d_Point=(FaceLandMark_Returns_Object.ListOfFaceLandMarks[int(LandmarkRef)-IndexingOffset])
            #populate X/Y 
            _2d_pts_FaceLandmarks.append([Temp2d_Point[0,0],Temp2d_Point[0,1]])
            #unpack tuple using *
            _3D_pts_FaceLandmarks.append([*_3D_points_Ref_dict[LandmarkRef][0]])

        except Exception as e:
            logging.critical("Error making 3D/2D face landmark list correspondences for pose solver, " + str(e) ,exc_info=True)#full stack trace
            raise Exception ("Error making 3D/2D face landmark list correspondences for pose solver, " + str(e))

    if len(_3D_pts_FaceLandmarks)!=len(_3D_points_Ref_dict):
        logging.warning("Warning, user defined 3d face landmarks not length matched with 2d points",exc_info=True)#full stack trace
        raise Exception ("Warning, user defined 3d face landmarks not length matched with 2d points")
        print("Warning, user defined 3d face landmarks not length matched with 2d points")
    #at this stage _2d_pts_FaceLandmarks and _3D_pts_FaceLandmarks should
    #both be the same length
    #and each element position correspond to same DLIB face landmark index
    #IE index of 2D tip of nose will be same as index of 3D tip of nose
    
    
    #populate 2d/3d correspondence structure compatible with opencv solvepnp function
    #TODO must be a more pythonic way to do this
    for x in range(0,len(_2d_pts_FaceLandmarks)):
        
        imagePoints_2D[x][0]=_2d_pts_FaceLandmarks[x][0]
        imagePoints_2D[x][1]=_2d_pts_FaceLandmarks[x][1]
        objectPoints_3D[x][0]=_3D_pts_FaceLandmarks[x][0]
        objectPoints_3D[x][1]=_3D_pts_FaceLandmarks[x][1]
        objectPoints_3D[x][2]=_3D_pts_FaceLandmarks[x][2]
        
    #run the 2d/3d correspondence solver
    #lots of methods here
    #TODO research pnp solvers
    retval, rvec_frmSolvePnp, tvec_frmSolvePnp=cv2.solvePnP(objectPoints_3D, imagePoints_2D, cameraMatrix, distCoeffs)
    NotesString = NotesString + " Pose solver is solvePnP "
    #if pose solver fails, return empty list
    if retval==False:
        return NamedTuples_Common.FacePoseReturns(0,FaceLandMark_Returns_Object.OriginalImage,None,None,None,None,None,None)


    #print(retval,rvec_frmSolvePnp)
    
    #DRAW illustrative face pose orientation line on image
    #nose is index 31 using DLIB landmark numbering system
    NoseStartPoint_Found2d=(FaceLandMark_Returns_Object.ListOfFaceLandMarks[int(31)-IndexingOffset])
    #rotate a vector to match solved pose of 2D face
    #This function might just create a projection matrix and then multiply each point 
    (NoseEndPoint_FromPose2d, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rvec_frmSolvePnp, tvec_frmSolvePnp, cameraMatrix, distCoeffs)
    #get copy of image with/without graphics from previous stage
    ImageWithPose=FaceLandMark_Returns_Object.ImageWithLandMarks.copy()
    ImageWithPose=cv2.line(ImageWithPose,
                           (int(NoseStartPoint_Found2d[0,0]),int(NoseStartPoint_Found2d[0,1])),
                           (int(NoseEndPoint_FromPose2d[0][0][0]),int(NoseEndPoint_FromPose2d[0][0][1])),
                           (255,0,0),2)

    #manually make projection matrix as experiment
    TestProjectMatrix,TestProjectMatrix2=GetProjectionMatrix_FacePoseFormat(ImportingSoftware_forPR=FacePoseImportingSoftware,
                                       Rotation_FromSolvepnp_forPR=rvec_frmSolvePnp,
                                       Translation_FromSolvepnp_forPR=tvec_frmSolvePnp,
                                       CameraMatrix_forPR=cameraMatrix)
    
    #get camera coordinates from projection matrixDIY_ProjectMatrix
    #we know this function works as it is used succesfully in other programs (dense reconstruction)
    C_cameraCoords=GetCameraCoords_FromProjectionMatrix(TestProjectMatrix,False)
    Cam1_Kmat_cv2, Cam1_Rmat_cv2 , Cam1_C_Or_Tmat_cv2 = decomposeP(TestProjectMatrix)
    C_cameraCoordsB=GetCameraCoords_FromProjectionMatrix(TestProjectMatrix2,False)
    Cam1_Kmat_cv2B, Cam1_Rmat_cv2B , Cam1_C_Or_Tmat_cv2B = decomposeP(TestProjectMatrix2)

    #create point like we use for projectpoints 
    paddedTmat_1x3=np.asmatrix(np.array([(0.0, 0.0, 1000.0,1)])).T
    NoseEndPoint_FromPose2d_Manual=(np.matmul(TestProjectMatrix2,paddedTmat_1x3))

    #draw line on image
    ImageWithPose=cv2.line(ImageWithPose,
                           (int(NoseStartPoint_Found2d[0,0]),int(NoseStartPoint_Found2d[0,1])),
                           (int(NoseEndPoint_FromPose2d_Manual[0,0]),int(NoseEndPoint_FromPose2d_Manual[1,0])),
                           (255,255,0),4)

    #draw 3d landmarks points used onto 2D image
    for x in range(0,len(objectPoints_3D)):
        if x>0:
         (previousTemp2d_landmark, jacobian) = cv2.projectPoints(np.array([(objectPoints_3D[x-1][0][0], objectPoints_3D[x-1][1][0], objectPoints_3D[x-1][2][0])]), rvec_frmSolvePnp, tvec_frmSolvePnp, cameraMatrix, distCoeffs)
         (Temp2d_landmark, jacobian) = cv2.projectPoints(np.array([(objectPoints_3D[x][0][0], objectPoints_3D[x][1][0], objectPoints_3D[x][2][0])]), rvec_frmSolvePnp, tvec_frmSolvePnp, cameraMatrix, distCoeffs)
         #draw dot on image (using line tool)
         ImageWithPose=cv2.line(ImageWithPose,
           (int(Temp2d_landmark[0][0][0]),int(Temp2d_landmark[0][0][1])),
           (int(previousTemp2d_landmark[0][0][0]),int(previousTemp2d_landmark[0][0][1])),
                       (255,0,255),6)

    
    #############################################################
   
    #CONVERT results from solvepnp to be compatible with importing software
    #coordinate system
    ConvertedRot_XYZ, ConvertedXYZString_rot=Convert_solvepnp_rotation(FacePoseImportingSoftware,tvec_frmSolvePnp,rvec_frmSolvePnp)
    ConvertedTrans_XYZ, ConvertedXYZString_pos=Convert_solvepnp_translation(FacePoseImportingSoftware,tvec_frmSolvePnp,rvec_frmSolvePnp)
    #print details onto illustrated image
    cv2.putText(ImageWithPose, ConvertedXYZString_rot, (50,50), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.7,color=(255,0,255),thickness=2)
    cv2.putText(ImageWithPose, ConvertedXYZString_pos, (50,150), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.7,color=(255,0,255),thickness=2)
    
    return NamedTuples_Common.FacePoseReturns(1,
                                              FaceLandMark_Returns_Object.OriginalImage,
                                              ImageWithPose,
                                              ConvertedRot_XYZ,
                                              ConvertedTrans_XYZ,
                                              None,
                                              [NoseStartPoint_Found2d[0,0],NoseStartPoint_Found2d[0,1]],
                                              NotesString)


        
def GetProjectionMatrix_FacePoseFormat(ImportingSoftware_forPR,
                                       Rotation_FromSolvepnp_forPR,
                                       Translation_FromSolvepnp_forPR,
                                       CameraMatrix_forPR):
    
    
    
    
    """Try and generate a projection matrix manually from solvepnp results,
    same results as using FindProjectionMatrix_frmStereoCal but optimised for
    results from solvepnp"""
    #experiment getting projection matrix
    #K= camera matrix 3*3
    #R = rotation?? ?x?
    #c=coords camera world
    #p= K [R | t]
    #t=-RC~
    
    #convert translation matrix to 3*3 rodrigues matrix (works both ways)
    rotM=cv2.Rodrigues(Rotation_FromSolvepnp_forPR)[0]#two results from this -[0] is the rotation matrix
    
    EulerRvec_badFormat=rot_params_rv(Rotation_FromSolvepnp_forPR)
    #sub obtimal way of copying format of incoming data and replacing elements 
    EulerRvec=(Rotation_FromSolvepnp_forPR).copy()
    EulerRvec[0:3,0]=EulerRvec_badFormat[:]
    
    #cheap way of copying data 
    rotM=cv2.Rodrigues(Rotation_FromSolvepnp_forPR)[0]#two results from this -[0] is the rotation matrix
    
    #print("is rotation matrix?:" , str(isRotationMatrix(rotM)))

    #print("EulerRvec:", EulerRvec)
    #get camera translation  this function returns (t=-RC~)
    Opencv_ConvertedTrans_XYZ, Opencv_ConvertedXYZString_trans=Convert_solvepnp_translation(ImportingSoftware_forPR,Rotation_FromSolvepnp_forPR,Translation_FromSolvepnp_forPR)
#    #unconvert translation TEST
#    Opencv_ConvertedTrans_XYZ=Translation_FromSolvepnp_forPR[:,0]
    
    #second way to get projection matrix - lets see if any difference!
    DIY_ProjectMatrix2=FindProjectionMatrix_frmStereoCal(CameraMatrix=CameraMatrix_forPR,
                                                                   RotationMatrix=rotM,
                                                                   TranslationVector=np.array(Opencv_ConvertedTrans_XYZ),
                                                                   AlternativeMethod=True)

    
    
    #build translation vector that the FindProjectionMatrix function handles
    Test_TransVector=np.array([0, 0, 0,1])
    Test_TransVector[0:3]=Opencv_ConvertedTrans_XYZ[0:3]
    #create empty 4*4 for RT matrix
    DIY_ProjectMatrix=np.zeros((3,4),np.float32)
    #DIY_ProjectMatrix[3,3]=1#needs to have "1" in this position
    #copy in Rotation Matrix to correct position
    DIY_ProjectMatrix[0:3,0:3]=np.mat(rotM)
    DIY_ProjectMatrix[0:3,3:4]=np.mat(Test_TransVector[0:3]).T#transform vector to cocatenate into 4th column of RT matrix
    #multiply camera matrix by RT  
    DIY_ProjectMatrix=np.matmul(np.asmatrix(CameraMatrix_forPR),np.asmatrix(DIY_ProjectMatrix))
    
    return DIY_ProjectMatrix,DIY_ProjectMatrix2

def FaceDetector_DLIB_CNN(InputImage,UnusedValue, cnn_face_detector):
    """number of faces found,Return input images with face borders, 
    list of face rectangles""" 
    #probably need CUDA library to speed it up??
    #current score = 84% from 62 analysed images
    
    RequiredWidth= 500#for speed only in DLiB - works at all sizes
    
    Scale_Ratio=(RequiredWidth /(InputImage.shape[1]))
    #override ratio
    Scale_Ratio=1
    logging.info("[INFO] DLIB facedetector resizing image " + str(RequiredWidth))
    width = int(InputImage.shape[1] * Scale_Ratio ) 
    height = int(InputImage.shape[0] * Scale_Ratio ) 
    dim = (width, height) 
    # resize image
    InputImage_Internal=cv2.resize(InputImage, dim, interpolation = cv2.INTER_AREA)     
    #InputImage_Internal=InputImage.copy()
    try:
    #run DLIB CNN face detector
        faces_cnn=cnn_face_detector(InputImage_Internal,0)#upsample image X times
    except Exception:
        logging.critical("Error running DLIB CNN face detector",exc_info=True)#full stack trace
        logging.info("Error running DLIB CNN face detector" )
        raise
    
    AcceptedFaces=0
    List_of_FaceBorders=[]
    #Draw Face boundaries
    #DLIB cnn detector    
    for (i, rect) in enumerate(faces_cnn):
        AcceptedFaces=AcceptedFaces+1
        x1 = rect.rect.left()
        y1 = rect.rect.top()
        x2 = rect.rect.right()
        y2 = rect.rect.bottom()
        
        #package into named tuple
        OpenCVBoundary=NamedTuples_Common.BoundingBox(x1, y1, x2, y2)
        List_of_FaceBorders.append(OpenCVBoundary)#[x1,y1,x2,y2])#(startX, startY, endX, endY)
        # Rectangle around the face
        cv2.rectangle(InputImage_Internal, (x1, y1), (x2, y2), (255, 255, 255), 3)
    #package return to match named tuple (havent done this quite right)
    #TODO
    return NamedTuples_Common.FaceDetectorReturns(AcceptedFaces,InputImage, InputImage_Internal,List_of_FaceBorders)
    #return AcceptedFaces,InputImage, InputImage_Internal,List_of_FaceBorders

def FaceLandMarks_TensorFlow(InputImage,FaceDetectorReturnsObject,TensorFlow_FaceLandMark_Predictor,detect_marks_Function):
    """Take in results (named tuple) from Face Detector stage and find facial landmarks"""
    try:
        
        InputImage_withLandmarks=InputImage.copy()  
        List_of_FaceLandmarks=[]
        FaceCount=0
        #For each face boundary, predict face landmarks
        for FaceBoundary in FaceDetectorReturnsObject.ListOfFaceBorderCorners:
            #DlibFaceShape = predictor(FaceDetectorReturnsOpenCV.OriginalImage, dlib.rectangle(*FaceBoundary))
            FaceCount=FaceCount+1
            #runs the face landmark predictor and returns a np array of shape [68,2]

            PredictFaceLandmarks=detect_marks_Function(InputImage_withLandmarks, TensorFlow_FaceLandMark_Predictor, FaceBoundary)
            List_of_FaceLandmarks.append(np.asmatrix(PredictFaceLandmarks))#to keep same format as DLIB landmarker
         #For each face boundary, draw face landmarks
         #input params for landmark annotations
         #expect 1 face here so for illustration change colour of any subsequential faces for ease of analysis
        Red=255
        Green=255
        size=2
        for FaceLandmarks in List_of_FaceLandmarks:
            InputImage_withLandmarks=draw_Keypoints_NP_array(InputImage_withLandmarks,FaceLandmarks,(Red,Green,0),size)
            InputImage_withLandmarks=draw_Centre_of_image(InputImage_withLandmarks)
            Red=int(Red/2)
            Green=int(Green/2)
        return NamedTuples_Common.FaceLandMarkReturns(FaceCount,InputImage,InputImage_withLandmarks,List_of_FaceLandmarks)
     
    except Exception:
        logging.critical("Error extracting Landmarks from Faces",exc_info=True)#full stack trace
        raise

def FaceLandMarks_DLIB(InputImage,FaceDetectorReturnsObject,DLIB_FaceLandMark_Predictor,DLIBrectangleObj):
    """Take in results (named tuple) from Face Detector stage and find facial landmarks"""
    try:
        
        InputImage_withLandmarks=InputImage.copy()  
        List_of_FaceLandmarks=[]
        FaceCount=0
        #For each face boundary, predict face landmarks
        #input params for landmark annotations
         #expect 1 face here so for illustration change colour of any subsequential faces for ease of analysis
        Red=255
        Green=255
        size=2
        for FaceBoundary in FaceDetectorReturnsObject.ListOfFaceBorderCorners:
            #DlibFaceShape = predictor(FaceDetectorReturnsOpenCV.OriginalImage, dlib.rectangle(*FaceBoundary))
            FaceCount=FaceCount+1
            #runs the face landmark predictor and returns a np array of shape [68,2]
            PredictFaceLandmarks=DLIB_FaceLandMark_Predictor(FaceDetectorReturnsObject.OriginalImage, DLIBrectangleObj(*FaceBoundary))
            Face_landmarks = np.matrix([[p.x, p.y] for p in PredictFaceLandmarks.parts()])

            List_of_FaceLandmarks.append(Face_landmarks)
         #For each face boundary, draw face landmarks 
        for FaceLandmarks in List_of_FaceLandmarks:
            #print("FaceLandMarks_DLIB", FaceLandmarks)
            
            InputImage_withLandmarks=draw_Keypoints_NP_array(InputImage_withLandmarks,FaceLandmarks,(Red,Green,0),size)
            InputImage_withLandmarks=draw_Centre_of_image(InputImage_withLandmarks)
            Red=int(Red/2)
            Green=int(Green/2)
        return NamedTuples_Common.FaceLandMarkReturns(FaceCount,InputImage,InputImage_withLandmarks,List_of_FaceLandmarks)
     
    except Exception:
        logging.critical("Error extracting Landmarks from Faces",exc_info=True)#full stack trace
        raise

def FaceDetector_OpenCV_DNN(InputImage,Min_Confidence,OpenCV_DNN_FaceDetector):
    """number of faces found,Return input images with face borders, 
    list of face rectangles"""
    ###Pass Ratio : 77% from 62 analysed images
    InputImage_Internal=InputImage.copy()  
    (h, w) = InputImage_Internal.shape[:2]
    ResizeImage_to=300
    try:
        
        logging.info("[INFO] FaceDetector OpenCV DNN resizing image " + str(ResizeImage_to))
        blob = cv2.dnn.blobFromImage(cv2.resize(InputImage_Internal, (ResizeImage_to, ResizeImage_to)), 1.0,
            	(ResizeImage_to, ResizeImage_to), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections and
        # predictions
        #logging.info("[INFO] computing object detections for  " + Data_FaceDetector_Object.Get_InputImage())
        OpenCV_DNN_FaceDetector.setInput(blob)
        detections = OpenCV_DNN_FaceDetector.forward()
    except Exception:
        logging.critical("FaceDetector DNN blob propogation error",exc_info=True)#full stack trace
        raise
        
    #Draw Face boundaries
    #OpenCV DNN detector  
    AcceptedFaces=0
    List_of_FaceBorders=[]
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > Min_Confidence:
    
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            AcceptedFaces=AcceptedFaces+1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            
            (startX, startY, endX, endY) = box.astype("int")
            
            #package into named tuple
            OpenCVBoundary=NamedTuples_Common.BoundingBox(startX, startY, endX, endY)
            
            
            List_of_FaceBorders.append(OpenCVBoundary)#[startX, startY, endX, endY])
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(InputImage_Internal, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(InputImage_Internal, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    #package return to match named tuple (havent done this quite right)
    #TODO
    return NamedTuples_Common.FaceDetectorReturns(AcceptedFaces,InputImage, InputImage_Internal,List_of_FaceBorders)

def CheckPythonVersion(Version, RaiseException):
    """CheckPythonVersion((3,4),RaiseException=True )
    """
    if not sys.version_info[:2] == Version:
        Errstr=("Error, I need python " + str(Version) + " not :"  + str(sys.version_info[:2]) )
        print (Errstr)
        if RaiseException==True:
            logging.critical(Errstr,exc_info=True)
            raise Exception (Errstr )
            
            
def ExtractNumber_fromstring(InputString):
    """Remove all non-numeric charactors (retain floating point and negative)"""
    
    
    try:
         #TODO this is a bit weak - try and get TRANSLATE working
         #FloatingNumber=InputString.translate(None,Str_Common.chars_NonNumeric_Floating)
         
         if '-' in InputString:
             FloatingNumber=float(re.sub("[^\d\.]", "", InputString))*-1
         else:
             FloatingNumber=(re.sub("[^\d\.]", "", InputString))
             
         return (FloatingNumber)
     
    except Exception as e:
        logging.critical("RemoveNonNums_fromstring" + str(e) ,exc_info=True)
        raise Exception ("error removing non-numeric chars from string " + str(InputString) + ": " + str(e))
               
def rot_params_rv(rvecs):
    """convert from SolvePnp rvecs (axis angle representation of rotation
    but with THETA encoded into the 3 numbers), into euler angles"""
    from math import pi,atan2,asin
    R = cv2.Rodrigues(rvecs)[0]
    roll=180*atan2(-R[2][1],R[2][2])/pi
    pitch=180*asin(R[2][0])/pi
    yaw=180*atan2(-R[1][0],R[0][0])/pi
    #experiment by converting back to radians
    rot_params=[roll/Numbers_Common.DegreePerRadian,
                pitch/Numbers_Common.DegreePerRadian,
                yaw/Numbers_Common.DegreePerRadian]
    return rot_params

class Data_FaceDetect_WorkingDetails:
        """Hold working details such as filepaths, face detector parameters and working objects,will raise execeptions on invalid data"""
        ClassName="REMOVED CODE : inspect.stack()[0][3]"#get classname off stack. Might be nicer to get instance name?
        
        
        def __init__(self):
            #self.ClassName=""
            self.Confidence=0.5#cut-off threshold for face acceptance
            self.Image_FilePath=None#location of image to be assessed
            self.Image_Working=None
            self.ProjectPath=None
            self.FaceLandMarks3Dpts_68="/Resources/FaceAnalysisResources/FaceLandmarks_3Dxyz_68_points.pp"#Meshlab PickPoints tool, 68 exported landmarks manually marked on a mesh and exported, correct corresponding index
            self.prototxt_FileNAME="/Resources/FaceAnalysisResources/deploy.prototxt.txt"#Caffe DNN weights
            self.OpencvDNN_Model_Params="/Resources/FaceAnalysisResources/res10_300x300_ssd_iter_140000.caffemodel"#SingleShot detector/Resnet DNN Model/Layers
            self.DlibCnn_weightsFile="/Resources/FaceAnalysisResources/mmod_human_face_detector.dat"
            self.DLIB_FaceLandMarks_PredictorFILE="/Resources/FaceAnalysisResources/shape_predictor_68_face_landmarks.dat"
            self.UserConfigFile="/Resources/UserConfig/Retopology_Config.txt"
            self.InputImages_Path=""
            self.OutputPassFolder="/FaceAnalysisOutput_Pass"
            self.OutputFailFolder="/FaceAnalysisOutput_Fails"
            self.OutputDEBUGFolder="/FaceAnalysisOutput_Debug"
            self.OutputFolderIndividual="/FaceAnalysisOutput"
            self.OutputFolderForRetop="/FaceAnalysisOutput_ForRetop"
            self.Face3DPose_InputSoftware=Str_Common.EMPTY#to convert to coordinate system
            self.ListofImages_InInputFolder=[]
            self.useUserDefined3d_Face_Landmarks=False#false=use hardcoded 3d faceland, true=try to load meshlab 3d points chosen by user
            self.OpenCV_DNN_FaceDetector=None#Open CV DNN Face detector
            self.DLIB_cnn_FaceDetector = None#dlib CNN face detector
            self.DLIB_FaceLandmark_Predictor=None#dlib 68pt landmark face predictor
        
        def Extract_FaceLandMarksxyz_PickPoints(self):
            """Extract XYZ points from several methods (PP File or hardcoded )
            
            need 68 face landmarks in 3d space to solve pose VS 2d landmarks found,
            at moment xyz points are generated from PP file used by manually plotting points
            in meshlab PICK POINTS tool using generic head model"""
            
            
            if self.useUserDefined3d_Face_Landmarks==False:
            
                logging.info("Getting static 3D landmarks, using hardcoded values - might have left and right mixed up!!") 
                   
                #HARDCODED test
                # index of landmark returned from dlib 68 points test and XYZ point
                _3D_points_Ref_dict_HardCode={}
                _3D_points_Ref_dict_HardCode[55]=[(-150.0, -150.0, -125.0), 'Left corner of the mouth']
                _3D_points_Ref_dict_HardCode[49]=[(150.0, -150.0, -125.0),' Right corner of the mouth']
                _3D_points_Ref_dict_HardCode[31]=[( 0.0, 0.0, 0.0),'tip of nose']
                _3D_points_Ref_dict_HardCode[9]=[(0.0, -330.0, -65.0),'Chin']
                _3D_points_Ref_dict_HardCode[46]=[(-225.0, 170.0, -135.0),'Left corner of the left eye'] 
                _3D_points_Ref_dict_HardCode[37]=[( 225.0, 170.0, -135.0),'Right corner of the right eye']
               
                 
                return _3D_points_Ref_dict_HardCode
                
            
            if self.useUserDefined3d_Face_Landmarks==True:
            
                
                logging.warning("Getting user-defined 3D face landmarks from meshlab PP file. if not aligned will cause poor results ") 
                
                
                #test 3d face landmark positions (68 of) needed to solvepnp with 2d landmarks found by dlib
                #self.test_file(self.Get_ProjectPath() + self.FaceLandMarks3Dpts_68,inspect.stack()[0][3],self.Get_ProjectPath() + self.FaceLandMarks3Dpts_68 + " does not exist - failure to find 3d positions of DLIB facial landmarks")
                #load file 
                with open(self.Get_ProjectPath() + self.FaceLandMarks3Dpts_68, 'r') as myfile:
                    data = myfile.read()
                
                #break up text file so each element is:
                #point y="-0.38664" z="0.273718" x="-0.350152" active="1" name="3"/>
                Delimited_Data_MixedCase=data.split('<')
                #set to lower case
                Delimited_Data = [item.lower() for item in Delimited_Data_MixedCase]
                
                List_of_delimitedElements=[]
                
                #for each element have a sub-list as follows:
                #1            point
                #2            y="-0.128161"
                #3            z="0.553672"
                #4            x="-0.11163"
                #5            active="1"
                #6            name="39"/>
                #warning: not always in same order!
                
                for XYZdata in Delimited_Data:
                    BrokenOutXYZdata=XYZdata.split(" ")
                    List_of_delimitedElements.append(BrokenOutXYZdata)
                
                
                #List_of_delimitedPoints=[]
                _3D_points_Ref_dict={}
                for XYZdataNode in List_of_delimitedElements:
                    #does element have "point" node? Might be an XYZ coord
                    if XYZdataNode.count(Str_Common.MeshLabPickPts_point)>0 and XYZdataNode.count(Str_Common.MeshLabPickPts_active1)>0:#is point selected to be "active"?
                        Xyzpoint_x=None
                        Xyzpoint_y=None
                        Xyzpoint_z=None
                        FaceLandMarkIndex=None
                        for XyzPoint in XYZdataNode:
                            if len(XyzPoint)>0:
                                if XyzPoint[0]==Str_Common.MeshLabPickPts_x:
                                    Xyzpoint_x=ExtractNumber_fromstring(XyzPoint)
                                elif XyzPoint[0]==Str_Common.MeshLabPickPts_y:
                                     Xyzpoint_y=ExtractNumber_fromstring(XyzPoint)
                                elif XyzPoint[0]==Str_Common.MeshLabPickPts_z:
                                    Xyzpoint_z=ExtractNumber_fromstring(XyzPoint)
                                elif Str_Common.MeshLabPickPts_Name in XyzPoint:
                                    FaceLandMarkIndex=ExtractNumber_fromstring(XyzPoint)
                        #check we have x,y,z and landmark index ("name") all loaded
                        if Xyzpoint_x is not None and Xyzpoint_y is not None and Xyzpoint_z is not None and FaceLandMarkIndex is not None :
                            #load all points into containers
                            #container for user/programmer reference
                            _3D_points_Ref_dict[int(FaceLandMarkIndex)]=[(Xyzpoint_x, Xyzpoint_y, Xyzpoint_z),'Automatic LandMark Indexing']
                         
                
                
                return _3D_points_Ref_dict
        
        
        def Set_List_Of_Images(self,List_of_Images_InInputFolder):
            if List_of_Images_InInputFolder==[]:
                logging.info("List of input images empty",exc_info=True)
                raise Exception ("List of input images empty")
            self.ListofImages_InInputFolder=List_of_Images_InInputFolder
            
        def Get_GetFaceLandMarkFile(self):
            temp=self.ProjectPath + self.DLIB_FaceLandMarks_PredictorFILE
            #self.test_String(temp,inspect.stack()[0][3],"Invalid filepath to face landmarks ")
            #self.test_file(temp,inspect.stack()[0][3],temp + " does not exist")
            return temp   
        
        def test_file(self, InputFile, FunctionName,OutputError):
            """Checks a file is valid - if not asserts an error"""
            assert(os.path.isfile(InputFile) ),"Class: " + self.ClassName + " fnct: " + FunctionName + " Err: " + OutputError
        
        def test_String(self,InputString, FunctionName, OutputError):
            """Checks a string is valid - if not asserts an error"""
            assert("".__eq__(InputString)==False),"Class: " + self.ClassName + " fnct: " + FunctionName + " Err: " + OutputError
        
        def Set_Confidence(self,InputFloat):
            #assert(InputFloat is not None),"Class: " + self.ClassName + " fnct: " + inspect.stack()[0][3] + " Err: " + "Invalid Confidence"
            try:
                float(InputFloat)
            except:
                raise ValueError("Class: " + self.ClassName + " fnct: " + inspect.stack()[0][3] + " Err: " + "Confidence NaN")
            assert(InputFloat not in range (0,1)),"Class: " + self.ClassName + " fnct: " + inspect.stack()[0][3] + " Err: " + "Confidence outwith range 0-1"
            self.Confidence=InputFloat
            
        def Get_Confidence(self):
            #assert(self.Confidence is not None),"Class: " + self.ClassName + " fnct: " + inspect.stack()[0][3] + " Err: " + "Invalid Confidence"
            try:
                float(self.Confidence)
            except:
                raise ValueError("Class: " + self.ClassName + " fnct: " + inspect.stack()[0][3] + " Err: " + "Confidence NaN")
            #assert(self.Confidence not in range (0,1)),"Class: " + self.ClassName + " fnct: " + inspect.stack()[0][3] + " Err: " + "Confidence outwith range 0-1"
            return self.Confidence
        
        def Set_InputImage(self,InputImage_FullPath):
            #self.test_String(InputImage_FullPath,inspect.stack()[0][3],"str err Invalid InputImage_FullPath")
            #self.test_file(InputImage_FullPath,inspect.stack()[0][3],InputImage_FullPath + " does not exist")
            self.Image_FilePath=InputImage_FullPath     
            try:  
                # load the input image
                self.Image_Working = cv2.imread(self.Get_InputImage())
            except Exception:
                logging.critical("Error loading image to be analysed",exc_info=True)#full stack trace
                logging.info("Image location: " +self.Get_InputImage() )
                raise            
            
        def Get_InputImage(self):
            #self.test_String(self.Image_FilePath,inspect.stack()[0][3],"Invalid Image_FilePath")
            return self.Image_FilePath
        
        def Correct_filename_leadSlash(self,InputString):
            """corrects slash formatting at start of filename (without path)"""
            StringList=list(InputString)
            if StringList[0]=="\\":
                StringList[0]="/"
            if StringList[0]!="/":
                StringList.insert(0,"/")
            return "".join(StringList)
            
        def Set_prototxt_FileNAME(self, InputPrototxt_FileName):
            #self.test_String(self.ProjectPath,inspect.stack()[0][3],"Invalid DNN ProjectPath")
            #self.test_String(InputPrototxt_FileName,inspect.stack()[0][3],"Invalid prototxt_FileNAME")
            TempString=self.Correct_filename_leadSlash(InputPrototxt_FileName)
            #self.test_file(self.ProjectPath+TempString,inspect.stack()[0][3],self.ProjectPath+InputPrototxt_FileName + " does not exist")
            self.prototxt_FileNAME=self.Correct_filename_leadSlash(TempString)
            
        def Get_prototxt_FileNAMEandPATH(self):
            #self.test_String(self.ProjectPath,inspect.stack()[0][3],"Invalid DNN ProjectPath")
            #self.test_String(self.prototxt_FileNAME,inspect.stack()[0][3],"Invalid prototxt_FileNAME")
            return self.ProjectPath+ self.prototxt_FileNAME
      
        def Set_ProjectPath(self,Folder_DNNWorkingDirectory):
            #self.test_String(Folder_DNNWorkingDirectory,inspect.stack()[0][3],"Invalid Folder_DNNWorkingDirectory")
            #assert(os.path.exists(Folder_DNNWorkingDirectory)),"Class: " + self.ClassName + " fnct: " + inspect.stack()[0][3] + " Err: " + Folder_DNNWorkingDirectory + " path does not exist"
            self.ProjectPath=Folder_DNNWorkingDirectory
            
        def Get_ProjectPath(self):
            #self.test_String(self.ProjectPath,inspect.stack()[0][3],"Invalid Folder_DNNWorkingDirectory")
            return self.ProjectPath
        def Get_ModelParams(self):
            #self.test_String(self.OpencvDNN_Model_Params,inspect.stack()[0][3],"Invalid working params")
            return self.Get_ProjectPath() + "/" + self.OpencvDNN_Model_Params
        def Set_InputImagesPath(self,InputImagesPath):
            #self.test_String(InputImagesPath,inspect.stack()[0][3],"Invalid InputImagesPath")
            #assert(os.path.exists(InputImagesPath)),"Class: " + self.ClassName + " fnct: " + inspect.stack()[0][3] + " Err: " + InputImagesPath + " path does not exist"
            self.InputImages_Path=InputImagesPath  
        def Get_InputImagesPath(self):
            #self.test_String(self.InputImages_Path,inspect.stack()[0][3],"Invalid InputImages_Path")
            return self.InputImages_Path
        
        def Initialise_OpenCV_DNN_FaceDetector(self):
            """load network model using pre-assigned filepath details"""
            try:
                print("[INFO] loading Opencv DNN model...")
                logging.info("loading Opencv DNN model...")
                self.OpenCV_DNN_FaceDetector = cv2.dnn.readNetFromCaffe(self.Get_prototxt_FileNAMEandPATH(),self.Get_ModelParams())
                
            except Exception:
                logging.critical("Error loading DNN network",exc_info=True)#full stack trace
                logging.info("dnn location: " +self.Get_prototxt_FileNAMEandPATH() )
                logging.info("dnn location: " +self.Get_ModelParams() )
                raise
        
        
        def Initialise_Dlib_CNN_FaceDetector(self,DLIB_FaceDetectorModelV1):
            """load network model using pre-assigned filepath details"""
            try:
                print("[INFO] loading DLIB CNN detector and weights...")
                logging.info("loading DLIB CNN detector and weights...")
                self.DLIB_cnn_FaceDetector = DLIB_FaceDetectorModelV1(self.ProjectPath  + self.DlibCnn_weightsFile)
                
            except Exception:
                logging.critical("Error loading CNN network",exc_info=True)#full stack trace
                logging.info("CNN location: " + str(self.ProjectPath + "/" + self.DlibCnn_weightsFile))
                raise
                
        def Initialise_Dlib_FaceLandMarks(self,DLIB_FaceShapePredictor):
            """load DLIB DLIB_FaceShapePredictor pretrained model using pre-assigned filepath details"""
            try:
                print("[INFO] loading DLIB_FaceShapePredictor")
                logging.info("loading DLIB_FaceShapePredictor")
                self.DLIB_FaceLandmark_Predictor = DLIB_FaceShapePredictor(self.ProjectPath  + self.DLIB_FaceLandMarks_PredictorFILE)
                
            except Exception:
                logging.critical("Error loading DLIB Face Predictor trained model",exc_info=True)#full stack trace
                logging.info("DLIB Face Predictor location: " + str(self.ProjectPath + "/" + self.DLIB_FaceLandMarks_PredictorFILE))
                raise
        
        def SaveImage_Debug(self,ImageToSave,InputImage_FileName):
            """debug save to arbitrary location"""
            try:
                
                cv2.imwrite(self.Get_ProjectPath() + "/" + self.OutputDEBUGFolder + "/" + InputImage_FileName,ImageToSave)
                return True
                
            except Exception:
                logging.critical("Error saving pass/fail face detect output",exc_info=True)#full stack trace
                raise
        def SaveImage(self,FilePath,InputImage):
            """Save image to arbitary folder"""
            cv2.imwrite(FilePath,InputImage)
        def SaveImageFail(self,FaceDetectorReturns,InputImage_FileName, InputImage):
            """Save image to failure folder"""
            cv2.imwrite(self.Get_ProjectPath() + "/" + self.OutputFailFolder + "/" + InputImage_FileName,InputImage)
        def SaveImagePass(self,FaceDetectorReturns,InputImage_FileName):
            """Save image to Pass folder"""
            #cv2.imwrite(self.Get_ProjectPath() + "/" + self.OutputPassFolder + "/" + InputImage_FileName,FaceDetectorReturns.)

        def SaveImagePassOrFail(self,FaceDetectorReturns,InputImage_FileName):
            """Saves to pass or fail folder depending on match 
            between manually counted faces embedded in image using bookend system
            and automatically detected faces.
            Returns pass or fail boolean"""
            #get face count the user has embedded into image title (using BOOKEND system)
            try:
                
                if FaceDetectorReturns.FaceCount==self.FindManualFaceCount_InImage(InputImage_FileName,Str_Common.NoOfFaces):
                    cv2.imwrite(self.Get_ProjectPath() + "/" + self.OutputPassFolder + "/" + InputImage_FileName,FaceDetectorReturns.ImageWithFaceBorders)
                    return True
                else:
                    cv2.imwrite(self.Get_ProjectPath() + "/" + self.OutputFailFolder + "/" + InputImage_FileName,FaceDetectorReturns.ImageWithFaceBorders)
                    return False
            except Exception:
                logging.critical("Error saving pass/fail face detect output",exc_info=True)#full stack trace
                raise
                
        def CalculateEfficiency(self):
            """Counts images in pass and fail folders and calculate an efficiency
            requires that input images have manual face count embedded in name using bookend system
            and pass fail folders set up correctly"""
            NoPassImages=len(GetList_Of_ImagesInfolder(self.Get_ProjectPath() + "/" + self.OutputPassFolder))
            NoTotalImages=NoPassImages+len(GetList_Of_ImagesInfolder(self.Get_ProjectPath() + "/" + self.OutputFailFolder))
            if NoTotalImages<1:
                return str(" Pass Ratio : 0% from " + str(NoTotalImages) + " analysed images")
         
            
            return str(" Pass Ratio : " + str(round(NoPassImages*100/NoTotalImages)) + "% from " + str(NoTotalImages) + " analysed images")
        def FindManualFaceCount_InImage(self,ImageFileName,BookEnd):
            """Searches for two instances of a string in the filename where the
            manually counted number of faces is sandwiched - EG C:/ImageFileNoOfface5NoOffaces.jpg"""
            NumberFound=ImageFileName.count(BookEnd)
            if (NumberFound!=2):
                print("WARNING: FindManualFaceCount_InImage No bookends found, so returning 1")
                return 1
                raise Exception ("Metric Mode Err: Could not find 2 facecount BookEnd " + BookEnd + " in " + ImageFileName)
            TempStringStage1=ImageFileName[(ImageFileName.index(BookEnd))+len(BookEnd):ImageFileName.rindex(BookEnd)]
            if not TempStringStage1.isdigit():
               raise Exception ("Metric Mode Err: Error trying to find manually counted faces in file, is there a number sandwiched between strings " + BookEnd + " in file " + ImageFileName)
            NumberFound=int(TempStringStage1)
            return NumberFound
        def InitialiseEverything_Debug(self,DLIB_FaceDetectorModelV1,DLIB_FacePredictor):
            
            #root logger always dictates to children
            #restart interpreter for operation
            log_format = '%(asctime)s %(filename)s: %(message)s'#enable timedate logging
            logging.basicConfig(filename='FacePoseTestLogger.log',format=log_format, level=logging.DEBUG)#will just drop in same folder as .py file?
            #load our serialized OpenCV DNN model from disk
            self.Initialise_OpenCV_DNN_FaceDetector()
            #load our serialized DLIB CNN model from disk
            self.Initialise_Dlib_CNN_FaceDetector(DLIB_FaceDetectorModelV1)
            #delete and recreate output folders
            #DeleteFiles_RecreateFolder(self.Get_ProjectPath() + self.OutputPassFolder)
            #DeleteFiles_RecreateFolder(self.Get_ProjectPath() + self.OutputFailFolder)
            DeleteFiles_RecreateFolder(self.Get_ProjectPath() + self.OutputDEBUGFolder)
            DeleteFiles_RecreateFolder(self.Get_ProjectPath() + self.OutputFolderIndividual)
            DeleteFiles_RecreateFolder(self.Get_ProjectPath() + self.OutputFolderForRetop)
            #find all images in the Input Folder
            self.Set_List_Of_Images(GetList_Of_ImagesInfolder(self.Get_InputImagesPath()))
            self.Initialise_Dlib_FaceLandMarks(DLIB_FacePredictor)
            
        def InitialistOpenCV_Debug(self):
            
            #root logger always dictates to children
            #restart interpreter for operation
            log_format = '%(asctime)s %(filename)s: %(message)s'#enable timedate logging
            logging.basicConfig(filename='FacePoseTestLogger.log',format=log_format, level=logging.DEBUG)#will just drop in same folder as .py file?
            #load our serialized OpenCV DNN model from disk
            self.Initialise_OpenCV_DNN_FaceDetector()
            #load our serialized DLIB CNN model from disk
            #self.Initialise_Dlib_CNN_FaceDetector(DLIB_FaceDetectorModelV1)
            #delete and recreate output folders
            DeleteFiles_RecreateFolder(self.Get_ProjectPath() + self.OutputPassFolder)
            DeleteFiles_RecreateFolder(self.Get_ProjectPath() + self.OutputFailFolder)
            DeleteFiles_RecreateFolder(self.Get_ProjectPath() + self.OutputDEBUGFolder)
            #find all images in the Input Folder
            self.Set_List_Of_Images(GetList_Of_ImagesInfolder(self.Get_InputImagesPath()))
            #self.Initialise_Dlib_FaceLandMarks(DLIB_FacePredictor)
            
            
        def shape_to_np(shape, dtype="int"):
            """Convert DLIB face landmark results to numPy array"""
            # initialize the list of (x, y)-coordinates
            coords = np.zeros((68, 2), dtype="int")
        	# loop over the 68 facial landmarks and convert them
        	# to a 2-tuple of (x, y)-coordinates
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
            # return the list of (x, y)-coordinates
            return coords 
        
        
def shape_to_np(shape, dtype="int"):
    """Convert DLIB face landmark results to numPy array"""
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype="int")
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords 

def ResizeImage_PreserveAspectRatio(InputImage,percent_of_original):
    ###resize image preserving aspect ratio
    scale_percent = percent_of_original # percent of original size
    width = int(InputImage.shape[1] * scale_percent / 100)
    height = int(InputImage.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(InputImage, dim, interpolation = cv2.INTER_AREA)
    return resized



class FaceAligner:
    ###from https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    def __init__(self, predictor, desiredLeftEye=(0.90, 0.90),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = Dict_Common.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = Dict_Common.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        
        
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
        # return the aligned face
        return output

def JSON_Save(InputFilePath, InputDictionary_Structure):
    #Serialise input dictionary to JSON human-readable file
    #dictionary can be hierarchical, but only basic data types
    try:
        with open(InputFilePath, 'w') as jsonFile:
            json.dump(InputDictionary_Structure, jsonFile)
    except Exception as e:
            print(e)
def JSON_Open(InputFilePath):
    ##open a JSON file
    #TODO this is pretty sketchy
    try:
        if os.path.exists(InputFilePath):
            # Opening JSON file
            fObj = open(InputFilePath,)

            # It returns JSON object as dictionary
            ogdata = json.load(fObj)
            # Closing file
            fObj.close()
            return ogdata

    except Exception as e:
        print("JSON_Open error attempting to open json file " + str(InputFilePath) + " " + str(e))
        raise Exception (e)
    return None
    
def GetTimeOfFile(InputFilePath):
    #gets time of file last modification in seconds since the epoch
    if os.path.exists(InputFilePath):
        modified_time = os.path.getmtime(InputFilePath)
        return (modified_time)
    return None
def GetCurrentTime():
    #gets time in seconds since the epoch
    return time.time()

def FindUserConfigFile_Parameter(InputParameter, InputFile,ParameterSuffix):
        #read in config file if it exists
        #InputParameter is string to find switch
        #Input file is location of user config file
        #ParameterSuffix to delineate working parameter 
        print("Searching for user configuration file ", InputFile)
        print("looking for switch within text (this must match class):",InputParameter, "searching for: ", ParameterSuffix, InputParameter )
        #check file exists - more pythonic to use a "try" here as file may disappear straight after presence check
        if os.path.isfile(InputFile)==True:
            #parse file to find user configuration options
            with open(InputFile) as f:
                lines = f.readlines()
            #run through each item of delimited textfile to find user parameters
            for CurrentLine in lines:
                #remove spaces
                NoSpacesCurrentLine=CurrentLine.replace(" ","")
                if ParameterSuffix in NoSpacesCurrentLine:
                #user parameters prefixed with this substring, for example "@"
                    if ParameterSuffix+ InputParameter.lower() in NoSpacesCurrentLine.lower():
                        #detected a parameter switch with prefix
                        PositionOfEquals=NoSpacesCurrentLine.find("=")
                        #if no "=" symbol - config file is malformed. Alert user in case the user expects behaviour and misses
                        #warnings
                        if PositionOfEquals==-1:
                            print("ERROR!! Malformed config file!! Cannot continue - please delete or repair")
                            print(InputFile)
                            input("Press Enter to continue...")
                            print(InputFile)
                            raise Exception ("Malformed config file - process cannot continue. Delete config file for default options")
                        #return substring for parameter value (after "=")
                        return True, (NoSpacesCurrentLine [PositionOfEquals+1:])

        #default as error
        print("User config file not found or invalid - default settings will be used - please restore config file from backup")
        print(InputFile)
        input("Press Enter to continue...")
        return False,"ERROR"


def GetUser_Parameter_For_Feature(InputConfigFile, ParameterStringENUMClass, ParamPrefix):
    ###Finds configuration text file, finds user parameter and returns it as ENUM of class
    ###InputConfigFile - full path of fil
    ###ParameterStringENUMClass - class of enums, with a TO STRING method: example:
    #     class FaceDetector_type(enum.Enum):
    # DLIB="DLIB"
    # OpenCV="OpenCV"
    # TensorFlow="TensorFlow"

    # @staticmethod
    # def from_str(Pass_In_Same_Class,label):
    #     #easiest way to pass back enum when we arent instancing the class 
    #     #check if any substrings are in test string
    #     if any(x in label.lower() for x in ["tensor","flow","tensorflow","tf"]):
    #         return True, Pass_In_Same_Class.TensorFlow
    #     if any(x in label.lower() for x in ["opencv","ocv"]):
    #         return True, Pass_In_Same_Class.OpenCV
    #     if any(x in label.lower() for x in ["dlib"]):
    #         return True, Pass_In_Same_Class.DLIB
    #     return False, "FAILURE TO PARSE"

    #NameOfParameter - can be string  name of parameter or send in name of ENUM class 3DVisLabLib.Str_Common.FaceDetector_type.__name__
    #ParamPrefix - prefix with determines if a line of text is a user parameter

    #Get user option for parameter
    #it will return the user option for this parameter- which may have been spelled incorrectly due to user error

    FindParam_Result, FindParam_String=(FindUserConfigFile_Parameter(str(ParameterStringENUMClass.__name__), InputConfigFile,ParamPrefix))
    #do not continue if process has failed
    if FindParam_Result==False:
        return False,"ERROR"

    #now feed the user option to find out what ENUM is used
    #need special method inside class to return the matching enum to input string
    ParameterParseSuccess, TypeToUse=ParameterStringENUMClass.from_str(FindParam_String)
    if ParameterParseSuccess==False:
        print("User config file not parsed correctly - default settings used for face detection or face landmarking - please restore")
        print(InputConfigFile)
        print("Cannot parse user text switch:", FindParam_String)
        input("Press Enter to continue...")
        return False,"ERROR"
    return True, TypeToUse



def Get_PCA_drawAxis(img, p_, q_, colour, scale):
    #https://docs.opencv.org/3.4/d1/dee/tutorial_introduction_to_pca.html
    p = list(p_)
    q = list(q_)
    
    angle = math.atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse =  math.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse *  math.cos(angle)
    q[1] = p[1] - scale * hypotenuse *  math.sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 *  math.cos(angle +  math.pi / 4)
    p[1] = q[1] + 9 *  math.sin(angle +  math.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 *  math.cos(angle -  math.pi / 4)
    p[1] = q[1] + 9 *  math.sin(angle -  math.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
def Get_PCA_getOrientation(pts, img):
    #https://docs.opencv.org/3.4/d1/dee/tutorial_introduction_to_pca.html
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    Get_PCA_drawAxis.drawAxis(img, cntr, p1, (0, 255, 0), 1)
    Get_PCA_drawAxis.drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle =  math.atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return angle
    
def pHash(cv_image):
    
        #hash the image
    #https://stackoverflow.com/questions/55685803/using-opencvs-image-hashing-module-from-python
    imgg = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    h=cv2.img_hash.pHash(imgg) # 8-byte hash
    pH=int.from_bytes(h.tobytes(), byteorder='big', signed=False)
    return pH

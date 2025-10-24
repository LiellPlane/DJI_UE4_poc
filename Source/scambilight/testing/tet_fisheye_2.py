#http://popscan.blogspot.com/2012/04/fisheye-lens-equation-simple-fisheye.html

import sys

import cv2
import numpy as np
import time
import math
import numpy as np
import fisheye_lib

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



def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def straight_copy_of_blog():
    img = cv2.imread(r"D:\hi8-tape-camera-with-fisheye-lens--samsung-vpw75--58859723.png")
    fish_eye = np.empty_like(img)
    ImageViewer_Quick_no_resize(img,0,False,False)
    width = img.shape[0]
    height = img.shape[1]
    for y in range(0, height):
        ny = ((2*y)/height)-1
        ny2 = ny*ny
        for x in range(0, width):
            nx = ((2*x)/width)-1     
            nx2 = nx*nx

            r = math.sqrt(nx2+ny2)

            #  discard pixels outside from circle!
            if (0.0 <= r and  r <= 1.0):
                nr = math.sqrt(1.0-r*r)
                nr = (r + (1.0-nr)) / 2.0

                # discard radius greater than 1.0
                if (nr <= 1.0):
                    theta = math.atan2(ny,nx)
                    nxn = nr*math.cos(theta)
                    nyn = nr*math.sin(theta)
                    x2 = int((((nxn+1)*width)/2.0))
                    y2 = int(((nyn+1)*height)/2.0)
                    if x2 < width and y2 < height:
                        fish_eye[x2, y2, :] = img[x,y,:]

    ImageViewer_Quick_no_resize(fish_eye,0,True,False)


def convert_video():
    input_vid_loc = [
            r"C:\Working\nonwork\SCAMBILIGHT\test_raspberrypi_v2.mp4",
            r"/home/scambilight/test_raspberrypi_v2.mp4"]
    cap = cv2.VideoCapture(input_vid_loc[0])
    outfolder = r"C:\Working\nonwork\SCAMBILIGHT\fish_eye"
    plopset = 0
    while True:
        plopset += 1
        suc, frame = cap.read()
        max_dim = max(list(frame.shape))
        fish_eye_frame = convert_to_fisheye(frame, 2000)
        ImageViewer_Quick_no_resize(fish_eye_frame,0,False,False)
        cv2.imwrite(f"{outfolder}\\0{plopset}.jpg", fish_eye_frame)


def test_more():
    img2 = cv2.imread(r"C:\Working\nonwork\SCAMBILIGHT\avater_calibrate_fake_fisheye.png")
    
    imgs = []
    for i in range(1500,10000,200):
        fisheriser = fisheye_lib.fisheye_tool(img_shape=img2.shape,image_circle_size=i)
        fart = fisheriser.fish_eye_image(img2.copy(),reverse=False)
        #fart = convert_to_fisheye(img.copy(), i, reverse=True)
        #fart = convert_to_fisheye(fart.copy(), i, reverse=True)
        imgs.append((fart, i))
    #fart = convert_to_fisheye(img, 1000)
    ImageViewer_Quick_no_resize(img2,0,True,False)
    for img, i in imgs:
        print(i)
        ImageViewer_Quick_no_resize(img,0,True,False)


def test_more2():
    img2 = cv2.imread(r"C:\Working\nonwork\SCAMBILIGHT\reaL_fish_eye_pic.png")
    #img2 = cv2.imread(r"D:\hi8-tape-camera-with-fisheye-lens--samsung-vpw75--58859723.png")
    fisheriser = fisheye_lib.fisheye_tool(
        img_shape=img2.shape,
        image_circle_size=max(
        img2.shape[1], img2.shape[0])-150)
    fart = fisheriser.fish_eye_image(img2.copy(),reverse=True)

    print(f"incoming{img2.shape}")
    ImageViewer_Quick_no_resize("unfished",img2,0,False,False)
    print(f"incoming{fart.shape}")
    ImageViewer_Quick_no_resize("fishy", fart,0,True,False)


def test_most4():
    img2 = cv2.imread(r"C:\Working\nonwork\SCAMBILIGHT\reaL_fish_eye_pic.png")
    #img2 = cv2.imread(r"D:\hi8-tape-camera-with-fisheye-lens--samsung-vpw75--58859723.png")
    ImageViewer_Quick_no_resize("originalimg", img2, 0,False,False)
    circle = max(
        img2.shape[1], img2.shape[0])
    circle = 1146
    fisheriser = fisheye_lib.fisheye_tool(
        img_width_height=tuple(reversed(img2.shape[0:2])),
        image_circle_size=circle)
    fart = np.zeros_like(img2)
    sample_area_left = 200
    sample_area_right = fart.shape[0] - 200
    sample_area_top = 200
    sample_area_lower = fart.shape[1] - 200
    contours = []
    # along top left to right
    lin = np.linspace(sample_area_left, sample_area_right, 50)
    for x in lin:
        contours.append((x, sample_area_top))
    # down right side
    lin = np.linspace(sample_area_top, sample_area_lower, 50)
    for y in lin:
        contours.append((sample_area_right, y))
    # along bottom right to left
    lin = np.linspace(sample_area_right, sample_area_left, 50)
    for x in lin:
        contours.append((x, sample_area_lower))
    # up left side
    lin = np.linspace(sample_area_lower, sample_area_top, 50)
    for y in lin:
        contours.append((sample_area_left, y))
    cont_ints = [(int(i[0]), int(i[1])) for i in contours]
    sample_area_lerp_contour = cont_ints
    for plop in sample_area_lerp_contour:
        fart[plop[0], plop[1], :] = (0,0,255)
    

    jobby = []
    for plop in sample_area_lerp_contour:
        res = fisheriser.pt_to_reverse_fisheye_inefficient(plop[0], plop[1])
        if res is not None:
            jobby.append(res)
    for plop in jobby:
       fart[plop[0], plop[1], :] = (255,0,0)

    for pt in jobby:
        res = fisheriser.brute_force_find_fisheye_pt(pt)
    
        if res is not None:
            fart[res[0], res[1], :] = (0,255,0)

    decircled = fisheriser.fish_eye_image(img2,reverse=True)
    ImageViewer_Quick_no_resize("decircled", decircled,0,False,False)
    refished = fisheriser.bruteforce_fish_eye_img(decircled)
    ImageViewer_Quick_no_resize("refished", refished,0,False,False)
    # if res is not None:
    #     convex_hull = fisheye_lib.convert_pts_to_convex_hull(res)

    #     cv2.drawContours(
    #         image=fart,
    #         contours=[convex_hull],
    #         contourIdx=-1,
    #         color=(100,200,250),
    #         thickness=5,
    #         lineType=cv2.LINE_AA)

    ImageViewer_Quick_no_resize("original", fart,0,True,False)


def test_more6():
    img_folder = r"C:\VMs\SharedFolder\temp_get_imgs\00.jpg"
    fisheriser = fisheye_lib.fisheye_tool(
        img_width_height=(1269, 972),
        image_circle_size=1250) 
    fart =fisheriser.fish_eye_image(cv2.imread(img_folder), reverse=True)
    ImageViewer_Quick_no_resize("fart", fart,0,True,False)

def test_more5():
    import os
    img_folder = r"C:\Working\nonwork\SCAMBILIGHT\real_fisheye_640_480.png"
    ALL_IMAGES = []
    for root, _, files in os.walk(img_folder):
        for file in files:
            ALL_IMAGES.append(os.path.join(root,file))
    
    fisheriser = fisheye_lib.fisheye_tool(
        img_width_height=(1269, 972),
        image_circle_size=1296 - 150)
    for im in ALL_IMAGES:
        defished = fisheriser.fish_eye_image(cv2.imread(im), reverse=True)
        ImageViewer_Quick_no_resize("defished", defished,0,False,False)


def test_more3():
    img2 = cv2.imread(r"C:\Working\nonwork\SCAMBILIGHT\test_real_Ffisheye.jpg")
    #img2 = cv2.imread(r"D:\hi8-tape-camera-with-fisheye-lens--samsung-vpw75--58859723.png")
    fisheriser = fisheye_lib.fisheye_tool(
        img_shape=img2.shape,
        image_circle_size=max(
        img2.shape[1], img2.shape[0]))
    #fart = fisheriser.fish_eye_image(img2.copy(),reverse=True)
    plop = np.zeros_like(img2)
    ImageViewer_Quick_no_resize("original", img2,0,False,False)
    for x in range(0,plop.shape[0]):
        for y in range(0,plop.shape[1]):
            fart= fisheriser.pt_to_reverse_fisheye_inefficient(x, y)
            if fart is not None:
                x_, y_ = fart
                plop[x_,y_] = img2[x,y]
    ImageViewer_Quick_no_resize("unfish", plop,0,False,False)
    teeehee = np.zeros_like(plop)

    for x in range(0,plop.shape[0]):
        for y in range(0,plop.shape[1]):
            fart = fisheriser.pt_to_reverse_fisheye_inefficient(x, y)
            if fart is not None:
                x_, y_ = fart
                teeehee[x,y] = plop[x_,y_]

    #fart = convert_to_fisheye(img, 1000)
    
    #print(f"incoming{img2.shape}")
    #ImageViewer_Quick_no_resize("unfished",img2,0,True,False)
    #print(f"incoming{fart.shape}")
    ImageViewer_Quick_no_resize("make_fish_again", teeehee,0,True,False)

def test_single_points():
    img = cv2.imread(r"C:\Working\nonwork\SCAMBILIGHT\reaL_fish_eye_pic.png")
    fart = np.zeros_like(img)

    sample_area_left = 100
    sample_area_right = fart.shape[0] - 100
    sample_area_top = 100
    sample_area_lower = fart.shape[1] - 100
    contours = []
    # along top left to right
    lin = np.linspace(sample_area_left, sample_area_right, 500)
    for x in lin:
        contours.append((x, sample_area_top))
    # down right side
    lin = np.linspace(sample_area_top, sample_area_lower, 500)
    for y in lin:
        contours.append((sample_area_right, y))
    # along bottom right to left
    lin = np.linspace(sample_area_right, sample_area_left, 500)
    for x in lin:
        contours.append((x, sample_area_lower))
    # up left side
    lin = np.linspace(sample_area_lower, sample_area_top, 500)
    for y in lin:
        contours.append((sample_area_left, y))
    cont_ints = [(int(i[0]), int(i[1])) for i in contours]
    sample_area_lerp_contour = cont_ints
    for plop in sample_area_lerp_contour:
        fart[plop[0], plop[1], :] = (255,255,255)
    ImageViewer_Quick_no_resize("fart", fart,0,True,False)
    #fart = np.zeros_like(img)
    fisheriser = fisheye_lib.fisheye_tool(
        img_shape=fart.shape,
        image_circle_size=1146)

    jobby = []
    for plop in sample_area_lerp_contour:
        res = fisheriser.pt_to_reverse_fisheye_inefficient(plop[0], plop[1])
        if res is not None:
            jobby.append(res)
    
    #for plop in jobby:
    #    fart[plop[0], plop[1], :] = (255,255,255)
    fart =fisheriser.fish_eye_image(fart, reverse=False)
    ImageViewer_Quick_no_resize("fart", fart,0,True,False)

    #fished = fisheriser.fish_eye_image(fart, reverse=True)
    #ImageViewer_Quick_no_resize(fished,0,True,False)

if __name__ == "__main__":
    test_more6()
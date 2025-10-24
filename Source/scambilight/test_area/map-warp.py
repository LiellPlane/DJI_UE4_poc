#https://stackoverflow.com/questions/60609607/how-to-create-this-barrel-radial-distortion-with-python-opencv

from PIL import ImageOps
import numpy as np
import time
import cv2

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
class BarrelDeformer:

    def transform(self, x, y):
        # center and scale the grid for radius calculation (distance from center of image)
        x_c, y_c = w / 2, h / 2 
        x = (x - x_c) / x_c
        y = (y - y_c) / y_c
        radius = np.sqrt(x**2 + y**2) # distance from the center of image
        m_r = 1 + k_1*radius + k_2*radius**2 # radial distortion model
        # apply the model 
        x, y = x * m_r, y * m_r
        # reset all the shifting
        x, y = x*x_c + x_c, y*y_c + y_c
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        return (*self.transform(x0, y0),
                *self.transform(x0, y1),
                *self.transform(x1, y1),
                *self.transform(x1, y0),
                )

    def getmesh(self, img):
        self.w, self.h = img.size
        gridspace = 1
        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))
        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]
        return [t for t in zip(target_grid, source_grid)]
    
# adjust k_1 and k_2 to achieve the required distortion
k_1 = 0.2
k_2 = 0.05
im = ImageOps.Image.open(r"C:\Working\GIT\DJI_UE4_poc\Source\scambilight\website\images\raw_image.jpg")
im.putalpha(255)
w, h = im.size
im_deformed = ImageOps.deform(im, BarrelDeformer())
out_img = np.hstack((np.array(im), np.array(im_deformed)))
ImageViewer_Quick_no_resize(out_img,0,True,False)
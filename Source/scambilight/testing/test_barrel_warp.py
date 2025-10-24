import PIL
import cv2
import numpy as np
#https://stackoverflow.com/questions/60609607/how-to-create-this-barrel-radial-distortion-with-python-opencv
from PIL import ImageOps, Image

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
        gridspace = 20

        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))
        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]
        return [t for t in zip(target_grid, source_grid)]
    
# adjust k_1 and k_2 to achieve the required distortion
k_1 = 0.2
k_2 = 0.05
#im = Image.open('house.png')
while True:
    im = Image.open(r"D:\undistorted_k1_-0p09__k2_-0p099.png")
    im.putalpha(255)
    w, h = im.size
    im_deformed = ImageOps.deform(im, BarrelDeformer())
    #cv2.imshow('image',np.array(im))
    #cv2.waitKey(0)
    print(k_1, k_2)
    cv2.imshow('image',np.array(im_deformed))
    cv2.waitKey(0)
    k_1 += 0.1
    k_2 += 0.1

plt.figure(figsize=(12,5))
plt.imshow(np.hstack((np.array(im), np.array(im_deformed)))), plt.axis('off')
plt.title('The original and the deformed image', size=20)
plt.show()
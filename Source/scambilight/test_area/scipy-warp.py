import numpy as np
#from matplotlib import pyplot as plt

import scipy.ndimage
#img: input image

# adjust k_1 and k_2 to achieve the required distortion
k_1 = 0.2
k_2 = 0.05

#img = imread("...")

h,w = [99,100] # img size

x,y = np.meshgrid(np.float32(np.arange(w)),np.float32(np.arange(h))) # meshgrid for interpolation mapping


# center and scale the grid for radius calculation (distance from center of image)
x_c = w/2 
y_c = h/2 
x = x - x_c
y = y - y_c
x = x/x_c
y = y/y_c

radius = np.sqrt(x**2 + y**2) # distance from the center of image

m_r = 1 + k_1*radius + k_2*radius**2 # radial distortion model

# apply the model 
x= x * m_r 
y = y * m_r

# reset all the shifting
x= x*x_c + x_c
y = y*y_c + y_c

distorted = scipy.ndimage.map_coordinates(img, [y.ravel(),x.ravel()])
distorted.resize(img.shape)
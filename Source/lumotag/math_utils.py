
import math
import numpy as np

def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees
def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner

def angle_between_points(p1, p2):
    d1 = p2[0] - p1[0]
    d2 = p2[1] - p1[1]
    if d1 == 0:
        if d2 == 0:  # same points?
            deg = 0
        else:
            deg = 0 if p1[1] > p2[1] else 180
    elif d2 == 0:
        deg = 90 if p1[0] < p2[0] else 270
    else:
        deg = math.atan(d2 / d1) / math.pi * 180
        lowering = p1[1] < p2[1]
        if (lowering and deg < 0) or (not lowering and deg > 0):
            deg += 270
        else:
            deg += 90
    return deg


def get_internal_angles_of_shape(contour):
    #raise NotImplementedError("this needs to be done next!")
    #triangle=[]
    ## have to go around angle
    # but need 3 points for each angle
    # so need to add last point at start 
    # and add first point at end 
    # maybe use reduce here

    #triangle=[]
    #triangle.append((-2,1))
    #triangle.append((1,5))
    #triangle.append((3,2))
    shape = [(i[0][0], i[0][1]) for i in contour]
    extended = [shape[-1]] + shape + [shape[0]]
    angles = []
    for i in range (len(extended)-2): 
        try:
            start = np.asarray(extended[i])
            mid = np.asarray(extended[i+1])
            end = np.asarray(extended[i+2])

            start = start - mid
            end = end -mid
            dot_prod = np.dot(end, start)
            mag_start = np.linalg.norm(start)
            mag_end = np.linalg.norm(end)
            res = math.degrees(
                math.acos(dot_prod / (mag_start*mag_end)))
            angles.append(res)
        except ValueError as e:
            print("skipping", e)
        
        
    return(sum(angles))
# sss
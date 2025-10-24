
import math
import numpy as np
from typing import Literal

DIXON_Q_VALS = {
    "q90": [0.941,0.765,0.642,0.560,0.507,0.468,0.437,0.412],
    "q95": [0.970,0.829,0.710,0.625,0.568,0.526,0.493,0.466],
    "q99": [0.994,0.926,0.821,0.740,0.680,0.634,0.598,0.568],
    "value_count_for_ref": [3,4,5,6,7,8,9,10]
    }

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

def filter_close_points(contour):
    dists = []
    next_pt = 1
    for _2dpt in range (0, contour.shape[0]):
        if _2dpt == contour.shape[0]-1:
            next_pt = 0
        else:
            next_pt = _2dpt + 1
        dists.append(
            np.linalg.norm(contour[_2dpt]-contour[next_pt]))
    # find outlier of small data with q-test
    # https://en.wikipedia.org/wiki/Dixon%27s_Q_test
    dists = sorted(
        {x: i for x, i in enumerate(dists)}.items(),
        key=lambda x:x[1])
    
    bad_points = [cont for cont in dists if cont[1] < 10.0]
    if len(bad_points) == 0:
        return False, 0

    new_cnt = list(contour)

    indices = [i[0] for i in bad_points]
    indices.sort(reverse=True)
    for i in indices:
        del(new_cnt[i])

    return True, np.asarray(new_cnt)


def filter_outlier_edges(contour):
    """if a shape is well-defined but has an
    errant small edge, remove this edge.

    For example, a equilateral triangle with a small
    blunt edge would be classified as a square

    only compatible with contours between 3 and 10 points"""

    if (3 < len(contour) < 10) is False:
        # figures are taken from qtable global value
        # cannot get outlier beyond these sample sizes
        return False, 0

    dists = []
    next_pt = 1
    for _2dpt in range (0, contour.shape[0]):
        if _2dpt == contour.shape[0]-1:
            next_pt = 0
        else:
            next_pt = _2dpt + 1
        dists.append(
            np.linalg.norm(contour[_2dpt]-contour[next_pt]))
    # find outlier of small data with q-test
    # https://en.wikipedia.org/wiki/Dixon%27s_Q_test
    dists = sorted(
        {x: i for x, i in enumerate(dists)}.items(),
        key=lambda x:x[1])
    range_ = dists[-1][1] - dists[0][1]
    gap = dists[1][1] - dists[0][1]
    q_ = gap/range_
    dixon_q = get_dixonQ(
        no_of_values=contour.shape[0],
        conf_pc="q90")
    if q_ < dixon_q:
        # probably not an outlier
        return False, 0

    # probably an outlier - lets chop out that edge
    # sorted so the first item, get index, this represents
    # the distance between nth and n+1th points
    index = dists[0][0]
    new_cnt = list(contour)
    del(new_cnt[index])
    new_cnt = np.asarray(new_cnt)

    return True, new_cnt


def get_dixonQ(no_of_values, conf_pc: Literal["q90", "q95", "q99"]):

    try:
        qval = DIXON_Q_VALS[conf_pc][no_of_values - 3] # offset for qval table
    except (KeyError, IndexError):
        raise Exception("No dixon qval found for", no_of_values, conf_pc)

    return qval




import sys

import cv2
import numpy as np

from jetcam.lib.linalg import Rotation
from jetcam.lib.camera_models import (
        ExtrinsicParams,
        IntrinsicParams,
        PTRCamera,
        Resolution,
        Sensor,
        calculate_homography,
)

from common.lib.units import MiliM, Meter, Deg, Rad

PITCH_LENGTH = 37
PITCH_WIDTH = 27
CAMERA_LOCATION = [PITCH_LENGTH / 2, 0, 4]

EXTRINSICS_F = ExtrinsicParams(
    tilt=Deg(30),
    pan=Deg(50),
    roll=Deg(0),
    translation=[Meter(PITCH_LENGTH / 2), Meter(0), Meter(4)],
)



def get_virtual_cam():
    extrinsics = ExtrinsicParams(
        tilt=Deg(30),
        pan=Deg(60),
        roll=Deg(0),
        translation=[Meter(PITCH_WIDTH/2), Meter(0), Meter(4)],
    )
    camera = PTRCamera(
        extrinsics=extrinsics,
        intrinsics=IntrinsicParams(
            sensor=Sensor(width=MiliM(3.68), height=MiliM(2.76)),
            resolution=Resolution(horizontal=1920 // 4, vertical=1080 // 4),
            focal_length=MiliM(3.04),
        ),
    )
    return camera

def get_fisheye_cam():
    w_f = 400
    h_f = 300
    s_w_f = 3.68
    s_h_f = 2.76
    f_f = 1.5
    camera = PTRCamera(
        extrinsics=EXTRINSICS_F,
        intrinsics=IntrinsicParams(
            sensor=Sensor(width=MiliM(s_w_f), height=MiliM(s_h_f)),
            resolution=Resolution(horizontal=w_f, vertical=h_f),
            focal_length=MiliM(f_f),
        ),
    )
    return camera


def _fisheye_to_world_map():

    # in the theta coord vector (0, 0, 1) is the one we are interested in
    # so we need to transform this vector to first camera cs:
    vects = np.array([
            [0, 0],
            [0, 0],
            [0, 1],
    ])
    fe_cam = get_fisheye_cam()
    to_world = np.linalg.inv(EXTRINSICS_F.matrix)
    s_w_f = fe_cam.intrinsics.sensor.width.value
    s_h_f = fe_cam.intrinsics.sensor.height.value
    f_f = fe_cam.intrinsics.focal_length.value
    def _fish_coords(x_f, y_f):
        u_f = (x_f - w_f / 2) / w_f * s_w_f
        v_f = (y_f - h_f / 2) / h_f * s_h_f

        r_f = (u_f ** 2 + v_f ** 2) ** 0.5
        theta = 2 * np.arctan2(r_f, 2 * f_f)

        r_r = np.tan(theta) * f_f

        b = r_r / r_f if r_f else 0
        u_r = u_f * b
        v_r = v_f * b

        vect = np.array([u_r, v_r, f_f, 0]).reshape(-1, 1) / 1e3
        in_world_cs = to_world @ vect
        (x_d, y_d, z_d, _), = in_world_cs.T
        if z_d >= 0:
            return -1, -1

        c_x, c_y, c_z = CAMERA_LOCATION
        a = -c_z / z_d

        x_p = c_x + a * x_d
        y_p = c_y + a * y_d
        return x_p * 10, y_p * 10

    result = []
    w_f, h_f = fe_cam.intrinsics.resolution
    for y in range(h_f):
        if not y % 10:
            print(format(y / h_f * 100, '.1f') + '%')
        result.append([
            _fish_coords(x, y)
            for x in range(w_f)
        ])
    
    return np.array(result, dtype=np.float32)


def calc_virtual_view_map(fe_view):
    virt_cam = get_virtual_cam()
    fe_cam = get_fisheye_cam()

    w_r, h_r = virt_cam.intrinsics.resolution
    s_w_r, s_h_r = virt_cam.intrinsics.sensor
    s_w_r = s_w_r.value
    s_h_r = s_h_r.value
    f_v = virt_cam.intrinsics.focal_length.value

    to_world_cs_from_v = np.linalg.inv(virt_cam.extrinsics.matrix)
    extr_f = fe_cam.extrinsics

    def virtual_view_to_fe(virt_x, virt_y):
        u_v = (virt_x - w_r / 2) / w_r * s_w_r
        v_v = (virt_y - h_r / 2) / h_r * s_h_r

        vect = np.array([u_v, v_v, f_v, 0]).reshape(-1, 1) / 1e3

        in_world_cs = to_world_cs_from_v @ vect
        (x_d, y_d, z_d, _), = in_world_cs.T

        c_x, c_y, c_z = CAMERA_LOCATION
        a = -c_z / z_d

        x_w = c_x + a * x_d
        y_w = c_y + a * y_d

        in_fe_cs = extr_f.matrix @ np.array(
                [x_w, y_w, 0, 1]).reshape(-1, 1)

        (x_f, y_f, z_f, _), = in_fe_cs.T
        if z_f <= f_v/1e3:
            return -1, -1
      
        f_f = fe_cam.intrinsics.focal_length.value / 1e3
        a = f_f / z_f
        u_r = a * x_f
        v_r = a * y_f
        r_r = (u_r ** 2 + v_r ** 2) ** 0.5

        theta = np.arctan2(r_r, f_f)
        r_f = 2 * f_f * np.tan(theta / 2)

        b = r_f / r_r if r_r else 0

        u_f = b * u_r
        v_f = b * v_r
      
        s_w_f = fe_cam.intrinsics.sensor.width.value / 1e3
        s_h_f = fe_cam.intrinsics.sensor.height.value / 1e3
        w_f, h_f = fe_cam.intrinsics.resolution
        p_x_f = u_f / s_w_f * w_f + w_f / 2
        p_y_f = v_f / s_h_f * h_f + h_f / 2
        return p_x_f, p_y_f
        
    result = []
    w_r, h_r = virt_cam.intrinsics.resolution
    virtual_view_to_fe(0, h_r * 2 / 3)
    for y in range(h_r):
        if not y % 10:
            print(format(y / h_r * 100, '.1f') + '%')
        result.append([
            virtual_view_to_fe(x, y)
            for x in range(w_r)
        ])
    
    return np.array(result, dtype=np.float32)


def get_pitch():
    max_w = PITCH_LENGTH * 10
    max_h = PITCH_WIDTH * 10
    img = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    cv2.rectangle(
            img,
            (0, 0),
            (max_w, max_h),
            color=(0, 128, 0),
            thickness=-1,
    )
    cv2.rectangle(
            img,
            (0, 0),
            (max_w, max_h),
            color=(255, 255, 255),
            thickness=5,
    )
    cv2.line(
            img,
            (max_w // 2, 0),
            (max_w // 2, max_h),
            color=(255, 255, 255),
            thickness=5,
    )
    cv2.circle(
            img,
            (max_w // 2, max_h // 2),
            radius=50,
            thickness=5,
            color=(255, 255, 255),
    )
    thed = 70
    cv2.rectangle(
            img,
            (0, max_h // 2 - thed),
            (thed, max_h // 2 + thed),
            color=(255, 255, 255),
            thickness=5,
    )
    cv2.rectangle(
            img,
            (max_w - thed, max_h // 2 - thed),
            (max_w, max_h // 2 + thed),
            color=(255, 255, 255),
            thickness=5,
    )

    return img



def main():
    pitch = get_pitch()
    try:
        mymap = np.load('.mymap.npy')
    except FileNotFoundError:
        mymap = _fisheye_to_world_map()
        np.save('mymap', mymap)

    from matplotlib import pyplot as plt
    # plt.plot([x for x, y in mymap[:, -1]])
    # plt.show()

    conv = cv2.remap(
            pitch, mymap, None, interpolation=cv2.INTER_LINEAR)
    virtual_view_map = calc_virtual_view_map(conv)
    virtual_view = cv2.remap(
            conv, virtual_view_map, None, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('fe', conv)
    cv2.imshow('pitch', pitch)
    cv2.imshow('virtual_view', virtual_view)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
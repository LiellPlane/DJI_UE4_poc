from initial_testing import (
    Scambi_unit,
    ImageViewer_Quick_no_resize,
    real_fish_eye_cam,
    get_homography,
    get_led_perimeter_pos,
    create_rectangle_from_centrepoint)
import os
import json
import cv2
import numpy as np
import fisheye_lib

def GetAllFilesInFolder_Recursive(root):
    ListOfFiles=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            FullpathOfFile=(os.path.join(path, name))
            ListOfFiles.append(FullpathOfFile)
    return ListOfFiles


def main():
    all_files=GetAllFilesInFolder_Recursive(r"C:\Working\nonwork\SCAMBILIGHT\real_fisheye_w_trackers_imgs")
    all_json = [i for i in all_files if i[-4:]=="json"]
    all_imgs = [i for i in all_files if i[-3:]=="jpg"]
    all_json_sorted = {int(i.split("\\")[-1].replace(".json","")): i for i in all_json}
    myKeys = list(all_json_sorted.keys())
    myKeys.sort()
    sorted_dict = {i: all_json_sorted[i] for i in myKeys}
    
    no_leds_vert = 10
    no_leds_horiz = 10
    move_in_horiz = 0.20
    move_in_vert = 0.20
    resize_ratio = 1.0 #expected input res 1080 * 1920
    sample_area_edge = 70 * resize_ratio
    demo_border_size = 10
    rfish = real_fish_eye_cam

    for index, img_info in enumerate(sorted_dict.values()):
        data = json.load( open( img_info ))
        image = cv2.imread(data["file"])
        pts = data["points"]
        if len(pts) != 4:
            continue

        fisheriser = fisheye_lib.fisheye_tool(
            img_width_height=(rfish.width, rfish.height),
            image_circle_size=rfish.fish_eye_circle)

        reverse_fish_pts = []
        for pt in pts:
            reverse_fish_pts.append(
                fisheriser.pt_to_reverse_fisheye_inefficient(pt[0], pt[1]))
        if None in reverse_fish_pts:
            continue

        homography_tool = get_homography(
            img_height_=rfish.height,
            img_width_=rfish.width,
            corners=np.asarray(reverse_fish_pts, dtype="float32"),
            target_corners=rfish.targets,
            resize_ratio=resize_ratio)

        prev = image

        scambi_units = []
        led_positions = get_led_perimeter_pos(prev, no_leds_vert, no_leds_horiz)
        for centre_, edgename in led_positions:
                centre_ = tuple((np.asarray(centre_)).astype(int))
                #cv2.circle(prev,plop,16,(255,0,100),-1)
                mid_screen = (np.array(tuple(reversed(prev.shape[:2])))/2).astype(int)[:2]
                vec_to_midscreen = mid_screen-np.asarray(centre_)
                #cv2.circle(prev,tuple(mid_screen),16,(255,0,100),-1)
                if edgename not in ["top", "lower", "left", "right"]:
                    raise Exception("edge name " + edgename + "not valid")
                if edgename in ["top", "lower"]:
                    new_pos = tuple((np.asarray(centre_) + (vec_to_midscreen * move_in_vert)).astype(int))
                if edgename in ["left", "right"]:
                    new_pos = tuple((np.asarray(centre_) + (vec_to_midscreen * move_in_horiz)).astype(int))
                left, right, top, lower = create_rectangle_from_centrepoint(new_pos, edge=sample_area_edge)
                scambi_units.append(Scambi_unit(
                    led_positionxy=centre_,
                    sample_area_left=left,
                    sample_area_right=right,
                    sample_area_top=top,
                    sample_area_lower=lower,
                    inverse_warp_m=homography_tool.inverse_trans_matrix,
                    img_shape=prev.shape,
                    img_circle=real_fish_eye_cam.fish_eye_circle)
                )

        for unit in scambi_units:
            unit.colour = unit.get_dominant_colour_flat(prev)
            prev = unit.draw_lerp_contour(prev)

        ImageViewer_Quick_no_resize(prev,0,False,False)
        filename = "D://fish_eye_jsons//" + "0" + str(index) + ".jpg"
        cv2.imwrite(filename, prev)
if __name__ == "__main__":
    main()

import requests
import json
import numpy as np

from libs.utils import (
    get_platform,
    _OS,
    TimeDiffObject,
    ImageViewer_Quick_no_resize,
    encode_img_to_str,
    img_height,
    img_width)
from libs.collections import (
    LedSpacing,
    Edges,
    lens_details,
    LedsLayout,
    config_corner)
from img_processing import clahe_equalisation

def get_corners_from_remote_config(config, img):
    """find corners from disorder of inputs in format:
    {
                "clickx": 299
                "clicky": 339}
    """
    #min_x = min([i['clickx'] for i in config])
    corners = {}
    corners["top_left"] = [0, 0]
    corners["top_right"] = [img_width(img), 0 ]
    corners["lower_right"] = [img_width(img), img_height(img),]
    corners["lower_left"] =  [0, img_height(img)]
    list_config_pts = [[i['clickX'], i['clickY']] for i in config]
    for pt_id, pt_coord in corners.items():
        match_pt, list_config_pts = find_closest(pt_coord, list_config_pts)
        arse = config_corner(flat_corner=corners[pt_id], real_corner=match_pt)
        corners[pt_id] = arse
    return corners


def find_closest(testpt: list [int, int], input_pts:list):
    dists = {
        np.linalg.norm(np.asarray(testpt)-np.asarray(i)):i
        for i in input_pts}
    pt = dists[sorted(dists)[0]]
    return pt, [i for i in input_pts if i != pt]

def upload_img_to_aws(img, url, action):
    
    print("uploading image")
    if action == "raw":
        action = "image_raw"
    elif action =="overlay":
        action = "image_overlay"
    else:
        raise Exception("bad action")
    img = clahe_equalisation(img, None)
    img_bytes = encode_img_to_str(img)
    myobj = {
        "authentication": "farts",
        "action": action,
        "payload": img_bytes
        }
    try:
        response = requests.post(url, json=myobj)
        print("Auto uploading image", response.text)
    except requests.exceptions.RequestException as e:
        print(e)
        print("could not connect first image upload to ", url)
    

def get_config_from_aws(url):
    print("getting config from aws")
    myobj = {
        "authentication": "farts",
        "action": "request_config"
        }
    positions = []
    try:
        response = requests.post(url, json=myobj)
        #TODO not good - why is this so arduous - can't be right
        clicked_positions = json.loads(json.loads(response.content)['config'])

        for elem in clicked_positions:
            # sorry
            positions.append({i:int((elem)[i]) for i in elem})
    except (requests.exceptions.RequestException, KeyError) as e:
        print(e)
        print("could not connect get config or find key from", url)

    return positions
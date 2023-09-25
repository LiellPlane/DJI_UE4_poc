
import requests
import json
import numpy as np
import time
import random

from factory import Camera_synchronous, ImageGenerator

from libs.utils import (
    encode_img_to_str,
    decode_image_from_str,
    str_to_bytes,
    bytes_to_str,
    img_height,
    img_width)

from libs.collections import (
    config_corner,
    clicked_xy,
    External_Config,
    config_regions)

from img_processing import clahe_equalisation
from multiprocessing import Process, Queue
from common import cors_headers


def send_sim_progress_update_to_AWS(progress_im):
    print("aws", progress)

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
    #list_config_pts = [[i['clickX'], i['clickY']] for i in config]
    list_config_pts = [[i.clickX, i.clickY] for i in config]
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




# class Aws_Camera_sync(Camera_synchronous):
    
#     def __init__(self, video_modes) -> None:
#         super().__init__(video_modes, ImageLibrary)


# class ImageLibrary(ImageGenerator):
    
#     def __init__(self, res) -> None:
#         pass


#     def get_image(self):
#         return get_image_from_aws()


def get_image_from_aws(url):
    print("getting raw image from aws")
    myobj = {
        "authentication": "farts",
        "action": "getimage_raw"
        }
    try:
        response = requests.post(url, json=myobj)
    except (requests.exceptions.RequestException, KeyError) as e:
        print(e)
        print("could not connect get config or find key from", url)
        return "Error connecting"
    events = json.loads(response.content)
    encoded_img = events['image']
    return decode_image_from_str(encoded_img)


def upload_img_to_aws(img, url, action):
    
    print("uploading image")
    if action == "raw":
        action = "image_raw"
    elif action =="overlay":
        action = "image_overlay"
    elif action =="perpwarp":
        pass
    else:
        raise Exception("bad action")
    #img = clahe_equalisation(img, None)
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

def check_event_validity(event: str):
    if event.lower() not in ["reset", "update_image", "none", "update_image_all"]:
        raise Exception("event malformed")

def check_events_from_aws(url):
    myobj = {
        "authentication": "farts",
        "action": "check_event"
        }
    try:
        response = requests.post(url, json=myobj)
    except (requests.exceptions.RequestException, KeyError) as e:
        print(e)
        print("could not connect get config or find key from", url)
        return "Error connecting"
    events = json.loads(response.content)
    if len(events) > 0:
        event = list(events[0].values())[0]
    else:
        event="None"

    check_event_validity(event)
    return event


class ExternalDataWorker_dummy():
    def __init__(
            self,
            url):
        self.url = url

    def _start(self):
        pass

    def check_for_action(self):
        return "None"


class ExternalDataWorker():
    def __init__(
            self,
            url):
        self.in_queue = Queue(maxsize=1)
        self.msg_queue = Queue(maxsize=1)
        self.url = url

    def _start(self):
    
        process = Process(
            target=self._run,
            args=(),
            daemon=True)

        process.start()

    def _run(self):
        while True:
            event = check_events_from_aws(self.url)
            time.sleep(10)
            self.msg_queue.put(
                event, block=True, timeout=None)

    def check_for_action(self):
        if self.msg_queue.empty():
            pass
        else:
            try:
                event = self.msg_queue.get_nowait()
                return event
            except Queue.Empty:
                pass
        return "None"
    
    
def get_config_from_aws(url):
    print("getting config from aws")
    myobj = {
        "authentication": "farts",
        "action": "request_config"
        }
    positions = []
    ext_config_pos = []
    try:
        response = requests.post(url, json=myobj)
        #TODO not good - why is this so arduous - can't be right
        clicked_positions = json.loads(json.loads(response.content)['config'])

        for elem in clicked_positions:
            # sorry
            positions.append({i:int((elem)[i]) for i in elem})
            
            ext_config_pos.append(clicked_xy(**elem))
        print(f"from AWS {clicked_positions}")
    except (requests.exceptions.RequestException, KeyError) as e:
        print(e)
        print("could not connect get config or find key from", url)
    return External_Config(
        fish_eye_clicked_corners=ext_config_pos)


def get_region_config_from_aws(url):
    print("getting config from aws")
    myobj = {
        "authentication": "farts",
        "action": "request_sample_config"
        }
    positions = []
    ext_config_pos = []
    try:
        response = requests.post(url, json=myobj)
        #TODO not good - why is this so arduous - can't be right
        ext_regions_config = json.loads(json.loads(response.content)['config'])
        
        expected_keys = list(config_regions.__dataclass_fields__.keys())
        incoming_keys = list(ext_regions_config.keys())

        if not set(expected_keys) == set(incoming_keys):
            print("expected_keys", expected_keys)
            print("incoming_keys", incoming_keys)
            raise Exception("incoming region config does not match")
        configured_regions = config_regions(**{k: float(v) for k, v in ext_regions_config.items()})
        return configured_regions
    except (requests.exceptions.RequestException, KeyError) as e:
        print(e)
        print("could not connect get config or find key from", url)
    
    return None

def get_ext_corners_or_use_default(
        ext_click_data: External_Config,
        default_corners,
        imgshape: any):
    if len(ext_click_data) > 3:
        real_corners = get_corners_from_remote_config(
            ext_click_data,
            imgshape)
        real_corners = [real_corners['top_left'].real_corner,
        real_corners['top_right'].real_corner,
        real_corners['lower_right'].real_corner,
        real_corners['lower_left'].real_corner]
        return real_corners
    else:
        print("not enough positions in remote config ", ext_click_data)
    return default_corners

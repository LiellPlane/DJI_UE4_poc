
import requests
import json
import numpy as np
import time
import random
from abc import ABC, abstractmethod
#from factory import filesystem_scambilight

from libs.utils import (
    encode_img_to_str,
    decode_image_from_str,
    str_to_bytes,
    bytes_to_str,
    img_height,
    img_width,
    get_platform)

from libs.collections import (
    config_corner,
    clicked_xy,
    lens_details,
    External_Config,
    config_regions,
    AllConfiguration,
    PhysicalTV_details)

from img_processing import clahe_equalisation
from multiprocessing import Process, Queue
from common import cors_headers
import libs.configs as configs


class filesystem_scambilight(ABC):
    def __init__(self) -> None:
        """file system specific to scambilight"""
        self.rootdir =  configs.RPI_ROOTDIR # probably should be in config
        self.configfile = configs.CONFIG_FILENAME
        self.session_token_file = configs.SESSIONTOKEN_FILENAME
        self.sessiontoken_key = configs.CONFIG_FILENAME
        self.config_key = configs.SESSIONTOKEN_FILENAME

    @abstractmethod
    def read_jsonfile(self, path: str)->dict:
        pass

    @abstractmethod
    def write_jsonfile(self, path:str, object_dict:json)->None:
        pass

    def get_filepath(self, input_filename: str)->str:
        return f"{self.rootdir}{input_filename}"
    
    @property
    def get_config_file(self):
        return self.read_jsonfile(self.get_filepath(self.configfile))[self.config_key]
    
    @property
    def get_session_token_file(self):
        return self.read_jsonfile(self.get_filepath(self.session_token_file))[self.sessiontoken_key]

    def save_config_file(self, input_dict: dict):
        """save the config file to the filesystem
        provide the input dictionary, this function will
        handle the particulars of the filesystem"""
        to_save = {self.config_key: input_dict}
        json_dict = json.dumps(to_save)
        return self.write_jsonfile(
            self.get_filepath(self.configfile),
            object_dict=json_dict
            )

    def save_session_file(self, input_str: dict):
        """save the config file to the filesystem
        provide the input dictionary, this function will
        handle the particulars of the filesystem"""
        to_save = {self.sessiontoken_key: input_str}
        json_dict = json.dumps(to_save)
        return self.write_jsonfile(
            self.get_filepath(self.session_token_file),
            object_dict=json_dict
            )

class sim_file_system(filesystem_scambilight):
    def __init__(self) -> None:
        super().__init__()
        self.configmemory = None
        self.session_memory = None
        self.save_session_file("admin")
        self.save_config_file(input_dict={"config": "TBC"})
        testconfig = self.get_config_file
        testsession = self.get_session_token_file

    def read_jsonfile(self, path: str)->dict:
        if "config" in path:
            return json.loads(self.configmemory)
        if "session" in path:
            return json.loads(self.session_memory)
    def write_jsonfile(self, path:str, object_dict:json)->None:
        if "config" in path:
            self.configmemory = object_dict
        if "session" in path:
            self.session_memory = object_dict


class raspberry_file_system(filesystem_scambilight):
    def __init__(self) -> None:
        super().__init__()
        # override rootdir class member
        
        #self.configmemory = None
        #self.session_memory = json.dumps("daisybankscambi")
        # do this in the meantime until we have a comissioning system
        self.save_config_file(input_dict={"config": "TBC"})
        self.save_session_file("daisybankscambi")

    def read_jsonfile(self, path: str) -> dict:
        # if "config" in path:
        #     return json.loads(self.configmemory)
        # if "session" in path:
        #     return json.loads(self.session_memory)
        with open(path, 'r') as file:
            data = json.load(file)
            return data

    def write_jsonfile(self, path:str, object_dict:json)->None:
        with open(path, 'w') as file:
            file.write(object_dict)
        # if "config" in path:
        #     self.configmemory = json.dumps(object_dict)
        # if "session" in path:
        #     self.session_memory = json.dumps(object_dict)

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


# def get_session_id():
#     # has to match the form from the website
#     # probably should fix this
#     return json.dumps("admin")


def get_image_from_aws(url, sessiontoken):
    print("getting raw image from aws")
    myobj = {
        "action": "getimage_raw",
        "sessiontoken": json.dumps(sessiontoken)
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


def upload_img_to_aws(img, url, action, sessiontoken):
    
    print("uploading image")

    #img = clahe_equalisation(img, None)
    img_bytes = encode_img_to_str(img)
    myobj = {
        "action": action,
        "payload": img_bytes,
        "sessiontoken": json.dumps(sessiontoken)
        }
    try:
        response = requests.post(url, json=myobj)
        print("Auto uploading image", response.text)
    except requests.exceptions.RequestException as e:
        print(e)
        print("could not connect first image upload to ", url)


def check_event_validity(event: str):
    # todo, should be made common somehow
    if event.lower() not in ["reset", "update_image", "none", "update_image_all", ""]:
        raise Exception("event malformed")


def check_events_from_aws(url, sessiontoken):
    myobj = {
        "action": "check_event",
        "sessiontoken": json.dumps(sessiontoken)
        }
    try:
        response = requests.post(url, json=myobj)
    except (requests.exceptions.RequestException, KeyError) as e:
        print(e)
        print("could not connect get config or find key from", url)
        return "Error connecting"
    events = json.loads(response.content)
    if len(events) > 0:
        event = events#list(events[0].values())[0]
    else:
        event="None"

    check_event_validity(event)
    return event


class ExternalDataWorker_dummy():
    def __init__(
            self,
            url,
            sessiontoken):
        self.url = url
        self.sessiontoken = sessiontoken

    def _start(self):
        pass

    def check_for_action(self):
        return "None"


class ExternalDataWorker():
    def __init__(
            self,
            url,
            sessiontoken):
        self.in_queue = Queue(maxsize=1)
        self.msg_queue = Queue(maxsize=1)
        self.url = url
        self.sessiontoken = sessiontoken
    def _start(self):
    
        process = Process(
            target=self._run,
            args=(),
            daemon=True)

        process.start()

    def _run(self):
        while True:
            try:
                event = check_events_from_aws(self.url, self.sessiontoken)
            except Exception as e:
                event = f"ERROR{e}"
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
    

def get_corners(body)->External_Config:
    positions = []
    clicked_positions = json.loads(body['corners'])
    ext_config_pos = []
    for elem in clicked_positions:
        # sorry
        positions.append({i:int((elem)[i]) for i in elem})
        
        ext_config_pos.append(clicked_xy(**elem))

    return External_Config(fish_eye_clicked_corners=ext_config_pos)


def get_lens_details(body)->lens_details:
    lens_config = json.loads(body['lens'])
    #lens_config.update({"corners": []})
    return lens_details(**lens_config)

def get_physical_tv_details(body)->PhysicalTV_details:
    details = json.loads(body['physical_tv_details'])
    return PhysicalTV_details(details)

def get_sample_region_details(body)->config_regions:
    ext_regions_config = json.loads(body['regions'])
    
    expected_keys = list(config_regions.__dataclass_fields__.keys())
    incoming_keys = list(ext_regions_config.keys())

    if not set(expected_keys) == set(incoming_keys):
        print("expected_keys", expected_keys)
        print("incoming_keys", incoming_keys)
        raise Exception("incoming region config does not match")
    return  config_regions(**{k: float(v) for k, v in ext_regions_config.items()})
 


def get_all_config_external(url, sessiontoken)->AllConfiguration:
    """ get all config simultaneously then
    cache result for future calls.
    this should only be updated on reset"""
    print("getting all config from aws")
    myobj = {
        "action": "getconfig",
        "sessiontoken": json.dumps(sessiontoken)
        }

    response = requests.post(url, json=myobj)

    body = json.loads(response.content)

    corners = get_corners(body)
    sample_regions = get_sample_region_details(body)
    lens_details = get_lens_details(body)
    phsyical_tv_details = get_physical_tv_details(body)
    return AllConfiguration(
        lens_details=lens_details,
        clicked_corners=corners,
        sample_regions=sample_regions,
        physical_tv_details=phsyical_tv_details
        )


    

    print(f"from AWS {clicked_positions}")
    # except (requests.exceptions.RequestException, KeyError) as e:
    #     print(e)
    #     print("could not connect get config or find key from", url)
    # return External_Config(
    #     fish_eye_clicked_corners=ext_config_pos)


# def get_config_from_aws(url, sessiontoken):
#     print("getting config from aws")
#     myobj = {
#         "action": "getconfig",
#         "sessiontoken": json.dumps(sessiontoken)
#         }
#     positions = []
#     ext_config_pos = []
#     try:
#         response = requests.post(url, json=myobj)
#         #TODO not good - why is this so arduous - can't be right
#         clicked_positions = json.loads(json.loads(response.content)['corners'])

#         for elem in clicked_positions:
#             # sorry
#             positions.append({i:int((elem)[i]) for i in elem})
            
#             ext_config_pos.append(clicked_xy(**elem))
#         print(f"from AWS {clicked_positions}")
#     except (requests.exceptions.RequestException, KeyError) as e:
#         print(e)
#         print("could not connect get config or find key from", url)
#     return External_Config(
#         fish_eye_clicked_corners=ext_config_pos)


# def get_region_config_from_aws(url):
#     print("getting config from aws")
#     myobj = {
#         "action": "getconfig",
#         "sessiontoken": get_session_id()
#         }
#     positions = []
#     ext_config_pos = []
#     try:
#         response = requests.post(url, json=myobj)
#         #TODO not good - why is this so arduous - can't be right
#         ext_regions_config = json.loads(json.loads(response.content)['regions'])
        
#         expected_keys = list(config_regions.__dataclass_fields__.keys())
#         incoming_keys = list(ext_regions_config.keys())

#         if not set(expected_keys) == set(incoming_keys):
#             print("expected_keys", expected_keys)
#             print("incoming_keys", incoming_keys)
#             raise Exception("incoming region config does not match")
#         configured_regions = config_regions(**{k: float(v) for k, v in ext_regions_config.items()})
#         return configured_regions
#     except (requests.exceptions.RequestException, KeyError) as e:
#         print(e)
#         print("could not connect get config or find key from", url)
    
#     return None

def calculate_which_corner(
    ext_click_data: External_Config,
    imgshape: any):
    """pass in corners clicked by user, and transform
    them to the format used here, also need to 
    discover what corner is the top left, top right 
    etc as currently this is not known"""
    real_corners = get_corners_from_remote_config(
        ext_click_data,
        imgshape)
    real_corners = [
        real_corners['top_left'].real_corner,
        real_corners['top_right'].real_corner,
        real_corners['lower_right'].real_corner,
        real_corners['lower_left'].real_corner
        ]
    return real_corners

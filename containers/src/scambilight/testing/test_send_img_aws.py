import requests
import json
from typing import List, Sequence
import base64
import uuid

import cv2
import numpy as np

    
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


def bytes_to_str(bytes_: bytes):
    return bytes_.decode()


def str_to_bytes(string_: str):
    return str.encode(string_)


def encode_img_to_str(img: np.ndarray):
    """Encode single image for compatibility with json msg

    Args:
        thumb: image as numpy array

    Returns:
        str"""
    img_string = base64.b64encode(
            cv2.imencode(
                ext='.jpg',
                img=img)[1]).decode()
    return img_string

def decode_image_from_str(encoded_image: str):
    """decodes image from string. Expects base64 encoding

    Args:
        encoded_image: str representing image

    Returns:
        np.array image"""
    jpg_original = base64.b64decode(encoded_image)
    return jpg_original
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)

    return cv2.imdecode(buf=jpg_as_np, flags=1)

def main():

    url = "https://yqnz152azi.execute-api.us-east-1.amazonaws.com/Prod/hello"
    img_path = r"C:\VMs\SharedFolder\temp_get_imgs\00.jpg"
    img = cv2.imread(img_path)

    #img2 = encode_img_to_str(img)
    #img3 = decode_image_from_str(img2)
    #ImageViewer_Quick_no_resize(img,0,True,False)


    img_string = encode_img_to_str(img)
    myobj = {
        "authentication": "farts",
        "action": "image_overlay",
        "payload": img_string
        }
    try:
        response = requests.post(url, json=myobj)
        print(response.text)
    except ConnectionError:
        print("could not connect")
    


    #getimage_raw
    #getimage_overlay
    get_img = {
        "authentication": "farts",
        "action": "getimage_raw"
        }

    response = requests.post(url, json=get_img)
    img_str = json.loads(response.text)["image"]
    img = decode_image_from_str(img_str)
    ImageViewer_Quick_no_resize(img,0,True,False)
if __name__ == '__main__':
    main()

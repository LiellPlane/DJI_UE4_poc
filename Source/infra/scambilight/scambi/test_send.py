import requests
import json
from typing import List, Sequence
import base64
import uuid

import cv2
import numpy as np


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


def main():
    url = "https://9dmar0gfgg.execute-api.us-east-1.amazonaws.com/Stage/hello"

    myobj = {
        "authentication": "farts",
        "item": "image",
        "payload": "nothing"
        }

    response = requests.post(url, json=myobj)

    print(response.text)


if __name__ == '__main__':
    main()

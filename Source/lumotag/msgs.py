import base64
import uuid
import cv2
from dataclasses import dataclass
from numpy import frombuffer, uint8
import numpy.typing as npt

@dataclass
class Report:
    my_id: str
    target: str
    timestamp: float
    img_as_str: str


@dataclass
class EQUIPMENT_ID:
    my_id: str

def generate_id():
    return f"ID{uuid.uuid4().hex}"

def bytes_to_str(bytes_: bytes):
    return bytes_.decode()

def str_to_bytes(string_: str):
    return str.encode(string_)

def encode_img_to_str(img: npt.NDArray):
    img_string = base64.b64encode(
            cv2.imencode(
                ext='.jpg',
                img=img)[1]).decode()
    return img_string


def decode_image_from_str(encoded_image: str):
    jpg_original = base64.b64decode(encoded_image)
    jpg_as_np = frombuffer(jpg_original, dtype=uint8)

    return cv2.imdecode(buf=jpg_as_np, flags=1)

import base64
import uuid
import cv2
from dataclasses import dataclass, asdict
import factory
import json
from numpy import frombuffer, uint8
from datetime import datetime
import time
from enum import Enum, auto
#import numpy.typing as npt


class AutoStrEnum(str, Enum):
    """
    StrEnum where auto() returns the field name.
    See https://docs.python.org/3.9/library/enum.html#using-automatic-values
    """
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name


class MessageTypes(AutoStrEnum):
    ERROR = auto()
    HIT_REPORT = auto()
    HELLO = auto()


@dataclass
class Report:
    my_id: str
    target: str
    timestamp: float
    img_as_str: str
    msg_type: str
    msg_string: str


@dataclass
class EQUIPMENT_ID:
    my_id: str

def generate_id():
    return f"ID{uuid.uuid4().hex}"

def bytes_to_str(bytes_: bytes):
    return bytes_.decode()

def str_to_bytes(string_: str):
    return str.encode(string_)

def encode_img_to_str(img):
    if img is None:
        return None
    img_string = base64.b64encode(
            cv2.imencode(
                ext='.jpg',
                img=img)[1]).decode()
    return img_string

def decode_image_from_str(encoded_image: str):
    if encoded_image is None:
        return None
    jpg_original = base64.b64decode(encoded_image)
    jpg_as_np = frombuffer(jpg_original, dtype=uint8)

    return cv2.imdecode(buf=jpg_as_np, flags=1)

def get_epoch_ts():
    return time.time()

def get_timestamp():
    """Timestamp should be in epoch time - but 
    for now make it human readable"""
    dt_obj = datetime.utcnow()
    return dt_obj

def human_readable_timestamp(dt_obj: datetime):
    return dt_obj.strftime('%H:%M:%S,%f')

def package_send_report(
        type_,
        image,
        target: str,
        messenger: factory.messenger,
        gun_config: factory.gun_config,
        message_str: str):
    """Send outgoing report, for example if
    a user has been identified during a trigger event
    
    input 
        type: value from MessageType enum
        image: numpy image | None
        target: identity if known, or enums TBD
        messenger: message object
        gun_config - correct gun config for equipment"""

    if type_ not in MessageTypes.__members__.keys():
        raise KeyError("outbound type not found in messagetype")

    msg_to_send = Report(
        my_id=gun_config.my_id,
        target=target,
        timestamp=get_epoch_ts(),
        img_as_str=encode_img_to_str(image),
        msg_type=type_,
        msg_string=message_str
    )
    messenger.send_message(
        package_dataclass_to_bytes(msg_to_send))

def package_dataclass_to_bytes(dataclass_):
    msg = str_to_bytes(json.dumps(asdict(dataclass_)))
    return msg
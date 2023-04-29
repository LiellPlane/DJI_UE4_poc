import base64
import uuid
import cv2
from dataclasses import dataclass, asdict, field
import factory
import json
from numpy import frombuffer, uint8
from datetime import datetime
import time
from enum import Enum, auto
from json.decoder import JSONDecodeError
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
    TEST = auto()


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
        messenger: factory.Messenger,
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

@dataclass
class Parsed_Msg():
    success: bool  = field(default=False)
    received_ts: any = field(default=None)
    msg_body: Report = field(default=None)
    error: str = field(default=None)

def parse_input_msg(in_msg: bytes):

    if in_msg is None:
        return(
            Parsed_Msg(
            success=False,
            error="input message empty"))

    if isinstance(in_msg, bytes) is False:
        return(
            Parsed_Msg(
            success=False,
            error="input message malformed"))

    try:
        msg = json.loads(bytes_to_str(in_msg))
    except JSONDecodeError:
        return(
            Parsed_Msg(
            success=False,
            error="error decoding message"))

    try:
        msg = Report(**msg)
    except TypeError:
        return(
            Parsed_Msg(
            success=False,
            error="message not in correct dataclass formt"))

    parsed_msg = Parsed_Msg()
    parsed_msg.received_ts = get_epoch_ts()
    parsed_msg.msg_body = msg
    parsed_msg.success = True

    return parsed_msg


def create_test_msg() -> bytes:
    msg_to_send = Report(
        my_id=factory.create_id(),
        target=None,
        timestamp=get_epoch_ts(),
        img_as_str=None,
        msg_type=MessageTypes.TEST.value,
        msg_string="5 6 7 8"
    )

    return package_dataclass_to_bytes(msg_to_send)


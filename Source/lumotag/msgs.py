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
from my_collections import AutoStrEnum
import hashlib
from pathlib import Path
import re
import subprocess
#import numpy.typing as npt

# def _compile_proto_file(proto_file: Path):
#     """Compiles a proto file and adds its hash to the generated file."""
#     # Calculate hash before compilation
#     proto_hash = hashlib.md5(proto_file.read_bytes()).hexdigest()
    
#     # Compile the proto file
#     result = subprocess.run(
#         ['protoc', f'--python_out=.', str(proto_file)],
#         capture_output=True,
#         text=True
#     )
    
#     if result.returncode != 0:
#         raise RuntimeError(f"Failed to compile {proto_file}: {result.stderr}")
    
#     # Add the hash to the generated file
#     generated_file = proto_file.parent / f"{proto_file.stem}_pb2.py"
#     with open(generated_file, 'r') as f:
#         content = f.read()
    
#     with open(generated_file, 'w') as f:
#         f.write(f'# Proto file hash: {proto_hash}\n')
#         f.write(content)

# # Validate proto files on module import
# def _validate_proto_files():
#     """Validates that the generated Python files match their proto definitions."""
#     proto_dir = Path(__file__).parent / 'protobuffers'
#     proto_files = list(proto_dir.glob('*.proto'))
    
#     for proto_file in proto_files:
#         # Change the suffix to _pb2.py for the generated file
#         generated_file = proto_file.parent / f"{proto_file.stem}_pb2.py"
        
#         if not generated_file.exists():
#             print(f"Compiling {proto_file} as generated file doesn't exist...")
#             _compile_proto_file(proto_file)
#             continue
            
#         # Calculate current proto file hash
#         current_hash = hashlib.md5(proto_file.read_bytes()).hexdigest()
        
#         # Read the generated file and extract the stored hash
#         generated_content = generated_file.read_text()
#         hash_match = re.search(r'# Proto file hash: ([a-f0-9]{32})', generated_content)
        
#         if not hash_match:
#             print(f"Recompiling {proto_file} as generated file doesn't contain a valid hash...")
#             _compile_proto_file(proto_file)
#             continue
            
#         stored_hash = hash_match.group(1)
        
#         # Compare hashes
#         if current_hash != stored_hash:
#             print(f"Recompiling {proto_file} as it has changed...")
#             _compile_proto_file(proto_file)

# # Run validation on import
# _validate_proto_files()

class MessageTypes(AutoStrEnum):
    ERROR = auto()
    HIT_REPORT = auto()
    HELLO = auto()
    TEST = auto()
    HEARTBEAT = auto()


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

def create_heartbeat_msg(config: factory.gun_config) -> bytes:
    msg_to_send = Report(
        my_id=config.my_id,
        target=None,
        timestamp=get_epoch_ts(),
        img_as_str=None,
        msg_type=MessageTypes.HEARTBEAT.value,
        msg_string=""
    )

    return package_dataclass_to_bytes(msg_to_send)


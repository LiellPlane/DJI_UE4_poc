
from enum import Enum
from dataclasses import dataclass

class AutoStrEnum(str, Enum):
    """
    StrEnum where auto() returns the field name.
    See https://docs.python.org/3.9/library/enum.html#using-automatic-values
    """
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name


@dataclass
class ImagingMode():
    camera_model: str
    res_width_height: tuple[int, int]
    doc_description: str
    shared_mem_reversed: bool
    special_notes: str



@dataclass
class SharedMem_ImgTicket:
    index: int
    res: dict
    buf_size: any
    id: int

@dataclass
class CropSlicing:
    left: int
    right: int
    top: int
    lower: int

@dataclass
class AffinePoints:
    top_left_w_h: tuple[int, int]
    top_right_w_h: tuple[int, int]
    lower_right_w_h: tuple[int, int]

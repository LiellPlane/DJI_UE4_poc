import numpy as np
from libs.collections import Scambi_unit_LED_only
from libs.utils import time_it_sparse


UDP_DELIMITER:bytes = b'\xAB\xCD\xEF' # be careful changing this - can mess up delimiting if for instance | or null

def transform_scambits_for_UDP(scambis: list[Scambi_unit_LED_only])->bytes:
    """pack data for efficient delivery across network"""

    output_payload = []
    with time_it_sparse("prep scambis for sending"):
        for scambiunit in scambis:
            pos_array = np.asarray(scambiunit.physical_led_pos, dtype="uint16")
            col_array = np.asarray(tuple(reversed(scambiunit.colour)), dtype="uint8")
            output_payload.append(pos_array.tobytes())
            output_payload.append(col_array.tobytes())

    return UDP_DELIMITER.join(output_payload)


def transform_UDP_message_to_scambis(message: bytes)->list[Scambi_unit_LED_only]:
    """transform received UDP message to scambi LED information"""
    data = message.split(UDP_DELIMITER)

    scambiunits: list[Scambi_unit_LED_only] = []
    with time_it_sparse("decode remote scambis"):
        for i in range(0, len(data), 2):
            scambiunits.append(Scambi_unit_LED_only(
                colour=tuple(np.frombuffer(data[i+1], dtype="uint8")),
                physical_led_pos=np.frombuffer(data[i], dtype="uint16")))
    return scambiunits
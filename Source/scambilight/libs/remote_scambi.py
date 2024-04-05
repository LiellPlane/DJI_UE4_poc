import numpy as np
import struct
from libs.collections import Scambi_unit_LED_only

UDP_DELIMITER:bytes = b'\xAB\xCD\xEF' # be careful changing this - can mess up delimiting if for instance | or null

def transform_scambits_for_UDP(scambis: list[Scambi_unit_LED_only])->bytes:
    """pack data for efficient delivery across network"""
    output_payload = []
    for scambiunit in scambis:

        pos_array = np.asarray(scambiunit.physical_led_pos, dtype="uint16")
        col_array = np.asarray(tuple(reversed(scambiunit.colour)), dtype="uint8")
        # pos_packed_data = struct.pack(
        #     '{}H'.format(len(pos_array)//2), *pos_array)
        # col_packed_data = struct.pack(
        #     '{}B'.format(len(col_array)), *col_array)
        
        #pos_array_unpacked = np.array(struct.unpack('{}H'.format(len(pos_packed_data)//2), pos_packed_data), dtype=np.uint16)
        #col_array_unpacked = np.array(struct.unpack('{}B'.format(len(col_packed_data)), col_packed_data), dtype=np.uint8)
        output_payload.append(pos_array.tobytes())
        output_payload.append(col_array.tobytes())

    return UDP_DELIMITER.join(output_payload)


def transform_UDP_message_to_scambis(plop, message: bytes)->list[Scambi_unit_LED_only]:
    """transform received UDP message to scambi LED information"""
    data = message.split(UDP_DELIMITER)

    scambiunits: list[Scambi_unit_LED_only] = []

    for index, i in enumerate(data[::2]):
        scambiunits.append(Scambi_unit_LED_only(
            colour=np.frombuffer(data[index+1], dtype="uint8"),
            physical_led_pos=np.frombuffer(i, dtype="uint16")))

    plop=1
    # for index, i in enumerate(data[::2]):
    #     positions.append(np.frombuffer(i, dtype="uint16"))
    # for index, i in enumerate(data[1::2]):
    #     colours.append(np.frombuffer(i, dtype="uint8"))
    #for index, i in enumerate(data):
    #    colours.append(np.array(struct.unpack('{}B'.format(len(i)), i), dtype=np.uint8))
    #for index, i in enumerate(data[::2]):
    #   positions.append(np.array(struct.unpack('{}H'.format(len(i)//2), i), dtype=np.uint16))
    #posyes = [np.array(struct.unpack('{}H'.format(len(i)//2), i), dtype=np.uint16) for i in data[::2]]
    #colours = [np.array(struct.unpack('{}B'.format(len(i)), i), dtype=np.uint8) for i in data[1::2]]
    plop=1

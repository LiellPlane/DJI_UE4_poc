#RUN AS SUDO!!!!!!


import time
from rpi_ws281x import *
import argparse
import random
import socket
import atexit
import json
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass


UDP_DELIMITER:bytes = b'\xAB\xCD\xEF' # be careful changing this - can mess up delimiting if for instance | or null

@contextmanager
def time_it_sparse(comment):
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        toc: float = time.perf_counter()
        if random.randint(1,100) < 4:
            print(f"{comment}:proc time = {1000*(toc - tic):.3f}ms")
@dataclass
class Scambi_unit_LED_only(): # testing - use the collections lib otherwise
    """cheat class so we don't have to pass the whole
    object to a class which expects these members
    
    not the best place for this but had to avoid circular imports"""
    colour: any
    physical_led_pos: any

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


def transform_UDP_message_to_scambis(bytesmessage: bytes)->list[Scambi_unit_LED_only]:
    """transform received UDP message to scambi LED information"""
    data = bytesmessage.split(UDP_DELIMITER)

    scambiunits: list[Scambi_unit_LED_only] = []
    
    for i in range(0, len(data), 2):
        scambiunits.append(Scambi_unit_LED_only(
            colour=np.frombuffer(data[i+1], dtype="uint8"),
            physical_led_pos=np.frombuffer(data[i], dtype="uint16")))
    return scambiunits


class UDPMessageReceiver:
    def __init__(self, host='0.0.0.0', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        atexit.register(self.close)

    def receive_message(self, buffer_size=10000):
        try:
            data, addr = self.socket.recvfrom(buffer_size)
            return data.decode(), addr
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None, None

    def receive_bytes_message(self, buffer_size=10000):
        try:
            data, addr = self.socket.recvfrom(buffer_size)
            return data, addr
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None, None

    def close(self):
        self.socket.close()
# LED strip configuration:
SCAMBILIGHT_STRIP = 300
LED_COUNT      = SCAMBILIGHT_STRIP     # Number of LED pixels.
LED_PIN        = 18      # GPIO pin connected to the pixels (18 uses PWM!).
#LED_PIN        = 10      # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ    = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 10      # DMA channel to use for generating a signal (try 10)
LED_BRIGHTNESS = 60      # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False   # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53



def test_set_all_LEDS(strip):
    """test response time"""
    for i in range(strip.numPixels()):
        color =  Color(
            random.randint(0,1)*255,
            random.randint(0,1)*255,
            random.randint(0,1)*255)
        strip.setPixelColor(i, color)
    strip.show()


def blank_all_leds(strip):
    """test response time"""
    for i in range(strip.numPixels()):
        color =  Color(
            0,
            0,
            0)
        strip.setPixelColor(i, color)
    strip.show()

# Main program logic follows:
if __name__ == '__main__':
    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clear', action='store_true', help='clear the display on exit')
    args = parser.parse_args()

    # Create NeoPixel object with appropriate configuration.
    strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
    # Intialize the library (must be called once before other functions).
    strip.begin()
    test_set_all_LEDS(strip)
    time.sleep(1)
    blank_all_leds(strip)
    print ('Press Ctrl-C to quit.')
    if not args.clear:
        print('Use "-c" argument to clear LEDs on exit')

    while True:
        print("ready for UDP yummies")
        receiver = UDPMessageReceiver()
        while True:
            message, address = receiver.receive_bytes_message()
            with time_it_sparse("TOTAL complete process remote scambis"):
                with time_it_sparse("decode remote scambis"):
                    scambiunits = transform_UDP_message_to_scambis(message)

                #print(message
                with time_it_sparse("set leds"):
                    for index, scambiunit in enumerate(scambiunits):
                        pos = list(scambiunit.physical_led_pos)
                        col = tuple(reversed(scambiunit.colour))
                        for p in pos:
                            strip.setPixelColor(int(p),Color(*col))
                with time_it_sparse("show strip"):
                    strip.show()
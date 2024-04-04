#RUN AS SUDO!!!!!!


import time
from rpi_ws281x import *
import argparse
import random
import socket
import atexit
import json

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
            message, address = receiver.receive_message()
            message = json.loads(message)
            #print(message)
            for fart in message:
                for _pos, _rgb in fart.items():
                    color =  Color(*_rgb,)
                    strip.setPixelColor(int(_pos), color)
                if "150" in fart.keys():
                    strip.show()
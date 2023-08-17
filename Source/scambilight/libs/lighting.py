import numpy as np
from abc import ABC, abstractmethod
import cv2
import time
import math
from dataclasses import dataclass, asdict
from time import perf_counter
from contextlib import contextmanager
import random
import enum
from typing import Optional
import requests
import base64
import json
from typing import Literal
from libs.utils import (
    get_platform,
    _OS)

PLATFORM = get_platform()
if PLATFORM == _OS.RASPBERRY:
    # sorry not sorry
    import rpi_ws281x as leds


class Leds(ABC):
    
    def __init__(self, site_led_layout):
        self.led_count      = 300 # number of led pixels.
        self.led_pin        = 18      # gpio pin connected to the pixels (18 uses pwm!).
        #led_pin        = 10      # gpio pin connected to the pixels (10 uses spi /dev/spidev0.0).
        self.led_freq_hz    = 800000  # led signal frequency in hertz (usually 800khz)
        self.led_dma        = 10      # dma channel to use for generating a signal (try 10)
        self.led_brightness = 255      # set to 0 for darkest and 255 for brightest
        self.led_invert     = False   # true to invert the signal (when using npn transistor level shift)
        self.led_channel    = 0
        self.LED_layout = site_led_layout()
        #self.req_led_cols = [(0, 0, 0)] * self.led_count

    @abstractmethod
    def set_LED_values(self):
        pass

    @abstractmethod
    def execute_LEDS(self):
        pass
    
    def get_LEDpos_for_edge_range(self, scambiunit):
        """ for each scambiunit we need to map it to a physical LED
        position for the LED library
        
        we normalise each edge, and have 0 starting as moving clockwise
        and encountering the edge"""
        print("calculating pos for ", scambiunit.edge)
        led_pos_for_edge = self.LED_layout.edges[scambiunit.edge]
        pos = led_pos_for_edge
        nm = np.clip(scambiunit.position_normed, 0, 1)
        nm_start = np.clip(scambiunit.position_norm_start, 0, 1)
        nm_end = np.clip(scambiunit.position_norm_end, 0, 1)
        final_pos_mid = int(np.interp(
            nm, [0, 1], [pos.clockwise_start,pos.clockwise_end]))
        final_pos_start = int(np.interp(
            nm_start, [0, 1], [pos.clockwise_start,pos.clockwise_end]))
        final_pos_end = int(np.interp(
            nm_end, [0, 1], [pos.clockwise_start,pos.clockwise_end]))

        output = [
            i for i
            in range(
            min(final_pos_start, final_pos_end),
            max(final_pos_start, final_pos_end))]
        
        if len(output) < 1:
            print("no LED pos output")
            print(final_pos_mid, final_pos_start, final_pos_end)
            print("trying again")
            output = list(set([final_pos_start, final_pos_mid, final_pos_end]))
            if len(output) < 1:
                raise Exception("invalid - no LED position")
        print(output)
        return output
    

    
class SimLeds(Leds):

    def set_LED_values(self, scambi_units: list):
        # don't do anything
        #if len(scambi_units) > self.led_count:
        #    raise Exception("Too many leds for configured strip")
        for index, scambiunit in enumerate(scambi_units):
            pos = scambiunit.physical_led_pos
            col = tuple(reversed(scambiunit.colour))
            for p in pos:
                pass

    def execute_LEDS(self):
        pass

    #def display(self, *args, **kwargs):
    #    ImageViewer_Quick_no_resize(*args, **kwargs)




class ws281Leds(Leds):
    
    def __init__(self, site_led_layout):
        super().__init__(site_led_layout)
        # Create NeoPixel object with configuration.
        self.strip = leds.Adafruit_NeoPixel(
            self.led_count,
            self.led_pin,
            self.led_freq_hz,
            self.led_dma,
            self.led_invert,
            self.led_brightness,
            self.led_channel)
        try:
        # Intialize the library (must be called once before other functions).
            self.strip.begin()
        except RuntimeError:
            print("**************")
            print("Try running as SUDO or ROOT user")
            print("**************")
        #print("FUDGE BEING USED!!! FIX PLEASE")
        print("ws281Leds")
        time.sleep(2)
        self.test_leds()
        
    def set_LED_values(self, scambi_units: list):
        #if len(scambi_units) > self.led_count:
        #    raise Exception("Too many leds for configured strip")
        for index, scambiunit in enumerate(scambi_units):
            pos = scambiunit.physical_led_pos
            col = tuple(reversed(scambiunit.colour))
            for p in pos:
                self.strip.setPixelColor(
                    p,
                    leds.Color(*col))

    def execute_LEDS(self):
        self.strip.show()

    def display(self, *args, **kwargs):
        #  no display - pass through
        pass

    def test_leds(self):
        for i in range (0, 50):
            for i in range(self.strip.numPixels()):
                color =  leds.Color(
                    random.randint(0,1)*255,
                    random.randint(0,1)*255,
                    random.randint(0,1)*255)
                self.strip.setPixelColor(i, color)
            self.execute_LEDS()
        for i in range (0, 50):
            for i in range(self.strip.numPixels()):
                color =  leds.Color(0, 0, 0)
                self.strip.setPixelColor(i, color)
            self.execute_LEDS()



import numpy as np
from abc import ABC, abstractmethod
import cv2
import time
import math
import socket
from dataclasses import dataclass, asdict
from time import perf_counter
from contextlib import contextmanager
import random
import sys
from functools import partial
import enum
from typing import Optional
import requests
import base64
import struct
import os
import json
from typing import Literal
from libs.utils import img_height, img_width
from libs.collections import (
    LedSpacing,
    Edges,
    lens_details,
    LedsLayout,
    config_corner,
    Scambi_unit_LED_only)


from libs.utils import (
    get_platform,
    _OS,
    time_it_sparse,
    )

from factory import TimeDiffObject

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from remote_scambi import (
    transform_scambits_for_UDP,
    transform_UDP_message_to_scambis,
    UDPMessageSender,
    UDPTransmitProcessWrapper,
    UDPTrasmit_RUSTsync
)


PLATFORM = get_platform()
if PLATFORM == _OS.RASPBERRY:
    # sorry not sorry
    import rpi_ws281x as leds
else:
    class FakeLedClass():
        def Color(self, *args, **kwargs):
            pass
    leds = FakeLedClass()



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
        self.LED_layout = site_led_layout
        self.flipflopper = True
        self.set_led_func = self.do_nothing_function
        #self.req_led_cols = [(0, 0, 0)] * self.led_count

    @abstractmethod
    def set_LED_values(self):
        pass

    @abstractmethod
    def execute_LEDS(self):
        pass
    
    def display_info_colours(self, _colour):
        black = leds.Color(0,0,0)
        for i in range(self.led_count):
            color = black
            self.set_led_func(i, color)
        self.execute_LEDS()
        for i in range(1, self.led_count):
            color = leds.Color(*_colour)
            self.set_led_func(i, color)
            self.set_led_func(i-1, black)
        self.execute_LEDS()
        time.sleep(0.05)
        for i in range(self.led_count):
            self.set_led_func(i, black)
        self.execute_LEDS()

    @abstractmethod
    def display_info_bar(self, pc_done, scambi_units):
        pass
    
    def get_LEDpos_for_edge_range(self, scambiunit):
        """ for each scambiunit we need to map it to a physical LED
        position for the LED library
        
        we normalise each edge, and have 0 starting as moving clockwise
        and encountering the edge"""
        print("calculating pos for ", scambiunit.edge)
        #led_pos_for_edge = self.LED_layout.edges[scambiunit.edge]
        led_pos_for_edge = [
            v for i, v
            in self.LED_layout.edges.items()
            if scambiunit.edge.value in i][0]# should just be one result
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
        #print(output)
        return output

    def set_LED_values_alternating(self, scambi_units: list[Scambi_unit_LED_only]):
        #if len(scambi_units) > self.led_count:
        #    raise Exception("Too many leds for configured strip")
        self.flipflopper = not self.flipflopper
        total = 0
        for _, scambiunit in enumerate(scambi_units):
            pos = scambiunit.physical_led_pos
            col = tuple(reversed(scambiunit.colour))
            for p in pos:
                total += 1
                if p%2 == int(self.flipflopper):
                    self.set_led_func(
                        p,
                        leds.Color(*col))

    
    def set_LED_values(self, scambi_units: list[Scambi_unit_LED_only]):
        for _, scambiunit in enumerate(scambi_units):
            pos = scambiunit.physical_led_pos
            col = tuple(reversed(scambiunit.colour))
            for p in pos:
                self.set_led_func(
                    p,
                    leds.Color(*col))

    def do_nothing_function(self, *args, **kwargs):
        pass

    def test_leds(self):
        black = leds.Color(0,0,0)
        for i in range(self.led_count):
            color = black
            self.set_led_func(i, color)
        self.execute_LEDS()
        for i in range(1, self.led_count):
            _colour = (
                random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255))
            color = leds.Color(*_colour)
            self.set_led_func(i, color)
            self.set_led_func(i-1, black)
            self.execute_LEDS()
        for i in range(self.led_count):
            self.set_led_func(i, black)
        self.execute_LEDS()


class SimLeds(Leds):


    def execute_LEDS(self):
        pass

    def display_info_bar(self, pc_done, scambi_units):
        print("progress bar", min(1, round(pc_done, 2)))
        self.set_LED_values(scambi_units)
        self.execute_LEDS()


   
class RemoteLeds(Leds):

    def __init__(self, site_led_layout):
        super().__init__(site_led_layout)
        self.sender = UDPTransmitProcessWrapper(
            host=self.LED_layout.receiver_hostname,
            port=self.LED_layout.port
        )
        self.leds_to_send = []


    def set_LED_values_alternating(self, scambi_units: list[Scambi_unit_LED_only]):

        self.flipflopper = not self.flipflopper
        for scambiunit in scambi_units:
            if len(scambiunit.physical_led_pos) > 2:
                scambiunit.physical_led_pos = scambiunit.physical_led_pos[int(self.flipflopper)::2]

        self.leds_to_send = scambi_units# transform_scambits_for_UDP(scambi_units)


    def set_LED_values(self, scambi_units: list[Scambi_unit_LED_only]):

        self.leds_to_send = scambi_units # transform_scambits_for_UDP(scambi_units)

    def execute_LEDS(self):
        #for led_packt in self.leds_to_send:
        #with time_it_sparse("send scambis over UDP"):
        self.sender.send_scambis(self.leds_to_send)


    def display_info_bar(self, pc_done, scambi_units):
        print("progress bar", min(1, round(pc_done, 2)))
        self.set_LED_values(scambi_units)
        self.execute_LEDS()


class RemoteLedsRust(RemoteLeds):

    def __init__(self, site_led_layout):
        # Call the __init__ of the base class of RemoteLeds
        super(RemoteLeds, self).__init__(site_led_layout)
        self.sender = UDPTrasmit_RUSTsync(
            host=self.LED_layout.receiver_hostname,
            port=self.LED_layout.port
        )
        self.leds_to_send = []
    def display_info_bar(self, pc_done, scambi_units):
        print("progress bar", min(1, round(pc_done, 2)))
        self.set_LED_values(scambi_units)
        self.execute_LEDS()

    
class ws281Leds(Leds):
    
    def __init__(self, site_led_layout):
        global leds
        import rpi_ws281x as leds
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
        self.set_led_func = self.strip.setPixelColor
        time.sleep(2)
        self.test_leds()
        

    def execute_LEDS(self):
        self.strip.show()

    def display(self, *args, **kwargs):
        #  no display - pass through
        pass

    # def test_leds(self):
    #     for i in range(self.strip.numPixels()):
    #         color =  leds.Color(
    #             random.randint(0,1)*255,
    #             random.randint(0,1)*255,
    #             random.randint(0,1)*255)
    #         self.strip.setPixelColor(i, color)



    def display_info_bar(self, pc_done, scambi_units):
        print("progress bar", min(1, round(pc_done, 2)))
        self.set_LED_values(scambi_units)
        self.execute_LEDS()

    
#cache this
def perimeter_spacing(img_dim, no_of_leds):
    perimeter_pxls = img_dim
    # maybe work out closest integer here 
    remainder = perimeter_pxls % no_of_leds
    spacing = (perimeter_pxls-remainder)/no_of_leds
    return int(spacing)


def get_led_perimeter_pos(img, no_leds_vert, no_leds_horiz) -> LedSpacing:
    # imagine moving around the screen in a clockwise manner to
    # determine what is 0% and 100% of an edge
    led_spacing_vert = perimeter_spacing(img_height(img), no_leds_vert)
    led_spacing_horiz = perimeter_spacing(img_width(img), no_leds_horiz)
    x = 1
    y = 0
    _reversed = 1
    while True:
        for pos in [(0,i ) for i in range(0, img_height(img), led_spacing_vert)]:
            yield LedSpacing(
                positionxy=pos,
                edge=Edges.LEFT,
                normed_pos_along_edge_mid=_reversed - round(pos[x]/img_height(img),3),
                normed_pos_along_edge_start=_reversed - round((pos[x]-(led_spacing_vert/2))/img_height(img),3),
                normed_pos_along_edge_end=_reversed - round((pos[x]+(led_spacing_vert/2))/img_height(img),3))

        for pos in [(i, 0) for i in range(0, img_width(img), led_spacing_horiz)]:
            yield LedSpacing(
                positionxy=pos,
                edge=Edges.TOP,
                normed_pos_along_edge_mid=round(pos[y]/img_width(img),3),
                normed_pos_along_edge_start=round((pos[y]-(led_spacing_horiz/2))/img_width(img),3),
                normed_pos_along_edge_end=round((pos[y]+(led_spacing_horiz/2))/img_width(img),3))

        for pos in [(img_width(img), i) for i in range(0, img_height(img), led_spacing_vert)]:
            yield LedSpacing(
                positionxy=pos,
                edge=Edges.RIGHT,
                normed_pos_along_edge_mid=round(pos[x]/img_height(img),3),
                normed_pos_along_edge_start=round((pos[x]-(led_spacing_vert/2))/img_height(img),3),
                normed_pos_along_edge_end=round((pos[x]+(led_spacing_vert/2))/img_height(img),3))

        for pos in [(i, img_height(img)) for i in range(0, img_width(img), led_spacing_horiz)]:
            yield LedSpacing(
                positionxy=pos,
                edge=Edges.LOWER,
                normed_pos_along_edge_mid=_reversed - round(pos[y]/img_width(img),3),
                normed_pos_along_edge_start=_reversed - round((pos[y]-(led_spacing_horiz/2))/img_width(img),3),
                normed_pos_along_edge_end=_reversed - round((pos[y]+(led_spacing_horiz/2))/img_width(img),3))
        break

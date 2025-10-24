import random
import logging
import queue
import time
import copy
from typing import Any
from heapq import heappush, heappop
from functools import reduce
from multiprocessing import Process, Queue, shared_memory
from subprocess import Popen, PIPE
import os
import numpy

class UI:
    def __init__(self) -> None:
        self.win_name = "LUI"
        self.win_size = (600, 600)
        self.buttons = {}
        
    def add_button()
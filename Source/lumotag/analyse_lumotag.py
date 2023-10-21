
import decode_clothID_v2 as decode_clothID
from multiprocessing import Process, Queue, shared_memory
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class SharedMem_ImgTicket:
    index: int
    res: dict
    buf_size: any


class ImageAnalyser_shared_mem():
    """class to provide image analysis results
    using shared memory as the input"""
    def __init__(self, sharedmem_buffs: dict) -> None:
        self.sharedmem_bufs = sharedmem_buffs
        self.safe_index = None
        self.input_shared_mem_index_q = Queue(maxsize=1)
        self.analysis_output_q = Queue(maxsize=1)
        func_args = (
            self.input_shared_mem_index_q,
            self.analysis_output_q)

        process = Process(
            target=self.async_imganalysis_loop,
            args=func_args,
            daemon=True)

        process.start()

    def trigger_analysis(self, mapped_details: SharedMem_ImgTicket):
        self.input_shared_mem_index_q.put(
            mapped_details,
            block=True,
            timeout=None)

    def async_imganalysis_loop(
            self,
            input_shared_mem_index_q,
            analysis_output_q):

        while True:
            # get index of last image buffer - this will be safe
            # until two conditions are met:
            # 1: a new asynchronous image has been generated
            # 2: we have called _next_ to get it
            shared_details = input_shared_mem_index_q.get(
                block=True,
                timeout=None
                )
            # grab the image out of shared memory using the
            # information (index, resolution of image)
            # from the input queue (usually from image generator)
            print("got image")
            img_buff = np.frombuffer(
                self.sharedmem_bufs[shared_details.index].buf,
                dtype=('uint8')
                    )[0:2020*1080].reshape(2020, 1080)

            #analysis_output_q.put(img_buff, block=True, timeout=None)


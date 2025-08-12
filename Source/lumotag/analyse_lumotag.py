
import decode_clothID_v2 as decode_clothID
from factory import decode_image_id
from multiprocessing import Process, Queue
from dataclasses import dataclass
import numpy as np
from functools import reduce
from utils import time_it
from my_collections import (
    SharedMem_ImgTicket,
    CropSlicing,
    ShapeItem)
from typing import Callable
from collections import OrderedDict
import configs
import time
import cv2


def debuffer_image(img_buff: memoryview, res: tuple[int, int]) -> np.ndarray:
    # common function to rebuild image from shared memory buffer
    bytesize = res[0] * res[1]
    return np.ndarray(
        shape=res,
        dtype=np.uint8,
        buffer=img_buff[:bytesize]
    )


@dataclass
class AnalysisOutput():
    imageid: str
    Results: list[ShapeItem | None]


class ImageAnalyser_shared_mem():
    """class to provide image analysis results
    using shared memory as the input"""
    def __init__(
            self,
            sharedmem_buffs: dict,
            safe_mem_details_func: Callable[[], SharedMem_ImgTicket],
            slice_details: CropSlicing,
            img_shrink_factor: int,
            OS_friendly_name: str,
            camera_source_class_ref: type,
            config: configs.base_find_lumotag_config,
            lumotag_func: Callable[[np.ndarray, decode_clothID.WorkingData], list]) -> None:
        self.sharedmem_bufs = sharedmem_buffs
        self.lumotag_func = lumotag_func
        self.camera_source_class_ref = camera_source_class_ref
        self.OS_friendly_name = OS_friendly_name
        self.safe_mem_details_func = safe_mem_details_func
        self.input_shared_mem_index_q = Queue(maxsize=1)
        self.analysis_output_q = Queue(maxsize=1)
        self.img_crop = slice_details
        self.img_shrink_factor = img_shrink_factor
        self.debug_config = config
        self.last_analysis_time = time.perf_counter()
        # self.ImageMem: OrderedDict[str, np.ndarray] = OrderedDict()
        self.currentimg: np.ndarray | None = None
        func_args = (
            self.input_shared_mem_index_q,
            self.analysis_output_q)

        process = Process(
            target=self.async_imganalysis_loop,
            args=func_args,
            daemon=True)

        process.start()
    def check_if_timed_out(self):
        if time.perf_counter() - self.last_analysis_time > 10: # wait in seconds
            return True
        return False
    def get_analysis_time_ms(self):
        return (time.perf_counter() - self.last_analysis_time) * 1000
    def trigger_analysis(self):
        
        #print("putting record for analyis", mapped_details)
        # if self.input_shared_mem_index_q.empty():# skip if procesing something already
        #     self.last_analysis_time = time.perf_counter() # reset time out
        #     self.input_shared_mem_index_q.put(
        #         self.safe_mem_details_func(),
        #         block=True,
        #         timeout=None)
        #print("putting record for analyis", mapped_details)
        if not self.input_shared_mem_index_q.full():# skip if procesing something already
            self.last_analysis_time = time.perf_counter() # reset time out
            self.input_shared_mem_index_q.put(
                self.safe_mem_details_func(),
                block=True,
                timeout=None)


    def async_imganalysis_loop(
            self,
            input_shared_mem_index_q,
            analysis_output_q):

        workingdata = decode_clothID.WorkingData(
            OS_friendly_name=self.OS_friendly_name,
            debugdetails=self.debug_config)

        while True:


            # get index of last image buffer - this will be safe
            # until two conditions are met:
            # 1: a new asynchronous image has been generated
            # 2: we have called _next_ to get it
            #print("ANALOL waiting in analysis loop for record")
            shared_details = input_shared_mem_index_q.get(
                block=True,
                timeout=None
                )
            # print(f"ANALOL received analysis details {self.OS_friendly_name}")
            #print("ANALOL received analysis details", shared_details)
            with time_it(f"analyse lumotag{self.OS_friendly_name}: total", workingdata.debug_details.PRINT_DEBUG):
                
                img_buff = debuffer_image(
                    self.sharedmem_bufs[shared_details.index].buf,
                    shared_details.res
                    )
                embedded_id = decode_image_id(img_buff)
                # add any cropping
                if self.img_crop is not None:
                    img_buff = img_buff[
                            self.img_crop.top:self.img_crop.lower,
                            self.img_crop.left:self.img_crop.right]

                if self.img_shrink_factor is not None:
                    # was using step sampling before - this is way slower than resize
                    target_size = (
                        int(img_buff.shape[1] // self.img_shrink_factor),
                        int(img_buff.shape[0] // self.img_shrink_factor)
                    )
                    img_buff = cv2.resize(img_buff, target_size, interpolation=cv2.INTER_NEAREST)  # type: ignore

                if img_buff.flags.owndata:
                    self.currentimg = img_buff
                else:
                    self.currentimg = img_buff.copy()

           # with time_it("analyse lumotag: find lumotag"):
                try:
                    contour_data: list[ShapeItem | None] = self.lumotag_func(
                        self.currentimg, workingdata)
                except Exception as e:
                    print(f"Error finding lumotag: {e}")
                    # this will explode but at least we get something back
                    analysis_output_q.put(e, block=True, timeout=None)
            #with time_it("analyse lumotag: prepare graphics"):
                for contour in contour_data:
                    if self.img_shrink_factor is not None:
                        contour.add_resize_offset(self.img_shrink_factor)
                    if self.img_crop is not None:
                        contour.add_offset_for_graphics([self.img_crop.left,self.img_crop.top])

                # correct contour data here? not sure if correct place
                # if len(contour_data) == 0:
                #     # no results - not interesting to us (yet)
                #     del self.ImageMem[embedded_id]
                
                # if len(self.ImageMem)> 100:
                #     _, _ = self.ImageMem.popitem(last=False)
            #print("ANALOL waiting to put response")
            # import time
            # import random
            # randimtew = random.randint(10,50)
            # time.sleep(randimtew/1000)

            analysis_output_q.put(AnalysisOutput(embedded_id, contour_data), block=True, timeout=None)

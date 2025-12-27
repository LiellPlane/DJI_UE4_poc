
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
import queue


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
        self.analysis_output_q = Queue(maxsize=10)  # main loop drains with while-not-empty
        self.img_crop = slice_details
        self.img_shrink_factor = img_shrink_factor
        self.debug_config = config
        self.current_analysis_time: float | None = None 
        self.last_analysis_time: float = 0.0
        # self.ImageMem: OrderedDict[str, np.ndarray] = OrderedDict()
        
        func_args = (
            self.input_shared_mem_index_q,
            self.analysis_output_q)

        process = Process(
            target=self.async_imganalysis_loop,
            args=func_args,
            daemon=True)

        process.start()

    def set_idle(self):
        # you can probably do this better looking at the queue
        self.current_analysis_time = None
    
    def check_if_timed_out(self):
        # you can probably do this better looking at the queue
        if self.current_analysis_time is None:
            # we can't time out if we have never been triggered 
            return False
        if time.perf_counter() - self.current_analysis_time > 10: # wait in seconds
            return True
        return False
    def get_analysis_time_ms(self):
        return self.last_analysis_time * 1000
    
    def get_analysis_result(self, block=False, timeout=0, drain=False) -> AnalysisOutput | None:
        """Get analysis result from queue and reset timeout timer.
        
        Returns:
            AnalysisOutput | Exception | None: Result from queue, or None if empty
            if its empty its either just started or not had time to process the image - so is not a 
            negative response
        """
        # deactivate timer - we arent processing anymore
        if drain is True:
            try:
                result = self.analysis_output_q.get(block=block, timeout=timeout)
                if self.current_analysis_time is not None:
                    self.last_analysis_time = time.perf_counter() - self.current_analysis_time
                self.current_analysis_time = None
                return result
            except queue.Empty:
                return None
        else:
            result = None
            try:
                # drain queue - get freshest analysis
                while True:
                    result = (
                        self.analysis_output_q.get(block=block, timeout=timeout)
                    )
            except queue.Empty:
                if result is not None:
                    if self.current_analysis_time is not None:
                        self.last_analysis_time = time.perf_counter() - self.current_analysis_time
                    self.current_analysis_time = None
                return result
    
    def trigger_analysis(self):
        """Trigger analysis on current frame. Replaces any stale queued frame."""
        # Always try to discard stale ticket (empty() is unreliable in multiprocessing)
        try:
            self.input_shared_mem_index_q.get_nowait()
        except queue.Empty:
            pass
        
        # Queue fresh ticket (should always succeed after clearing, but handle edge case)
        self.current_analysis_time = time.perf_counter()
        try:
            self.input_shared_mem_index_q.put_nowait(self.safe_mem_details_func())
        except queue.Full:
            pass  # Rare race: just skip this frame

        # # ORIGINAL: skip if queue full (drops new frame if busy)
        # if not self.input_shared_mem_index_q.full():
        #     self.current_analysis_time = time.perf_counter()
        #     self.input_shared_mem_index_q.put(
        #         self.safe_mem_details_func(),
        #         block=True,
        #         timeout=None)


    def async_imganalysis_loop(
            self,
            input_shared_mem_index_q,
            analysis_output_q):
        try:
            workingdata = decode_clothID.WorkingData(
                OS_friendly_name=self.OS_friendly_name,
                debugdetails=self.debug_config)
            currentimg: np.ndarray | None = None
            last_frame_id = -1
            last_result: AnalysisOutput | None = None
            
            while True:
                # Wait for ticket from trigger_analysis()
                shared_details = input_shared_mem_index_q.get(block=True, timeout=None)
                
                # Return cached result if same frame (prevents flickering)
                if shared_details.frame_id == last_frame_id and last_result is not None:
                    analysis_output_q.put_nowait(last_result)
                    continue
                
                last_frame_id = shared_details.frame_id
                
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
                        currentimg = img_buff
                    else:
                        currentimg = img_buff.copy()

                    contour_data: list[ShapeItem | None] = self.lumotag_func(
                        currentimg, workingdata)

                    for contour in contour_data:
                        if self.img_shrink_factor is not None:
                            contour.add_resize_offset(self.img_shrink_factor)
                        if self.img_crop is not None:
                            contour.add_offset_for_graphics([self.img_crop.left,self.img_crop.top])

                #print("ANALOL waiting to put response")
                last_result = AnalysisOutput(embedded_id, contour_data)
                analysis_output_q.put_nowait(last_result)
                
        except Exception as e:
            # Put exception on queue so main process can detect failure
            analysis_output_q.put(e, block=True, timeout=None)
            raise  # Re-raise to kill subprocess - main will see via queue

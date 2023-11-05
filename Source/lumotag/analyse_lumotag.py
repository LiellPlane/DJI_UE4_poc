
import decode_clothID_v2 as decode_clothID
from multiprocessing import Process, Queue, shared_memory
from dataclasses import dataclass
import numpy as np
from functools import reduce
from utils import time_it
import time
import random
from my_collections import SharedMem_ImgTicket, CropSlicing

class ImageAnalyser_shared_mem():
    """class to provide image analysis results
    using shared memory as the input"""
    def __init__(
            self,
            sharedmem_buffs: dict,
            slice_details: CropSlicing) -> None:
        self.sharedmem_bufs = sharedmem_buffs
        self.safe_index = None
        self.input_shared_mem_index_q = Queue(maxsize=1)
        self.analysis_output_q = Queue(maxsize=1)
        self.img_crop = slice_details
        func_args = (
            self.input_shared_mem_index_q,
            self.analysis_output_q)

        process = Process(
            target=self.async_imganalysis_loop,
            args=func_args,
            daemon=True)

        process.start()

    def trigger_analysis(self, mapped_details: SharedMem_ImgTicket):
        print("putting record for analyis", mapped_details)
        self.input_shared_mem_index_q.put(
            mapped_details,
            block=True,
            timeout=None)

    def async_imganalysis_loop(
            self,
            input_shared_mem_index_q,
            analysis_output_q):
        workingdata = decode_clothID.WorkingData(debug=False)
        while True:
            # get index of last image buffer - this will be safe
            # until two conditions are met:
            # 1: a new asynchronous image has been generated
            # 2: we have called _next_ to get it
            print("ANALOL waiting in analysis loop for record")
            shared_details = input_shared_mem_index_q.get(
                block=True,
                timeout=None
                )
            print("ANALOL received analysis details", shared_details)
            with time_it("analyse lumotag"):
                # shared memory is in chunks of 4096 - so have to slice it
                bytesize = reduce((lambda x, y: x * y), shared_details.res)
                # grab the image out of shared memory using the
                # information (index, resolution of image)
                # from the input queue (usually from image generator)
                img_buff = np.frombuffer(
                    self.sharedmem_bufs[shared_details.index].buf,
                    dtype=('uint8')
                        )[0:bytesize].reshape(shared_details.res)
                
                # add any cropping
                img_buff = img_buff[
                        self.img_crop.left:self.img_crop.right,
                        self.img_crop.top:self.img_crop.lower]
                print(f"analysing img_buff shape {img_buff.shape}")
                contour_data = decode_clothID.find_lumotag(
                    img_buff, workingdata)
                
                for contour in contour_data:
                    # TODO check xy orientation correct
                    contour.add_offset_for_graphics([self.img_crop.top,self.img_crop.left])
                    
                # correct contour data here? not sure if correct place
                
            print("ANALOL waiting to put response")

            analysis_output_q.put((contour_data, self.img_crop), block=True, timeout=None)

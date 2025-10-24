def multiprocess_set_colours():
    # This doesnt work with raspberry pi w 2 - maybe if getting one
    # with more cores
    # some ROI are bigger than others
    random.shuffle(scambi_units)
    scambis_per_core = int(len(scambi_units)/cores_for_col_dect)
    # chop up list of scambiunits for parallel processing
    proc_scambis = [
        async_cam_lib.Process_Scambiunits(
            scambiunits=scambi_units[i:i+scambis_per_core],
            subsample_cutoff=subsample_cut,
            flipflop=False)
        for i
        in range(0,len(scambi_units), scambis_per_core)]

    while True:
        with time_it("main loop"):
            with time_it("get img"):
                prev = next(cam)
            with time_it("load imgs"):
                for scamproc in proc_scambis:
                    scamproc.in_queue.put(prev, block=True)
            scambis_cols = {}
            with time_it("wait for colors"):
                for scamproc in proc_scambis:
                    scambis_cols.update(scamproc.done_queue.get(block=True, timeout=None))
            with time_it("rebuild scambi colours"):
                for unit in scambi_units:
                    unit.colour = scambis_cols[unit.id]
            if PLATFORM == _OS.WINDOWS:
                for index, unit in enumerate(scambi_units):
                    unit.draw_warped_boundingbox(prev)
                    prev = unit.draw_warped_led_pos(
                        prev,
                        unit.colour,
                        offset=(0, 0),
                        size=10)
            with time_it("set colours"):
                led_subsystem.set_LED_values(scambi_units)
                led_subsystem.execute_LEDS()
            if PLATFORM == _OS.WINDOWS:
                ImageViewer_Quick_no_resize(prev,0,True,False)


class Find_Screen():
    def __init__(self) -> None:
        self.motion_img = None
        self.firstFrame = None
        self.backSub = cv2.createBackgroundSubtractorKNN(
            history=5000,
            dist2Threshold=4000.0,
            detectShadows=False)
        self.kernel = np.ones((50,50),np.float32)/25
    def input_image(self, img):
        img_resize = cv2.resize(img,(640, 480))
        fgMask = self.backSub.apply(img_resize)
        if self.motion_img is None:
            self.motion_img = np.zeros_like(fgMask)
            self.motion_img = self.motion_img.astype(int)
        
        dst = cv2.filter2D(fgMask,-1,self.kernel)
        self.motion_img = np.add(self.motion_img, dst)
        
        #if self.motion_img.max() > 1000:
        self.motion_img = np.subtract(self.motion_img, 50)
        self.motion_img = np.clip(self.motion_img, 0, 2**32)
        #fgMask = cv2.fastNlMeansDenoising(fgMask)
        #fgMask = cv2.fastNlMeansDenoising(fgMask,None,10,10,7,21)
        print(self.motion_img.max())
        output = self.motion_img/(self.motion_img.max()/254)
        output = self.motion_img.astype("uint8")
        ImageViewer_Quick_no_resize(img,0,False,False)

def test_find_screen():
    input_vid = r"C:\Working\nonwork\SCAMBILIGHT\test_raspberrypi_v2.mp4"
    screen_finder = Find_Screen()
    cap = cv2.VideoCapture(input_vid)
    while True:
        suc, frame = cap.read()
        screen_finder.input_image(frame)


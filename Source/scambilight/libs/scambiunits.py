import numpy as np
import struct
import cv2
import math
import uuid
from dataclasses import dataclass
import random
import libs.fisheye_lib as fisheye_lib
from typing import Optional
import libs.async_cam_lib as async_cam_lib
from libs.utils import  convert_pts_to_convex_hull
from libs.collections import lens_details, config_regions, Edges
from libs.lighting import get_led_perimeter_pos
from collections import deque


@dataclass
class ScambiInit():
    led_positionxy: tuple
    sample_area_left: int
    sample_area_right: int
    sample_area_top: int
    sample_area_lower: int
    inverse_warp_m: any
    img_shape: Optional[any]
    img_circle: Optional[int]
    edge: Edges
    position_normed: float
    position_norm_start: float
    position_norm_end: float
    id: int

@dataclass
class ScambiWarp():
    roi = []
    warped_led_pos: any = None
    warped_bounding_rect: any = None
    bb_left: int = None
    bb_right: int = None
    bb_top: int = None
    bb_lower: int = None
    sample_area_lerp_contour: any = None
    convex_hulls_lerp_contour: any = None
    @property
    def bb_height(self):
        # error here can mean not initialised
        return abs(self.bb_lower-self.bb_top)
    @property
    def bb_width(self):
         # error here can mean not initialised
        return abs(self.bb_right-self.bb_left)



class Scambi_unit():
    def __init__(self,
                 init_object: ScambiInit):

        """supply led and roi positions on a flat representation of
        the screen
        
        inverse_warp will map these positions with the inverse homography
        if the screen isn't presented flat
        
        img_shape and img_circle are optional parameters that will 
        additionally warp the already perspective warped rois
        to a fisheye of img_circle
        
        needs to be manually initialised by calling .initialise"""
        self.initobj = init_object
        self.perpwarp = ScambiWarp()
        self.fishwarp = ScambiWarp()
        self.ID = str(uuid.uuid4())
        #self.roi = []
        #self.warped_led_pos = None
        #self.warped_bounding_rect = None
        #self.bb_left = None
        #self.bb_right = None
        #self.bb_top = None
        #self.bb_lower = None
        self.last_dom_col_dif = None
        self.colour = (
            0,
            0,
            255)
        #self.sample_area_lerp_contour = None
        #self.convex_hulls_lerp_contour = None
        self.physical_led_pos = None
        self.colour_history = deque(maxlen=5)
        
    def initialise(self):
        self.warp_rois_homograpy()
        self.fisheye_rois()
        #print(f"scambiunit {self.initobj.id} initialised")

    def assign_physical_LED_pos(self, pos: int):
        self.physical_led_pos = pos

    @property
    def edge(self):
        return self.initobj.edge

    @property
    def position_normed(self):
        return self.initobj.position_normed
    
    @property
    def position_norm_start(self):
        return self.initobj.position_norm_start

    @property
    def position_norm_end(self):
        return self.initobj.position_norm_end

    def lerp_sample_area(self):
        """Create interpolation between corners so we can
        warp in a non-linear fashion such as fish eye"""
        contours = []
        # along top left to right
        lin = np.linspace(self.initobj.sample_area_left, self.initobj.sample_area_right, 10)
        for x in lin:
            contours.append((x, self.initobj.sample_area_top))
        # down right side
        lin = np.linspace(self.initobj.sample_area_top, self.initobj.sample_area_lower, 10)
        for y in lin:
            contours.append((self.initobj.sample_area_right, y))
        # along bottom right to left
        lin = np.linspace(self.initobj.sample_area_right, self.initobj.sample_area_left, 10)
        for x in lin:
            contours.append((x, self.initobj.sample_area_lower))
        # up left side
        lin = np.linspace(self.initobj.sample_area_lower, self.initobj.sample_area_top, 10)
        for y in lin:
            contours.append((self.initobj.sample_area_left, y))
        cont_ints = [(int(i[0]), int(i[1])) for i in contours]
        return cont_ints

    def warp_rois_homograpy(self):
        """warp rois if we just have a perspective warp"""
        # warp sample colour region
        temp_roi = []
        temp_roi.append(np.asarray([self.initobj.sample_area_left, self.initobj.sample_area_top, 1]))
        temp_roi.append(np.asarray([self.initobj.sample_area_left, self.initobj.sample_area_lower, 1]))
        temp_roi.append(np.asarray([self.initobj.sample_area_right, self.initobj.sample_area_lower, 1]))
        temp_roi.append(np.asarray([self.initobj.sample_area_right, self.initobj.sample_area_top, 1]))
        self.perpwarp.roi = []
        for pt in temp_roi:
            homog_coords = np.matmul(self.initobj.inverse_warp_m, pt)
            new_pt = list((np.floor(homog_coords[0:2]/homog_coords[-1])).astype(int))
            self.perpwarp.roi.append(new_pt)
        self.perpwarp.roi = np.asarray(self.perpwarp.roi).astype(np.int32)
        self.perpwarp.roi = self.perpwarp.roi.reshape((-1, 1, 2))
        # if self.perpwarp.roi.shape[0] != 4:
        #     print("something funky happening", self.perpwarp.roi)
        #     raise Exception("perpwarp.roi bad shape!! should only be 4 elements")
        # warp expected LED region
        temp_led_pos = np.asarray(self.initobj.led_positionxy + (1,))
        homog_coords = np.matmul(self.initobj.inverse_warp_m, temp_led_pos)
        self.perpwarp.warped_led_pos = tuple((np.floor(homog_coords[0:2]/homog_coords[-1])).astype(int))

        #get bounding box for ROI
        self.perpwarp.warped_bounding_rect = get_bounding_box(self.perpwarp.roi)
        left, top, w, h = self.perpwarp.warped_bounding_rect
        right = left + w
        lower = top + h
        self.perpwarp.bb_left = left
        self.perpwarp.bb_right = right
        self.perpwarp.bb_top = top
        self.perpwarp.bb_lower = lower

        #warp the lerped rois (rois with points between corners for non-linear warping)
        warped_lerp_roi = []
        lerp_area = self.lerp_sample_area()
        for lerp_pt in lerp_area:
            temp_lerp_pos = np.asarray(lerp_pt + (1,))
            homog_coords = np.matmul(self.initobj.inverse_warp_m, temp_lerp_pos)
            warped_lerp_pt = tuple((np.floor(homog_coords[0:2]/homog_coords[-1])).astype(int))
            warped_lerp_roi.append(warped_lerp_pt)
        self.perpwarp.sample_area_lerp_contour = warped_lerp_roi
        
    def fisheye_rois(self):
        """warp the rois to fisheye - expected to have
        warped perspective first"""
        
        fisheyeser = fisheye_lib.fisheye_tool(
            img_width_height=tuple(reversed(self.initobj.img_shape[0:2])),
            image_circle_size=self.initobj.img_circle)
        
        temp_pts = []

        for pt in self.perpwarp.sample_area_lerp_contour:
            temp_pts.append(
                fisheyeser.brute_force_find_fisheye_pt(pt))

        self.fishwarp.sample_area_lerp_contour = temp_pts
        self.fishwarp.convex_hulls_lerp_contour = convert_pts_to_convex_hull(
            self.fishwarp.sample_area_lerp_contour)

        self.fishwarp.warped_led_pos = fisheyeser.brute_force_find_fisheye_pt(self.perpwarp.warped_led_pos)
        
        #recalculate
        self.fishwarp.roi = np.asarray([np.asarray(fisheyeser.brute_force_find_fisheye_pt(list(pt[0]))) for pt in self.perpwarp.roi]).reshape(-1, 1, 2)
        self.fishwarp.warped_bounding_rect =get_bounding_box(self.fishwarp.roi)
        left, top, w, h = self.fishwarp.warped_bounding_rect
        right = left + w
        lower = top + h
        self.fishwarp.bb_left = left
        self.fishwarp.bb_right = right
        self.fishwarp.bb_top = top
        self.fishwarp.bb_lower = lower
    
    def draw_lerp_contour(self, img):

        cv2.drawContours(
            image=img,
            contours=[self.fishwarp.convex_hulls_lerp_contour],
            contourIdx=-1,
            color=(100,200,250),
            thickness=1,
            lineType=cv2.LINE_AA)
        
        return img

    def draw_warped_roi(self, img):
        #for pt in self.roi:
        #    img[pt[1], pt[0], :] = (255, 255, 255)
        img = cv2.polylines(img,
                            [self.fishwarp.roi],
                      isClosed=True, color=(random.randint(30,255),random.randint(30,255),random.randint(30,255)),
                      thickness=1)
        
        #cv2.drawContours(img, [br], -1, (0,0,255), 1)
        #cv2.circle(img,(br[2], br[3]),2,(255,255,255),-1)
        #cv2.rectangle(img,(br[0], br[1]),(br[2], br[3]),(0,255,0),3)
        return img
    
    def draw_warped_boundingbox(self,img):
        x, y, w, h = self.fishwarp.warped_bounding_rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 1)

        return img

    def draw_led_pos(self, img, colour=None, offset=None, size=16):

        pos = self.initobj.led_positionxy
        if offset is not None:
            pos = (pos[0] + offset[0], pos[1] + offset[1])
        if colour is None:
            cv2.circle(img,pos,size,(255,0,100),-1)
        else:
            cv2.circle(img,pos,size,colour,-1)

    def draw_warped_led_pos(self, img, colour=None, offset=None, size=16):
    
        pos = self.fishwarp.warped_led_pos
        if offset is not None:
            pos = (pos[0] + offset[0], pos[1] + offset[1])
        if colour is None:
            cv2.circle(img,pos,size,(255,0,100),-1)
        else:
            cv2.circle(img,pos,size,colour,-1)

        return img

    def draw_rectangle(self, img):
        draw_rectangle(
            self.initobj.sample_area_left,
            self.initobj.sample_area_right,
            self.initobj.sample_area_top,
            self.initobj.sample_area_lower, img)

    def get_dominant_colour_perspective(self, img):
        pass

    def get_mean_colour(self, img, subsample):
        sample_area = img[
            self.fishwarp.bb_top:self.fishwarp.bb_lower,
            self.fishwarp.bb_left:self.fishwarp.bb_right,
            :]
        
        self.colour = tuple(
            [int(i) for i in sample_area.mean(axis=0).mean(axis=0)])
        return self.colour

    def set_dom_colour_with_auto_subsample(self, img, cut_off):
        try:
            min_edge = min([self.fishwarp.bb_height, self.fishwarp.bb_width])
            subsampling = math.ceil(min_edge/cut_off)
            if subsampling < 1:
                raise Exception("problem with subsampling", self.fishwarp.bb_height, self.fishwarp.bb_width)
            col = self.get_dominant_colour_flat(img, subsampling)
            #col = self.average_colour(col, self.colour)
            #col = self.lerp_color(col)
            self.colour_history.append(tuple(int(i) for i in col))

            # lets try and mitigate update rate of TV creating flickering
            # if the next colour is darker - then average it out
            # otherwise - use current colour
            
            isless = np.all(np.asarray(self.colour_history[-1]) < np.asarray(self.colour_history[-2]))
            if isless is True:
                # current colour is bright - maybe its catching the TV refresh cycle. So lets try and
                # smooth it out with the downside of slowing response
                average = (np.asarray(self.colour_history[-1]) + np.asarray(self.colour_history[-2])) / 2
                self.colour = tuple(int(i) for i in average)
            else:
                self.colour = tuple(int(i) for i in col)
            #return self.lerp_color(col)
        except:
            return self.colour

    def lerp_color(self, input_color: tuple[float,float,float]):
        # lerp colors using f(t) = t.t.t
        return tuple([((x/255)**3)*255 for x in input_color])

    def get_dominant_colour_flat(self, img, subsample):
        """keep subsample between 1 (unity) and 4 usually"""
        sample_area = img[
            self.fishwarp.bb_top:self.fishwarp.bb_lower:subsample,
            self.fishwarp.bb_left:self.fishwarp.bb_right:subsample,
            :]
        if random.randint(0,5000) < 2:
            print(f"{self.fishwarp.bb_lower-self.fishwarp.bb_top} * {self.fishwarp.bb_right-self.fishwarp.bb_left}")
        data = np.reshape(sample_area, (-1,3))
        data = np.float32(data)
        epsilon = 1.0
        max_iter = 10
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,max_iter,epsilon)
        flags = cv2.KMEANS_PP_CENTERS
        _, _, centers = cv2.kmeans(
            data=data,
            K=1,
            bestLabels=None,
            criteria=criteria,
            attempts=5,
            flags=flags)
        dom_col = centers[0].astype(np.int32)
        dom_col = [int(i) for i in dom_col]

        col = tuple(dom_col)

        # plop = (
        #     int((col[0] + self.colour[0])/2),
        #     int((col[1] + self.colour[1])/2),
        #     int((col[2] + self.colour[2])/2))
        #self.colour = plop

        return col

    @staticmethod
    def average_colour(col1, col2):
        plop = (
            int((col1[0] + col2[0])/2),
            int((col1[1] + col2[1])/2),
            int((col1[2] + col2[2])/2))
        return plop
    
    def is_sample_area_smaller(self, cut_off: int):
        tp = self.fishwarp
        if (abs(tp.bb_right - tp.bb_left)) < cut_off or (abs(tp.bb_lower-tp.bb_top)) < cut_off:
            return True
        return False

def get_bounding_box(x):
    return (
        min(x[:,0][:,0]),
        min(x[:,0][:,1]),
        abs(max(x[:,0][:,0]) - min(x[:,0][:,0])),
        abs(max(x[:,0][:,1]) - min(x[:,0][:,1]))
    )

def get_dominant_colour_flat_vectorize(img, list_of_scambiunits):
 
 
    return 
    for unit in list_of_scambiunits:

        sample_area = img[
            unit.bb_top:unit.bb_lower,
            unit.bb_left:unit.bb_right,
            :]
        plop=1
    if random.randint(0,500) < 2:
        print(f"{self.bb_lower-self.bb_top} * {self.bb_right-self.bb_left}")
    data = np.reshape(sample_area, (-1,3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(
        data=data,
        K=1,
        bestLabels=None,
        criteria=criteria,
        attempts=10,
        flags=flags)
    dom_col = centers[0].astype(np.int32)
    dom_col = [int(i) for i in dom_col]
    return tuple([int(i) for i in centers[0]])
    return (0, 0, 0)


class HomographyTool():
    
    def __init__(
            self,
            img_height_,
            img_width_,
            corners,
            target_corners):
        """ calibrate_pts = np.asarray([
        [419, 204],
        [1707, 327],
        [1761, 1038],
        [91, 786]], dtype="float32")"""
        self._img_height = img_height_ 
        self._img_width = img_width_ 
        self._corners = np.asarray(corners,  dtype="float32") 
        self._target_corners = np.asarray(target_corners,  dtype="float32") 
        self.trans_matrix = cv2.getPerspectiveTransform(
                    self._corners,
                    self._target_corners)
        self.inverse_trans_matrix = np.linalg.inv(
            self.trans_matrix
            )

    def warp_img(self, img):
        return(cv2.warpPerspective(
            img,
            self.trans_matrix,
            list(img.shape[0:2]).reverse()
            ))


def generate_scambis(
        img_shape: tuple,
        regions: config_regions,
        optical_details: lens_details,
        homography_tool: HomographyTool,
        led_subsystem: any,
        initialise: True,
        init_cores: Optional[int],
        progress_bar_func: Optional[callable]):
    scambi_units = []
    led_positions = get_led_perimeter_pos(img_shape, regions.no_leds_vert, regions.no_leds_horiz)
    #print("got get_led_perimeter_pos")
    for index,  led in enumerate(led_positions):
        #print(f"calculating scambiunit {index}/{regions.no_leds_vert+regions.no_leds_vert+regions.no_leds_horiz+regions.no_leds_horiz}")
        centre_ = tuple((np.asarray(led.positionxy)).astype(int))
        #cv2.circle(prev,plop,16,(255,0,100),-1)
        mid_screen = (np.array(tuple(reversed(img_shape[:2])))/2).astype(int)[:2]
        vec_to_midscreen = mid_screen-np.asarray(centre_)
        #cv2.circle(prev,tuple(mid_screen),16,(255,0,100),-1)
        #regions.move_in_vert = -0.4
        #regions.move_in_horiz = 0.6
        if led.edge  not in [Edges.TOP, Edges.LOWER, Edges.LEFT, Edges.RIGHT]:
            raise Exception("edge name " + led.edge + "not valid")


        # we have the rectangular positions (ideal) before warping 
        # this adjustment allows us to move the positions, but
        # if we use both X and Y of vector then the positions shrink together, 
        # we just want them to keep their length but change vertical/horiz positions
        # so remove the dimension we want to keep static
        if led.edge  in [Edges.TOP, Edges.LOWER]:
            vec = vec_to_midscreen * regions.move_in_vert
            vec[0] = 0
            new_pos =  tuple((np.asarray(centre_) + vec).astype(int))
            #new_pos = tuple((np.asarray(centre_) + (vec_to_midscreen * regions.move_in_vert)).astype(int))

            #new_pos = tuple([new_pos[0], np.asarray(centre_)[1].astype(int)])
        if led.edge  in [Edges.LEFT, Edges.RIGHT]:
            vec = vec_to_midscreen * regions.move_in_horiz
            vec[1] = 0
            new_pos =  tuple((np.asarray(centre_) + vec).astype(int))
            #new_pos = tuple((np.asarray(centre_) + (vec_to_midscreen * regions.move_in_horiz)).astype(int))
            #new_pos = tuple([np.asarray(centre_)[0].astype(int), new_pos[1]])
        
        left, right, top, lower = create_rectangle_from_centrepoint(new_pos, edge=regions.sample_area_edge)
        init = ScambiInit(led_positionxy=centre_,
            sample_area_left=left,
            sample_area_right=right,
            sample_area_top=top,
            sample_area_lower=lower,
            inverse_warp_m=homography_tool.inverse_trans_matrix,
            img_shape=img_shape,
            img_circle=optical_details.fish_eye_circle,
            edge=led.edge,
            position_normed=led.normed_pos_along_edge_mid,
            position_norm_start=led.normed_pos_along_edge_start,
            position_norm_end=led.normed_pos_along_edge_end,
            id=index)
        scambi_units.append(Scambi_unit(init)
        )

    for scambi in scambi_units:
        scambi.assign_physical_LED_pos(led_subsystem.get_LEDpos_for_edge_range(scambi))

    if initialise is True:
        # initialise scambis - this can take a while
        random.shuffle(scambi_units)
        scambis_per_core = int(len(scambi_units)/init_cores)
        # chop up list of scambiunits for parallel processing
        proc_scambis = [
            async_cam_lib.Process_Scambiunits(
                scambiunits=scambi_units[i:i+scambis_per_core],
                subsample_cutoff=regions.subsample_cut,
                flipflop=False)
            for i
            in range(0,len(scambi_units), scambis_per_core)]

        # get initialised scambiunits from parallel processing
        initialised_scambi_units = []
        finished_procs = []
        total_leds = (regions.no_leds_vert + regions.no_leds_horiz) * 2
        while len(initialised_scambi_units) != len(scambi_units):
            if len(finished_procs) == len(proc_scambis):
                raise Exception("procs finished but not correct # of scambis")
            for index, scamproc in enumerate(proc_scambis):
                if index in finished_procs:
                    continue
                data = scamproc.initialised_scambis_q.get(
                    block=True,
                    timeout=None)
                if isinstance(data, Scambi_unit):
                    initialised_scambi_units.append(data)
                    progress_bar_func(
                        len(initialised_scambi_units)/(total_leds+1), initialised_scambi_units)
                elif isinstance(data, async_cam_lib.FinishedProcess):
                    finished_procs.append(index)
                else:
                    raise Exception("unexpected item in processing area")
            
        # flatten nested list
        #scambi_units = [item for sublist in scambi_units for item in sublist]
        # flatten nested list
        #scambi_units = [item for sublist in scambi_units for item in sublist]
        return initialised_scambi_units
    return scambi_units
    


def create_rectangle_from_centrepoint(centrepoint, edge):
    half_edge = int(edge/2)
    posx =centrepoint[0]
    posy = centrepoint[1]
    left = posx - half_edge
    right = posx + half_edge
    top = posy - half_edge
    lower = posy + half_edge
    return left, right, top, lower


def draw_rectangle(left, right, top, down, img):
    rec = cv2.rectangle(img,
                  (left, top),
                  (right, down),
                  (0, 100, 255),
                  8)
    return rec


# class AnalyseRefreshRate():
#     def __init__(self, scambiunits: list[Scambi_unit]) -> None:
#         """Analyse and diagnose errors with scambiunit sampling, such
#         as refresh rate of TV, by grabbing a timeslice of samples
        
#         Will sample from one randomly selected sambiunit for N seconds
#         then move to next one"""

#         self.samples = 2500
#         self.sampledict = dict()
#         self.current_sample = None
#         self.scambunits = scambiunits
#         self._samp_counter = 0
#         self._samp_scambiunit = 0

#     def sample(self):
#         """grab a sample of colour from scambiunit
#         this function will handle organising sampling schedule, 
#         therefore keep calling it every event loop"""
#         if self._samp_scambiunit ==len(self.scambunits)-1:
#             return None
#         scambi_to_sample = self.scambunits[self._samp_scambiunit]
#         # initialise np array
#         if self._samp_scambiunit not in self.sampledict:
#             self.sampledict[self._samp_scambiunit] = {}
#             self.sampledict[self._samp_scambiunit]["timestamp"] = []
#             self.sampledict[self._samp_scambiunit]["samples"] = np.zeros((self.samples + 1, 3), dtype=int)
#             self._samp_counter = 0
#         # add sample
#         self.sampledict[self._samp_scambiunit]["samples"][self._samp_counter]  = np.array(scambi_to_sample.colour)
#         self.sampledict[self._samp_scambiunit]["timestamp"].append(time.time())

#         self._samp_counter += 1

#         if self._samp_counter > self.samples:
#             # scambiunit samping done - convert np array to list for jsonability
#             self.sampledict[self._samp_scambiunit]["samples"] = self.sampledict[self._samp_scambiunit]["samples"].tolist()
#             output = json.dumps(self.sampledict[self._samp_scambiunit])
#             # be careful with raspberry pi memory
#             self.sampledict[self._samp_scambiunit] = None
#             self._samp_scambiunit += 1
#             return output

#         return None
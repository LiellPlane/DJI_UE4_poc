import cv2
import math
from collections import deque
from functools import lru_cache
from enum import Enum, auto
import os
import numpy as np
import time
import random
import logging
from contextlib import contextmanager
from typing import Iterator, Literal, Annotated, Optional
from dataclasses import dataclass
from skimage.draw import line
from my_collections import CropSlicing, AffinePoints, UI_ready_element, ScreenPixelPositions
from math import floor
try:# TODO FIX
    import utils
except:
    print("this is really bad please fix scambilight import issue")

Array3x3 = Annotated[np.ndarray, (3, 3)]
RED = (0, 0, 255)
BLUE = (255, 0, 0)


class EventLogOverlay:
    """Event log overlay for game events (optimized for Raspberry Pi)
    
    Shows recent events in a semi-transparent box with cached rendering.
    Events automatically fade and expire:
    - < 5 seconds: bright color (based on log level)
    - 5-10 seconds: fades to dark color
    - > 10 seconds: dark color
    - > 20 seconds: removed
    
    Log levels (uses standard logging library - fully compatible with comms_http.LogEvent):
    - logging.CRITICAL (50): Bright red, bold text
    - logging.WARNING (30): Orange/yellow
    - logging.INFO (20): Green
    
    Cache refreshes every 1 second to update colors and remove old events.
    
    Usage:
        import logging
        from comms_http import HTTPComms, LogEvent
        
        # Create overlay
        log_overlay = EventLogOverlay(rotation=90)
        log_overlay.set_header("GAME LOG")
        
        # Add events directly
        log_overlay.add_event("Player joined", logging.INFO)  # Green
        log_overlay.add_event("Low ammo!", logging.WARNING)   # Orange
        log_overlay.add_event("Eliminated!", logging.CRITICAL)  # Red (bold)
        
        # Or pop from HTTPComms and display
        event: Optional[LogEvent] = http_comms.pop_oldest_event()
        if event:
            log_overlay.add_event(event.text, event.level)
        
        # Apply to frame
        log_overlay.apply_to_image(frame)  # Fast in-place overlay
    """
    
    # Class variables for log levels (standard logging library values - optional shortcuts)
    CRITICAL = logging.CRITICAL  # 50
    WARNING = logging.WARNING    # 30
    INFO = logging.INFO          # 20
    
    def __init__(
        self,
        position: Optional[tuple[int, int]] = None,  # None = auto top-right
        box_size: tuple[int, int] = (600, 150),  # Before rotation (width, height)
        rotation: Literal[0, 90, -90, 180, 270] = 90,
        max_events: int = 5,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        text_color: tuple[int, int, int] = (20, 190, 10),  # Slightly dark green (BGR) - used for INFO
        bg_color: tuple[int, int, int] = (0, 0, 0),
        bg_alpha: float = 0.75,
        line_spacing: int = 5,
    ):
        self._initial_position = position  # None means calculate from image size
        self.box_size = box_size
        self.rotation = rotation
        self.max_events = max_events
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.text_color = text_color
        self.bg_color = bg_color
        self.bg_alpha = bg_alpha
        self.line_spacing = line_spacing
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Define bright colors for each log level (BGR format) - using standard logging levels
        self._level_colors = {
            logging.CRITICAL: (0, 0, 255),      # Pure bright red
            logging.WARNING: (100, 220, 255),   # Light orange/yellow
            logging.INFO: (20, 190, 10)         # Green (existing color)
        }
        
        # Define dark colors for faded state (BGR format)
        self._level_dark_colors = {
            logging.CRITICAL: (0, 0, 80),       # Dark red
            logging.WARNING: (0, 60, 100),      # Dark orange
            logging.INFO: (0, 60, 0)            # Dark green
        }
        
        self.events = deque(maxlen=max_events)
        self.event_timestamps = deque(maxlen=max_events)  # Track when each event was added
        self.event_levels = deque(maxlen=max_events)      # Track log level for each event
        self._static_header = None  # Static text at top (doesn't scroll)
        self._cached_overlay = None
        self._cached_mask = None  # Cache the alpha mask too
        self._cache_dirty = True
        self._last_refresh_time = time.time()  # For 1-second refresh cycle
        
        # Pre-calculate line height
        (_, h), baseline = cv2.getTextSize(
            "Ay", self.font, self.font_scale, self.font_thickness
        )
        self.line_height = h + baseline + self.line_spacing
        
        # Pre-calculate rotated dimensions (constant)
        if self.rotation in [90, -90, 270]:
            # After 90° rotation: width becomes height, height becomes width
            self._rotated_width = self.box_size[1]   # height (150) becomes width
            self._rotated_height = self.box_size[0]  # width (300) becomes height
        else:
            self._rotated_width = self.box_size[0]
            self._rotated_height = self.box_size[1]
        
        # Cache position after first calculation
        self._cached_position = None
        
        # Pre-calculate max text width for truncation
        self._max_text_width = self.box_size[0] - 10
    
    def set_header(self, header_text: Optional[str]) -> None:
        """Set static header text at top (doesn't scroll with events)"""
        self._static_header = header_text
        self._cache_dirty = True
    
    def add_event(self, event_text: str, level: int = None) -> None:
        """Add new event (pushes old events up/out)
        
        Args:
            event_text: Text to display
            level: Log level (use logging.CRITICAL, logging.WARNING, logging.INFO). Defaults to random for testing.
        """
        if level is None:
            level = random.choice([logging.CRITICAL, logging.WARNING, logging.INFO])
        self.events.append(event_text)
        self.event_timestamps.append(time.time())
        self.event_levels.append(level)
        self._cache_dirty = True
    
    def clear_events(self) -> None:
        """Clear all events"""
        self.events.clear()
        self.event_timestamps.clear()
        self.event_levels.clear()
        self._cache_dirty = True
    
    def _render_overlay(self) -> np.ndarray:
        """Render event box WITH rotation baked in (only called when cache dirty)"""
        width, height = self.box_size
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[:, :] = self.bg_color
        
        # Draw static header at top if set
        if self._static_header is not None:
            header_y_pos = self.line_height + 5  # Position from top
            header_text = self._static_header
            
            # Truncate header if too wide (estimate chars to remove to avoid repeated getTextSize)
            (text_w, _), _ = cv2.getTextSize(
                header_text, self.font, self.font_scale, self.font_thickness
            )
            if text_w > self._max_text_width:
                # Estimate chars per pixel and truncate in one go
                chars_to_remove = int((text_w - self._max_text_width + 15) / (text_w / len(header_text))) + 1
                header_text = header_text[:max(3, len(header_text) - chars_to_remove)] + "..."
            
            # Bright blue color for header (BGR)
            header_color = (255, 100, 0)
            cv2.putText(
                overlay, header_text, (5, header_y_pos),
                self.font, self.font_scale, header_color,
                self.font_thickness, cv2.LINE_AA
            )
        
        # Draw scrolling events top-to-bottom (becomes left-to-right after 90° rotation)
        # Start after header (or at top if no header)
        if self._static_header is not None:
            y_pos = self.line_height + 10 + self.line_height  # After header + spacing
        else:
            y_pos = self.line_height + 5
        
        current_time = time.time()
        
        for event_text, timestamp, level in zip(self.events, self.event_timestamps, self.event_levels):
            if y_pos > height - self.line_spacing:
                break
            
            # Get colors for this log level
            bright_color = self._level_colors[level]
            dark_color = self._level_dark_colors[level]
            
            # Calculate age and color fade
            age_seconds = current_time - timestamp
            if age_seconds < 5:
                # Bright color for < 5 seconds
                event_color = bright_color
            elif age_seconds < 10:
                # Fade from bright to dark between 5-10 seconds
                fade_progress = (age_seconds - 5) / 5  # 0.0 to 1.0
                event_color = tuple(
                    int(bright_color[i] * (1 - fade_progress) + dark_color[i] * fade_progress)
                    for i in range(3)
                )
            else:
                # Dark color for >= 10 seconds (until removed at 20)
                event_color = dark_color
            
            # Truncate if too wide (estimate chars to remove to avoid repeated getTextSize)
            (text_w, _), _ = cv2.getTextSize(
                event_text, self.font, self.font_scale, self.font_thickness
            )
            if text_w > self._max_text_width:
                # Estimate chars per pixel and truncate in one go
                chars_to_remove = int((text_w - self._max_text_width + 15) / (text_w / len(event_text))) + 1
                event_text = event_text[:max(3, len(event_text) - chars_to_remove)] + "..."
            
            # Make CRITICAL messages bolder
            thickness = self.font_thickness + 1 if level == logging.CRITICAL else self.font_thickness
            
            cv2.putText(
                overlay, event_text, (5, y_pos),
                self.font, self.font_scale, event_color,
                thickness, cv2.LINE_AA
            )
            y_pos += self.line_height  # Move DOWN (becomes right after rotation)
        
        # Rotate once here (only happens when regenerating cache)
        if self.rotation == 90:
            overlay = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation in [-90, 270]:
            overlay = cv2.rotate(overlay, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.rotation == 180:
            overlay = cv2.rotate(overlay, cv2.ROTATE_180)
        
        return overlay
    
    def _get_position(self, image_shape: tuple[int, int]) -> tuple[int, int]:
        """Calculate top-right position once and cache it"""
        if self._cached_position is not None:
            return self._cached_position
        
        if self._initial_position is not None:
            self._cached_position = self._initial_position
            return self._cached_position
        
        # Calculate top-right position using pre-calculated rotated dimensions
        img_height, img_width = image_shape[:2]
        x = img_width - self._rotated_width - 10
        y = 10
        
        self._cached_position = (x, y)
        return self._cached_position
    
    def apply_to_image(self, image: np.ndarray) -> None:
        """Apply pre-rotated overlay to image (in-place modification)
        
        Args:
            image: BGR image to overlay events on (modified in-place)
        """
        # Check if 1 second has passed - time to refresh and cleanup
        current_time = time.time()
        if current_time - self._last_refresh_time >= 1.0:
            # Remove events older than 20 seconds
            while self.events and (current_time - self.event_timestamps[0]) > 20.0:
                self.events.popleft()
                self.event_timestamps.popleft()
                self.event_levels.popleft()
            
            # Trigger cache refresh to update colors
            self._cache_dirty = True
            self._last_refresh_time = current_time
        
        if not self.events and self._static_header is None:
            return  # Nothing to render
        
        if self._cache_dirty:
            self._cached_overlay = self._render_overlay()  # Rotation happens here
            
            # Pre-compute mask once (only text pixels are blended, black bg is transparent)
            gray_overlay = cv2.cvtColor(self._cached_overlay, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
            mask_normalized = mask.astype(float) / 255.0
            mask_alpha = mask_normalized * self.bg_alpha
            self._cached_mask = mask_alpha  # Cache it!
            
            self._cache_dirty = False
        
        x, y = self._get_position(image.shape)  # Cached after first call
        
        # Use pre-calculated rotated dimensions (no per-frame calculation)
        width = self._rotated_width
        height = self._rotated_height
        
        # Clip to image bounds
        y_end = min(y + height, image.shape[0])
        x_end = min(x + width, image.shape[1])
        h = y_end - y
        w = x_end - x
        
        if h > 0 and w > 0:
            roi = image[y:y_end, x:x_end]
            overlay_region = self._cached_overlay[:h, :w]
            mask_alpha = self._cached_mask[:h, :w]
            
            # Invert mask for background weight
            inv_mask = 1.0 - mask_alpha
            
            # Fast blend: text pixels get blended, black background stays transparent
            roi[:] = (overlay_region * mask_alpha[:, :, np.newaxis] + 
                     roi * inv_mask[:, :, np.newaxis]).astype(np.uint8)


def interpolate_points(start, end, steps):
    return np.linspace(start, end, steps)

def ease_in_out_quad(t):
    return 2 * t**2 if t < 0.5 else 1 - (-2 * t + 2)**2 / 2

def ease_in_out_quart(t):
    return 8 * t**4 if t < 0.5 else 1 - (-2 * t + 2)**4 / 2

def ease_in_out_sine(t):
    return -(np.cos(np.pi * t) - 1) / 2

def ease_in_out_cubic(t):
    return np.where(t < 0.5, 4 * t**3, 1 - (-2 * t + 2)**3 / 2)

def interpolate_points_eased(start, end, steps):
    t = np.linspace(0, 1, steps)
    eased_t = ease_in_out_cubic(t)
    return start + eased_t[:, np.newaxis] * (end - start)

@dataclass
class CamDisplayTransform:
    cam_image_shape: tuple[int]
    

def darken_image(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def gray2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)



def radial_motion_blur(image, intensity=30):
    center = (image.shape[0]//2, image.shape[1]//2)
    h, w = image.shape
    y, x = np.ogrid[:h, :w]
    
    # Calculate angle and distance from center for each pixel
    angle = np.arctan2(y - center[1], x - center[0])
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Normalize distances
    max_dist = np.sqrt(w**2 + h**2)
    norm_dist = dist_from_center / max_dist
    
    # Create motion blur
    blurred = np.zeros_like(image, dtype=np.float32)
    for i in range(intensity):
        offset = i / intensity
        x_offset = offset * np.cos(angle) * norm_dist * intensity
        y_offset = offset * np.sin(angle) * norm_dist * intensity
        
        x_shift = np.round(x + x_offset).astype(int)
        y_shift = np.round(y + y_offset).astype(int)
        
        # Ensure we don't go out of bounds
        x_shift = np.clip(x_shift, 0, w-1)
        y_shift = np.clip(y_shift, 0, h-1)
        
        blurred += image[y_shift, x_shift] / intensity
    
    return blurred.astype(np.uint8)

@dataclass
class TransformsDetails:
    longrange_to_shortrange_perwarp: Array3x3 # warp calculated to seamlessly embed longrange image into closerange(quilt)
    closerange_to_display: CamDisplayTransform # transform to fit output display (with aspect ratio)
    longrange_to_display: CamDisplayTransform # transform to fit output display (with aspect ratio)
    transition_steps: int # moving between longe range and close range
    transition_time_secs: float # desired transition time to be LERPED
    display_image_shape: tuple[int]
    displayrotation: Literal[0, 90, 180, 270] # rotation of LCD screen on chassis
    slice_details_close_range: Optional[CropSlicing]
    slice_details_long_range: Optional[CropSlicing]


class CameraTransitionState(Enum):
    CLOSERANGE = auto()
    LONGRANGE = auto()
    TRANSITIONING = auto()


class TransformManager:
    def __init__(self, transformdetails: TransformsDetails):
        """all transitions assume that index 0 is closerange engated and last index is longrange engaged"""
        self._transitions_direction = 1
        self._triggered_time = time.perf_counter()
        self._current_managed_index = 0
        self.transformdetails: TransformsDetails = transformdetails
        # CR camera starts as fully engaged, then warps in to engage the LR. 
        # we need a starting point - so we use the points from the LR mapped to SR space (so embedded in the image)
        # then this transitions to fully engaged LR. These transforms can be used to warp the CR
        self.LR_lerp: list = self._get_close_to_long_transition_points()
        self.CR_transition_m: list[Array3x3] = self._get_transition_Matrices(
            target_array=self.LR_lerp,
            source_array=self.LR_lerp[:, 0]
            )
        # this transition is the LR from embedded in the CR to fully engaged.
        # we need the static transform of the embedded LR in CR space, and we also need the 
        # transforms lerping from this embedded to fully engaged. This needs 2 transforms
        self.LR_transition_m: list[Array3x3] = self._longrange_transition_calc_m(
            self.CR_transition_m,
            self.transformdetails.longrange_to_shortrange_perwarp
            )
        # there is a final transform which displays these images for the display. These are much smaller
        # so any blurring operations are best done here
        # we also have different aspect ratios of LR and CR, so the transition will have to include this
        # transform as well, so the display field is recalculated each transition with this new output shape
        self.displaytransition_lerp: list = self._get_display_transition_points()
        self.display_affine_transition_m = self._get_display_affine_transitions()
        self.display_warp_transition_m: list[Array3x3] = self._convert_affine_to_3x3(
            self.display_affine_transition_m
            )
        # # that didnt seem to work - so lets add a lerp between the two camera shapes as well
        # self.shape_transition_m: list[Array3x3] = self._get_transition_Matrices(
        #     target_array=self.displaytransition_lerp,
        #     source_array=self.displaytransition_lerp[:, 0]
        #     )
        self.CR_all_transition_m: list[Array3x3] = self._matmul_lists(
            list1=self.display_warp_transition_m,
            list2=self.CR_transition_m 
            )
        self.LR_all_transition_m: list[Array3x3] = self._matmul_lists(
            list1=self.display_warp_transition_m,
            list2=self.LR_transition_m 
            )
        # this is optional
        if self.transformdetails.slice_details_close_range and self.transformdetails.slice_details_long_range:
            self.lerped_slice_details = self.lerp_slice_details()

    def get_lerped_targetzone_slice(self, index):
        """the slices have to be provided or this will die"""
        return self.lerped_slice_details[index]

    def lerp_slice_details(self):
        cr_slice_np  = np.asarray(self.transformdetails.slice_details_close_range.get_as_tuple())
        lr_slice_np  = np.asarray(self.transformdetails.slice_details_long_range.get_as_tuple())
        lerped =  lerp_arrays(cr_slice_np,lr_slice_np,self.transformdetails.transition_steps)
        output = []
        for i in range(0, lerped.shape[1]):

            output.append(CropSlicing(
                left=lerped[:,i][0],
                right=lerped[:,i][1],
                top=lerped[:,i][2],
                lower=lerped[:,i][3]
                ))
        return output

    def get_display_affine_transformation(self, index):
        return self.display_affine_transition_m[index]
    
    def trigger_transition(self):
        '''alert manager that we want to it to generate transform indexes proportional to time delta'''
        self._transitions_direction *= -1
        self._triggered_time = time.perf_counter()


    def get_transition_state(self) -> CameraTransitionState:
        if self._current_managed_index <= 0:
            return CameraTransitionState.CLOSERANGE
        elif self._current_managed_index >= self.transformdetails.transition_steps-1: # zero based thing
            return CameraTransitionState.LONGRANGE
        else:
            return CameraTransitionState.TRANSITIONING
        

    def get_deltatime_transition(self):
        '''if you have triggered the trigger_transitions, use this to get current index proportional time delta and configured time span'''
        if any([
            (self._current_managed_index >= self.transformdetails.transition_steps) and self._transitions_direction == 1,
            (self._current_managed_index <= 0) and self._transitions_direction == -1
        ]):
            "don't bother calculating everything"
            pass
        else: 
            
            time_delta_sec = time.perf_counter() - self._triggered_time
            self.transformdetails.transition_time_secs
            self.transformdetails.transition_steps
            percent_done = time_delta_sec/self.transformdetails.transition_time_secs
            steps_in_timespan = int(self.transformdetails.transition_steps * percent_done) * self._transitions_direction
            self._current_managed_index += steps_in_timespan
            
            self._current_managed_index = min(max(0, self._current_managed_index), self.transformdetails.transition_steps)
        
        self._triggered_time = time.perf_counter()
        return self._current_managed_index

    @staticmethod
    def _matmul_lists(list1: list[Array3x3], list2: list[Array3x3]) -> list[Array3x3]:
        assert len(list1) == len(list2)
        matrices = []
        for mat1, mat2 in zip(list1, list2):
            matrices.append(np.matmul(mat1, mat2))
        return matrices

    def _convert_affine_to_3x3(self, affinetransforms: list) -> list [Array3x3]:
        matrices = []
        for affine_t in affinetransforms:
            warp_matrix = np.eye(3, dtype=np.float32)
            warp_matrix[:2, :] = affine_t
            matrices.append(warp_matrix)
        return matrices
    
    def _get_display_affine_transitions(self):
        transforms = []
        for i in range(0, self.displaytransition_lerp.shape[1]):
            corners = self.displaytransition_lerp[:, i]
            shape = (
                int(corners[:, 1].max()),
                int(corners[:, 0].max())
            )
            transforms.append(get_fitted_affine_transform(
                    cam_image_shape=shape,
                    display_image_shape=self.transformdetails.display_image_shape,
                    rotation=self.transformdetails.displayrotation
                    )
                )
        return transforms

    @staticmethod
    def _longrange_transition_calc_m(cr_transition_m: list[Array3x3], perpwarp: Array3x3):
        matrices = []
        for mat in cr_transition_m:
            matrices.append(np.matmul(mat, perpwarp))
        return matrices

    def _get_transition_Matrices(
            self,
            target_array: list[np.ndarray],
            source_array: np.ndarray):
        matrices = []
        for i in range(0, target_array.shape[1]):
            matrices.append(self._calc_perp_transform(
                src_points=source_array,
                dst_points=target_array[:, i]
                )
                )
        return matrices

    def _get_close_to_long_transition_points(self):
        long_range_corners = get_imagecorners_as_np_array(self.transformdetails.longrange_to_display.cam_image_shape)
        long_range_corners_in_SR_coords = mtransform_array_of_points(long_range_corners,self.transformdetails.longrange_to_shortrange_perwarp )
        #close_range_corners = get_imagecorners_as_np_array(self.transformdetails.closerange_to_display.cam_image_shape)
        lerped = self._get_lerped_points(long_range_corners_in_SR_coords, long_range_corners)
        return lerped

    def _get_display_transition_points(self):
        long_range_corners = get_imagecorners_as_np_array(self.transformdetails.longrange_to_display.cam_image_shape)
        close_range_corners = get_imagecorners_as_np_array(self.transformdetails.closerange_to_display.cam_image_shape)
        lerped = self._get_lerped_points(close_range_corners, long_range_corners)
        return lerped

    @staticmethod
    def _calc_perp_transform(src_points, dst_points) -> np.ndarray:

        return cv2.getPerspectiveTransform(
            np.array(src_points, dtype=np.float32),
            np.array(dst_points, dtype=np.float32)
            )

    def _get_lerped_points(self, startarray, endarray):
        """lerp between two sets of points, for instance provide 4 corners of one image and 4 corners of another and lerp between them"""
        interpolated_points = [
            interpolate_points_eased(start, end, self.transformdetails.transition_steps)
            for start, end
            in zip(
            startarray,
            endarray
            )
        ]
        return np.array(interpolated_points)

def draw_border_rectangle(image, thickness=2, color=(255, 255, 255)):
    """Draw a rectangle outline around the border of an image.
    
    Args:
        image: Input image (modified in-place)
        thickness: Thickness of the border rectangle outline
        color: Color of the rectangle outline (B, G, R) for BGR images
    
    Returns:
        None (modifies image in-place for maximum speed)
    """
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), color, thickness)


def lerp_arrays(a, b, num_points):
    t = np.linspace(0, 1, num_points)
    return np.array([np.interp(t, [0, 1], [a_i, b_i]) for a_i, b_i in zip(a, b)])

def get_imagecorners_as_np_array(imgshape: tuple[int]):
    return np.asarray([(0, imgshape[0]), (0,0), (imgshape[1], 0), (imgshape[1], imgshape[0])])


def mtransform_array_of_points(myarray:np.ndarray, mytransform: Array3x3) -> np.ndarray :
    homogeneous_points = np.column_stack((myarray, np.ones(len(myarray))))
    transformed_points = np.dot(homogeneous_points, mytransform.T)
    transformed_points_2d = transformed_points[:, :2] / transformed_points[:, 2:]
    return transformed_points_2d


def read_img(img_filepath):
    return cv2.imread(img_filepath)


def compute_and_apply_perpwarp(src_img, dst_img, src_points, dst_points):
    # Convert points to numpy arrays
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation to the source image
    height, width = dst_img.shape[:2]
    result = cv2.warpPerspective(src_img, matrix, (width, height))

    return result, matrix

def overlay_warped_image(background, warped):
    # Ensure the images have the same size and are mono
    assert background.shape == warped.shape, "Images must have the same dimensions"
    assert len(background.shape) == 2 and len(warped.shape) == 2, "Images must be mono (single channel)"
    
    # Ensure images are 8-bit unsigned integer type
    background = background.astype(np.uint8)
    warped = warped.astype(np.uint8)

    # Create a mask based on non-black pixels in the warped image
    _, mask = cv2.threshold(warped, 1, 255, cv2.THRESH_BINARY)

    # Black-out the area of warped image in background
    background_masked = cv2.bitwise_and(background, cv2.bitwise_not(mask))

    # Combine the background and warped image
    result = cv2.add(background_masked, warped)

    return result

def overlay_warped_image_alpha(background, warped, alpha=0.1):
    # Ensure the images have the same size and are mono
    assert background.shape == warped.shape, "Images must have the same dimensions"
    assert len(background.shape) == 2 and len(warped.shape) == 2, "Images must be mono (single channel)"
    
    # Ensure images are 8-bit unsigned integer type
    background = background.astype(np.uint8)
    warped = warped.astype(np.uint8)

    # Create a mask based on non-black pixels in the warped image
    _, mask = cv2.threshold(warped, 1, 255, cv2.THRESH_BINARY)

    # Convert mask to float and normalize
    mask = mask.astype(float) / 255.0

    # Apply alpha to the mask
    mask *= alpha

    # Invert the mask
    inv_mask = 1.0 - mask

    # Blend the images
    result = (background * inv_mask + warped * mask).astype(np.uint8)

    return result


def overlay_warped_image_alpha_feathered(background, warped, alpha=0.1, feather_amount=10):
    # Ensure the images have the same size and are mono
    assert background.shape == warped.shape, "Images must have the same dimensions"
    assert len(background.shape) == 2 and len(warped.shape) == 2, "Images must be mono (single channel)"
    
    # Ensure images are 8-bit unsigned integer type
    background = background.astype(np.uint8)
    warped = warped.astype(np.uint8)

    # Create a mask based on non-black pixels in the warped image
    _, mask = cv2.threshold(warped, 1, 255, cv2.THRESH_BINARY)

    # Apply feathering to the mask
    mask = cv2.GaussianBlur(mask, (feather_amount*2+1, feather_amount*2+1), 0)

    # Convert mask to float and normalize
    mask = mask.astype(float) / 255.0

    # Apply alpha to the mask
    mask *= alpha

    # Invert the mask
    inv_mask = 1.0 - mask

    # Blend the images
    result = (background * inv_mask + warped * mask).astype(np.uint8)

    return result


def apply_perp_transform(matrix, src_img, dst_img):
    
    # Apply the perspective transformation to the source image
    height, width = dst_img.shape[:2]
    result = cv2.warpPerspective(src_img, matrix, (width, height))

    return result


def concat_image(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation = cv2.INTER_NEAREST)
    return cv2.hconcat([img1,img2 ])

def clahe_equalisation(img, claheprocessor):
    if claheprocessor is None:
        claheprocessor = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(50,50))
    # colour
    if len(img.shape) >2:
        #luminosity
        lab_image=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab_image)
        #equ = cv2.equalizeHist(l)
        #updated_lab_img1=cv2.merge((equ,a,b))
        clahe_img= claheprocessor.apply(l)
        updated_lab_img1=cv2.merge((clahe_img,a,b))
        CLAHE_img = cv2.cvtColor(updated_lab_img1,cv2.COLOR_LAB2BGR)
    # grayscale
    else:
        CLAHE_img = claheprocessor.apply(img)
    return CLAHE_img

def _3_chan_equ(img):
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return equalized_img

def mono_img(img):
    if len(img.shape) < 3:
        return img
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def invert_img(img):
    return np.invert(img)

def equalise_img(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def blur_img(img, filtersize = 7):
    return cv2.GaussianBlur(img,(filtersize,filtersize),0)

def blur_average(img, filtersize = 7):
    kernel = np.ones((filtersize,filtersize),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    return dst

def normalise(img):
    image2_Norm = cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
    return image2_Norm

def threshold_img(img, high=255):
    #_ , th3 = cv2.threshold(img, low, 255,cv2.THRESH_BINARY)
    th3 = cv2.adaptiveThreshold(img,high,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,1)
    #_,th3 = cv2.threshold(img,low,high,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th3 = cv2.adaptiveThreshold(img,high,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return th3

def threshold_img_static(img, low=0, high=255):
    #_ , th3 = cv2.threshold(img, low, 255,cv2.THRESH_BINARY)
    #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,1)
    _,th3 = cv2.threshold(img,low,high,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th3 = cv2.adaptiveThreshold(img,high,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return th3

def edge_img(gray):
    #edges = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
    #edges = cv2.Canny(image=img, threshold1=100, threshold2=200)

    # Smoothing without removing edges.

    #     d- Diameter of each pixel neighborhood that is
    #     used during filtering. If it is non-positive, it is
    #     computed from sigmaSpace.
    # sigmaColor -Filter sigma in the color space. A larger value of the 
    # parameter means that farther colors within the pixel neighborhood
    # will be mixed together, resulting in larger areas of semi-equal color.

    # sigmaSpace - Filter sigma in the coordinate space. A larger value
    # of the parameter means that farther pixels will influence each other
    # as long as their colors are close enough. When d>0, it specifies the
    # neighborhood size regardless of sigmaSpace. Otherwise, d is proportional
    # to sigmaSpace.
    gray_filtered = cv2.bilateralFilter(gray, 5, 10, 4)

    # Applying the canny filter
    #edges = cv2.Canny(gray, 60, 120)
    edges_filtered = cv2.Canny(gray_filtered, 0, 60)

    # Stacking the images to print them together for comparison
    #images = np.hstack((gray, edges, edges_filtered))
    
    return gray_filtered

def simple_canny(blurred_img, lower, upper):
    # wide = cv2.Canny(blurred, 10, 200)
    # mid = cv2.Canny(blurred, 30, 150)
    # tight = cv2.Canny(blurred, 240, 250)
    return cv2.Canny(blurred_img, upper, lower, 7,L2gradient = False)

def get_hist(img):
    #fig = plt.figure()
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #plt.hist(img.ravel(),256,[0,256]); plt.show()
    # plt.plot(hist)
    # plt.ylabel("histogram")
    # plt.ylim([0, max(hist)])
    # graph_image = np.array(fig.canvas.get_renderer()._renderer)
    # plt.cla()
    # plt.clf()
    # plt.close()
    
    # graph_image = np.array(fig.canvas.get_renderer()._renderer)
    return hist

def cut_square(img):
    length = 100 
    center_x = int(img.shape[0]/2)
    center_y = int(img.shape[1]/2)
    top = center_y - length
    lower = center_y + length
    left = center_x - length
    right = center_x + length
    cut = img[left:right,top:lower,:]
    return cut

def contours_img(img):
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    cv2.drawContours(out, contours, -1, 255,1)
    #cv2.drawContours(image=out, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    return out

def dilate(InputImage):
    kernel = np.ones((3, 3), np.uint8)
    img_blur = cv2.medianBlur(InputImage, 3)
    dilated_image = cv2.dilate(img_blur, kernel, iterations = 1)
    #eroded_image = cv2.erode(dilated_image, kernel, iterations = 5)
    return dilated_image

def median_blur(inputimage, kernalsize):
    return  cv2.medianBlur(inputimage, kernalsize)

def image_resize_ratio(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)

    # return the resized image
    return resized

def get_resized_equalaspect(inputimage, screensize):

    tofit_height = inputimage.shape[0]
    tofit_width = inputimage.shape[1]
    to_fit_height2width = tofit_height / tofit_width
    screensize_height = screensize[0]
    screensize_width = screensize[1]

    # fit to height and check if width is ok
    test_height = screensize_height
    test_width = screensize_height / to_fit_height2width
    dim = None
    if test_width > screensize_width:
        test_width = screensize_width
        test_height = test_width * to_fit_height2width
        if test_height > screensize_height:
            raise Exception(
                "screen resize with aspect ratio has failed in heght & width cases, bad")

    dim = (
        int(np.floor(test_width)),
        int(np.floor(test_height)))

    return cv2.resize(inputimage, dim, interpolation = cv2.INTER_NEAREST)

def resize_centre_img(inputimage, screensize):

    # this is slow - might be faster passing in the image again?
    # TODO
    # empty is faster than zeros
    emptyscreen = np.zeros((screensize + (3,)), np.uint8)

    if screensize[0] < screensize[1]:
        image = image_resize_ratio(
            inputimage,
            height=screensize[0])
    else:
        image = image_resize_ratio(
            inputimage,
            width=screensize[1])


    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 

    offset_x = (emptyscreen.shape[0] - image.shape[0]) // 2
    offset_y = (emptyscreen.shape[1] - image.shape[1]) // 2
    emptyscreen[
        offset_x:image.shape[0]+offset_x,
        offset_y:image.shape[1]+offset_y,
        :] = image

    # should be equal scaling for dims as maintains aspect ratio
    scale_factor = (image.shape[0] / inputimage.shape[0])
    return emptyscreen, scale_factor


def add_cross_hair(image, adapt, lerp = 0):
    thick = 3
    vis_block = 30 + int(lerp * 60)
    midx = image.shape[0] // 2
    midy = image.shape[1] // 2
    # TODO another potential lag point
    if adapt is True:
        col = int(
            image[midx-50:midx+50:4, midy-50:midy+50:4, :].mean())
    else:
        col = 255

    red = int(255 * (1-lerp))
    green = int((lerp) * max(col, 50))

    image[midx - thick : midx + thick, 0:midy-vis_block, 2] = red
    image[midx - thick : midx + thick, 0:midy-vis_block, 1] = green
    image[midx - thick : midx + thick, midy+vis_block:-1, 2] = red
    image[midx - thick : midx + thick, midy+vis_block:-1, 1] = green

    image[0:midx-vis_block, midy - thick : midy + thick, 2] = red
    image[0:midx-vis_block, midy - thick : midy + thick, 1] = green
    image[midx+vis_block:-1, midy - thick : midy + thick, 2] = red
    image[midx+vis_block:-1, midy - thick : midy + thick, 1] = green
 

class lerped_add_crosshair():
    def __init__(self) -> None:
        self.lerper = utils.Lerp(
            start_value=1,
            end_value=0,
            duration=0.15,
            easing="ease_in_out_cubic"
            )
        self.buffer = deque(maxlen=4)
        self.last_target_acquired = False
        #self.debouncecheck = utils.SequenceDetector(3,[True,True,True])

    def add_target_state(self, target_state):
        self.buffer.append(target_state)
        return sum(x for x in self.buffer) > 2

    def add_cross_hair(self, image, adapt, target_acquired=False):
        """wrap the add cross hair function so we can lerp it easily
        lerp in when target is acquired and lerp back out when lost"""
        target_acquired_ok = self.add_target_state(target_acquired)
        self.lerper.set_direction_forward(target_acquired_ok)
        add_cross_hair(image, adapt, self.lerper.get_value())
        self.last_target_acquired = target_acquired



def get_internal_section(imgshape, size: tuple[int, int]):
    midx = imgshape[0] // 2
    midy = imgshape[1] // 2
    regionx = size[0] // 2
    regiony = size[1] // 2
    left = max(midx-regionx, 0)
    right = min(midx+regionx, imgshape[0])
    top = max(midy-regiony, 0)
    lower = min(midy+regiony, imgshape[1])
    return CropSlicing(left=left, right=right, top=top, lower=lower)

def implant_internal_section(img, img_to_implant):

    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if len(img_to_implant.shape) < 3:
        img_to_implant = cv2.cvtColor(img_to_implant, cv2.COLOR_GRAY2RGB)
    #draw white border around area to implant
    img_to_implant[2:img_to_implant.shape[0]-2, 2,:] = 255
    img_to_implant[2:img_to_implant.shape[0]-2, img_to_implant.shape[1]-1,:] = 255
    img_to_implant[2, 2:img_to_implant.shape[1]-2,:] = 255
    img_to_implant[img_to_implant.shape[0]-2, 2:img_to_implant.shape[1]-1,:] = 255
    midx = img.shape[0] // 2
    midy = img.shape[1] // 2
    regionx = img_to_implant.shape[0] // 2
    regiony = img_to_implant.shape[1] // 2
    # specifying the area to implant is incase of odd sized half, so might miss 
    # a pixel leading to broadcast error
    img[midx-regionx:midx+regionx,
       midy-regiony:midy+regiony, :] = img_to_implant[0:regionx*2, 0:regiony*2]
    return img

# def bresenham_line_wikipedia(x0, y0, x1, y1):
# https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm


def bresenham_line_ski(x1,y1,x2, y2):
    rr, cc = line(x1,y1,x2, y2)
    return [i for i in zip(rr, cc)]


def efficient_line_sampler(x1, y1, x2, y2, num_samples):
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return np.tile([x1, y1], (num_samples, 1))

    steps = max(abs(dx), abs(dy))
    
    if steps < num_samples - 1:
        # For short lines, use linear interpolation
        t = np.linspace(0, 1, num_samples)
        x = np.round(x1 + t * dx).astype(int)
        y = np.round(y1 + t * dy).astype(int)
    else:
        # For longer lines, use step-based approach
        indices = np.round(np.linspace(0, steps, num_samples)).astype(int)
        x = np.round(x1 + indices * (dx / steps)).astype(int)
        y = np.round(y1 + indices * (dy / steps)).astype(int)
    
    return np.column_stack((x, y))

def get_affine_transform(pts1, pts2):
    """from 2 sets of 3 corresponding points
    calculate the affine transform"""
    return cv2.getAffineTransform(pts1, pts2)


def do_affine(img, T, row_cols: tuple[int, int]):
    return cv2.warpAffine(img, T, row_cols, flags=cv2.INTER_NEAREST)


def rotate_pt_around_origin(point, origin, degrees):
    radians = np.deg2rad(degrees)
    x,y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + (cos_rad * adjusted_x) + (sin_rad * adjusted_y)
    qy = offset_y + (-sin_rad * adjusted_x) + (cos_rad * adjusted_y)
    return qx, qy


# def add_ui_elements(
#         image,
#         element_package: UI_ready_element,
#         fade_norm: float
#         ) -> None:
#     image[
#             element_package.position.top:element_package.position.lower,
#             element_package.position.left:element_package.position.right,
#             0
#         ] = (element_package.image * fade_norm).astype(np.uint8)

def add_ui_elementsv2(
        image,
        position: ScreenPixelPositions,
        image_to_insert: np.array,
        channel: int,
        fade_norm: float
        ) -> None:
    if fade_norm < 0.01:
        return
    #rand = random.randint(-2, 2)

    image[
            position.top: position.lower,
            position.left: position.right,
            channel
        ] = (image_to_insert * fade_norm).astype(np.uint8)

    return


def resize_image(inputimage, width, height):
    return cv2.resize(inputimage, (width, height), interpolation = cv2.INTER_NEAREST)


def draw_pattern_output(image, patterndetails, debug=False): # ShapeItem - TODO 
    """draw graphics for user if a pattern is found
    TODO: maybe want floating numbers etc above this which
    will eventually need a user registry"""
    min_bbox = patterndetails.boundingbox_min
    cX, cY = patterndetails.centre_x_y
    closest_corners = patterndetails.closest_corners
    # corners of square
    cv2.circle(image, tuple(min_bbox[0]), 3, RED, 1)
    cv2.circle(image, tuple(min_bbox[2]), 3, RED, 1)
    cv2.circle(image, tuple(min_bbox[1]), 3, RED, 1)
    cv2.circle(image, tuple(min_bbox[3]), 3, RED, 1)


    # centre of pattern
    cv2.circle(image, (cX, cY), 5, RED, 1)
   
    # bounding box of contour - this does not handle perspective
    cv2.drawContours(image, [min_bbox], 0, RED)

    cv2.fillPoly(image, [min_bbox], RED)
    #draw barcode sampling lines - for illustration only
    # may not match exactly with generated sampled lines
    if debug is False:
        cv2.line(image, tuple(closest_corners[0]), tuple(closest_corners[2]), BLUE, 1) 
        cv2.line(image, tuple(closest_corners[1]), tuple(closest_corners[3]), BLUE, 1)


def load_img_set_transparency():
    #  Debug code until we have a user avatar delivery system
    imgfoler = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    imgfoler = imgfoler[:imgfoler.index("Source")]
    imgsource = f"{imgfoler}Source/lumotag/avatars/chick.png"
    return cv2.imread(imgsource, cv2.IMREAD_UNCHANGED)


def get_fitted_affine_transform(
    cam_image_shape,
    display_image_shape,
    rotation: Literal[0, -90, 90, 270, 180]
    ):
    """Get the matrix transform to rotate and scale a camera image to fit
    in a display image space, maintaining aspect ratio.
    
    cam image:
    
    _______    ^
    |^ ^ ^|    |
    |^ ^ ^|    | up vector
    |^_^_^|    |
    
    display LCD image:
    
    |-------------------|
    |                   |  <------- UP vector (user POV)
    |-------------------|

    and we want to rotate the cam image 90 degrees (so its correct for viewer)
    then scale and centralise to fit in display image

    |------|<-<-<-|------|
    |      |< < < |      |  <------- UP vector (user POV)
    |------|<-<-<-|------|

    Note the implanted camera image has been rotated to fit.

    this function only works for 90 degree angles
    """
    if rotation in [90, -90, 270]:
        reverse_output_shape = tuple(reversed(display_image_shape[0:2]))
        # if planning for 90 degrees, swap image dims
        input_targets, output_targets = get_affine_points(
            cam_image_shape,
            reverse_output_shape)
        output_targets = rotate_affine_targets(
            output_targets,
            rotation,
            reverse_output_shape)

        diffs = (np.asarray(reverse_output_shape) - np.asarray(display_image_shape[0:2]))/2
        output_targets.add_offset_h(diffs[1])
        output_targets.add_offset_w(diffs[0])

    elif rotation == 180:
        input_targets, output_targets = get_affine_points(
            cam_image_shape,
            display_image_shape)
        # have to flip output targets
        output_targets = rotate_affine_targets(
            output_targets,
            rotation,
            display_image_shape)

    elif rotation == 0:
        input_targets, output_targets = get_affine_points(
            cam_image_shape,
            (display_image_shape))

    return get_affine_transform(
        pts1=np.asarray(input_targets.as_array(), dtype="float32"),
        pts2=np.asarray(output_targets.as_array(), dtype="float32"))


def get_affine_points(incoming_img_dims, outgoing_img_dims) -> AffinePoints:
    """Return the corresponding points to fit the incoming image central to the
    view screen maintaining the aspect ratio, to be used to calculate affine
    transform
    
    inputs:
    incoming_img_dims: numpy array .shape
    outcoming_img_dims: numpy array .shape

    return source points, target points
    """
    incoming_w = incoming_img_dims[1]
    incoming_h = incoming_img_dims[0]
    outgoing_w = outgoing_img_dims[1]
    outgoing_h = outgoing_img_dims[0]
    incoming_pts = AffinePoints(
        top_left_w_h=(0, 0),
        top_right_w_h=(incoming_w, 0),
        lower_right_w_h=(incoming_w, incoming_h))
    # pick any ratio
    ratio = outgoing_h / incoming_h
    # if resizing with aspect ratio doesn't fit, do the other way
    if floor(incoming_w * ratio) > outgoing_w:
        ratio = outgoing_w / incoming_w
    output_fit_h = floor(incoming_h * ratio)
    output_fit_w = floor(incoming_w * ratio)
    # test to make sure aspect ratio is 
    if abs((incoming_h/incoming_w) - (outgoing_h/outgoing_w)) > 2:
        raise ValueError("error calculating output image dimensions")
    
    if abs(output_fit_w-outgoing_w)>1 and abs(output_fit_h-outgoing_h)>1:
        raise ValueError("error calculating output image dimensions")
    # get 3 corresponding points from the output view - keeping in mind
    # any rotation
    w_crop_in = (outgoing_w - output_fit_w) // 2
    h_crop_in = (outgoing_h - output_fit_h) // 2
    view_pts = AffinePoints(
        top_left_w_h=(w_crop_in, h_crop_in),
        top_right_w_h=(w_crop_in + output_fit_w, h_crop_in),
        lower_right_w_h=(w_crop_in + output_fit_w, h_crop_in + output_fit_h))

    return incoming_pts, view_pts


def rotate_affine_targets(targets, degrees, outputscreen_shape):
    mid_img = [int(x/2) for x in outputscreen_shape[0:2][::-1]] # get reversed dims
    new_target = AffinePoints(
                top_left_w_h=rotate_pt_around_origin(targets.top_left_w_h, mid_img, degrees),
                top_right_w_h=rotate_pt_around_origin(targets.top_right_w_h, mid_img, degrees),
                lower_right_w_h=rotate_pt_around_origin(targets.lower_right_w_h, mid_img, degrees))
    return new_target


def test_viewer(
        inputimage,
        pausetime_Secs=0,
        presskey=False,
        destroyWindow=True):

    cv2.imshow("img", inputimage)
    if presskey==True:
        cv2.waitKey(0); #any key

    if presskey==False:
        if cv2.waitKey(20) & 0xFF == 27:
                pass
    if pausetime_Secs>0:
        time.sleep(pausetime_Secs)
    if destroyWindow==True: cv2.destroyAllWindows()


def rotate_img_orthogonal(img, rotation: Literal[0, 90, -90, 180, 270]):
    if rotation in [0, 360]:
        return img
    if rotation in [90]:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotation in [-90, 270]:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation in [180]:
        return cv2.rotate(img, cv2.ROTATE_180)
    raise ValueError("bad rotation req", rotation)


def get_empty_lumodisplay_img(imgshape: tuple[int, int]):
    return np.zeros(
            (imgshape + (3,)), np.uint8)


def print_text_in_boundingbox(text: str, grayscale: bool):
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1.0
    FONT_THICKNESS = 2

    (label_width, label_height), baseline = cv2.getTextSize(
        text,
        FONT,
        FONT_SCALE,
        FONT_THICKNESS)

    label_patch = np.zeros((label_height + baseline, label_width, 3), np.uint8)

    cv2.putText(
        label_patch,
        text,
        (0, label_height),
        FONT,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS)
    
    if grayscale is True:
        label_patch = cv2.cvtColor(label_patch, cv2.COLOR_BGR2GRAY)

    return label_patch


def fast_sample(image, coordinates):
    coords = np.array(coordinates, dtype=np.int32)
    y_coords, x_coords = coords[:, 1], coords[:, 0]
    sampled_pixels = image[y_coords, x_coords]
    return sampled_pixels.astype(np.uint8)


def normalise_np_array(data):
    data_min = data.min()
    data_max = data.max()
    if data_max > data_min:
        normalized_data = (data - data_min) / (data_max - data_min)
    else:
        normalized_data = np.zeros_like(data, dtype=np.float16)
    return normalized_data

def binarize_barcode(normalized_data):
    return (normalized_data > 0.5).astype(np.int8)


def add_text_to_image(image, text):
    height, width = image.shape[:2]
    
    # Calculate font scale based on image width
    font_scale = width/800 # This will make the text size relative to the image width
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)  # Red in BGR
    thickness = 1
    
    # Position text in top left with some padding
    position = (10, 50)  # (x, y) coordinates
    
    # Put text on image
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    return image


def quick_image_viewer(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@lru_cache
def rotate_points_right_angle(points, angle, width, height):
    """
    Rotate points by the given angle.

    :param points: List of tuples [(x1, y1), (x2, y2), ...]
    :param angle: Rotation angle in degrees (0, 90, 180, 270)
    :param width: Original image width
    :param height: Original image height
    :return: List of tuples with rotated points
    """
    rotated = []
    for (x, y) in points:
        if angle in [0, 360]:
            new_x = x
            new_y = y
        elif angle == 270:
            new_x = y
            new_y = width - x
        elif angle == 180:
            new_x = width - x
            new_y = height - y
        elif angle == 90:
            new_x = height - y
            new_y = x
        else:
            raise ValueError("Angle must be 0, 90, 180, or 270 degrees")
        rotated.append((new_x, new_y))
    return rotated


def apply_hud_flicker(image, flicker_density=0.9):
    """
    Apply a simple HUD flicker effect by creating random black spots.
    
    Parameters:
    - image: numpy array (height, width) or (height, width, channels)
    - flicker_density: float 0-1, what proportion of pixels to affect
    
    Returns:
    - Modified image array
    """
    # Work on a copy to preserve original
    result = image.copy()
    
    # Get image dimensions
    if len(image.shape) == 2:
        h, w = image.shape
        channels = 1
    else:
        h, w, channels = image.shape
    
    # Calculate number of pixels to flicker
    n_pixels = int(h * w * flicker_density)
    
    # Generate random positions efficiently
    y_pos = np.random.randint(0, h, n_pixels)
    x_pos = np.random.randint(0, w, n_pixels)
    
    if channels == 1:
        # Set random pixels to black
        result[y_pos, x_pos] = 0
    else:
        # Set random pixels to black in all channels
        result[y_pos, x_pos, :] = 0
    
    return result


def find_black_blobs(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Threshold for black regions - very fast operation
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours - relatively fast
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours


@lru_cache(maxsize=1)
def get_mser_detector():
    print("Creating new MSER detector - cache miss")  # Debug statement
    # Configure MSER parameters for better performance
    mser = cv2.MSER_create(
        min_area=1000,  # Minimum area size (50*50)
        delta=5,
        # max_variation=0.25
    )
    return mser

def get_mser_regions(image, preprocess=False):
    """Get MSER regions as contours from input image.
    
    Args:
        image: Input grayscale image
        preprocess: Whether to apply preprocessing (default: True)
        
    Returns:
        List of contours in format compatible with cv2.findContours output
    """
    # Optional preprocessing to improve detection speed and quality
    if preprocess:
        # Resize large images
        max_dim = 800
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
            
        # Reduce noise while preserving edges
        image = cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)
    
    # Get cached MSER detector
    mser = get_mser_detector()
    
    # Detect regions using numpy array directly
    msers, bboxes = mser.detectRegions(image)
    return msers, bboxes
    # More efficient contour conversion using numpy operations
    if regions:
        contours = [np.expand_dims(region, 1) for region in regions]
        return contours
    return []


def visualize_mser_regions1(image_shape, regions):
    output = np.zeros(image_shape + (3,), dtype=np.uint8)
    for region in regions:
        # Generate random BGR color (50-200 range to avoid too dark/bright colors)
        color = (
            random.randint(50, 200),  # B
            random.randint(50, 200),  # G 
            random.randint(50, 200)   # R
        )
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.polylines(output, [hull], 1, color, 2)
    return output

def visualize_mser_regions(image_shape, contours):
    """Visualize MSER regions with random colors
    Args:
        image_shape: Shape of original image (height, width)
        contours: List of contours in OpenCV format (from get_mser_regions or findContours)
        
    Returns:
        RGB image with randomly colored MSER region contours
    """
    # Debug print for contours
    print(f"Number of contours: {len(contours)}")
    if len(contours) > 0:
        print(f"Shape of first contour: {contours[0].shape}")
    
    # Input validation
    if not contours:
        return np.zeros(image_shape + (3,), dtype=np.uint8)
    
    # Create output image (3 channels for RGB)
    output = np.zeros(image_shape + (3,), dtype=np.uint8)
    
    # Generate random colors for each region
    for contour in contours:
        if contour is not None and len(contour) > 0:
            color = (
                random.randint(50, 200),  # B
                random.randint(50, 200),  # G
                random.randint(50, 200)   # R
            )
            # Changed thickness from -1 to 2 to draw contour lines
            cv2.drawContours(output, [contour], -1, color, 1)
    
    return output

def detect_mser_regions(image):
    """Legacy function that combines detection and visualization.
    Consider using get_mser_regions() and visualize_mser_regions() separately."""
    contours = get_mser_regions(image)
    return visualize_mser_regions(image.shape, contours)



def visualize_mser_bboxes(image, bboxes):
    """Draw bounding boxes from MSER regions on an image.
    
    Args:
        image: Input image (grayscale or BGR)
        bboxes: List of bounding boxes in format (x, y, w, h)
    
    Returns:
        Image with drawn bounding boxes
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
        
    # Draw each bounding box with a random color
    for bbox in bboxes:
        # Generate random BGR color (50-200 range to avoid too dark/bright colors)
        color = (
            random.randint(50, 200),  # B
            random.randint(50, 200),  # G 
            random.randint(50, 200)   # R
        )
        x, y, w, h = bbox
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
    
    return vis_image

def apply_bloom_effect(image, threshold=200, blur_size=7, intensity=1.5):
    # Create a bright pass filter
    bright_pass = np.zeros_like(image)
    bright_mask = np.max(image, axis=2) > threshold
    bright_pass[bright_mask] = image[bright_mask]
    
    # Apply multiple blur passes for better bloom
    bloom = cv2.GaussianBlur(bright_pass, (0, 0), blur_size)
    bloom = cv2.GaussianBlur(bloom, (0, 0), blur_size * 2)
    
    # Combine original image with bloom
    result = cv2.addWeighted(image, 1.0, bloom, intensity, 0)
    return result


def apply_anti_aliasing(image, blur_size=1.5):
    """Apply a subtle blur for anti-aliasing effect."""
    return cv2.GaussianBlur(image, (0, 0), blur_size)


def apply_noise(image, variance=10, coverage=0.3):
    """Apply random noise to an image.
    
    Args:
        image: Input image
        variance: Amount of noise (higher = more noise)
        coverage: Percentage of pixels to affect (0.0 to 1.0)
    """
    noise = np.random.normal(0, variance, image.shape).astype(np.int16)
    mask = np.random.random(image.shape) < coverage
    result = image.astype(np.int16) + (noise * mask)
    return np.clip(result, 0, 255).astype(np.uint8)


def create_health_bar(health_value, width=400, height=600, num_segments=10, use_anti_aliasing=True, use_noise=True, high_health_color='green'):
    # Create a black canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate segment height and add padding using percentages
    padding_top = int(math.ceil(height * 0.067))  # 6.7% of height
    padding_bottom = int(math.ceil(height * 0.067))  # 6.7% of height
    padding_sides = int(math.ceil(width * 0.1))  # 10% of width
    usable_height = height - (padding_top + padding_bottom)
    segment_height = usable_height // num_segments
    segment_width = width - (2 * padding_sides)  # Leave margin on both sides
    
    # Calculate number of active segments
    active_segments = int(health_value * num_segments)
    
    # Define colours
    if high_health_color.lower() == 'blue':
        high_health_colour = (255, 0, 0)  # Blue in BGR
    else:  # default to green
        high_health_colour = (0, 255, 0)  # Green in BGR
    low_health_colour = (0, 0, 255)   # Pure Red
    threshold = 0.4
    
    # Calculate corner radius (make it proportional to segment height)
    corner_radius = int(math.ceil(segment_height * 0.4))  # 40% of segment height for more prominent rounding
    
    # Draw segments from bottom to top
    for i in range(num_segments):
        if i < active_segments:
            # Determine colour based on health
            is_low_health = health_value <= threshold
            colour = low_health_colour if is_low_health else high_health_colour
            
            # Calculate segment position with padding
            y_start = height - padding_bottom - (i + 1) * segment_height
            y_end = height - padding_bottom - i * segment_height
            
            # Create segment with gradient
            segment = np.zeros((segment_height, segment_width, 3), dtype=np.uint8)
            
            # Draw the main rounded rectangle
            # Draw the center rectangle
            cv2.rectangle(segment, (corner_radius, 0), (segment_width - corner_radius, segment_height), colour, -1)
            cv2.rectangle(segment, (0, corner_radius), (segment_width, segment_height - corner_radius), colour, -1)
            
            # Draw the four rounded corners using ellipses
            # Top-left corner
            cv2.ellipse(segment, (corner_radius, corner_radius), (corner_radius, corner_radius), 180, 0, 90, colour, -1)
            # Top-right corner
            cv2.ellipse(segment, (segment_width - corner_radius, corner_radius), (corner_radius, corner_radius), 270, 0, 90, colour, -1)
            # Bottom-left corner
            cv2.ellipse(segment, (corner_radius, segment_height - corner_radius), (corner_radius, corner_radius), 90, 0, 90, colour, -1)
            # Bottom-right corner
            cv2.ellipse(segment, (segment_width - corner_radius, segment_height - corner_radius), (corner_radius, corner_radius), 0, 0, 90, colour, -1)
            
            # Add yellow highlight line (adjusted for rounded corners)
            highlight_y = int(math.ceil(segment_height * 0.3))
            cv2.line(segment, (corner_radius, highlight_y), 
                    (segment_width - corner_radius, highlight_y), (0, 255, 255), 2)
            
            # Add enhanced highlight effect
            highlight = np.zeros_like(segment)
            highlight_height = int(math.ceil(segment_height * 0.7))
            
            # Draw the highlight with rounded corners
            # Center rectangle
            cv2.rectangle(highlight, (corner_radius, 0), (segment_width - corner_radius, highlight_height), (255, 255, 255), -1)
            cv2.rectangle(highlight, (0, corner_radius), (segment_width, highlight_height - corner_radius), (255, 255, 255), -1)
            
            # Rounded corners for highlight
            # Top-left corner
            cv2.ellipse(highlight, (corner_radius, corner_radius), (corner_radius, corner_radius), 180, 0, 90, (255, 255, 255), -1)
            # Top-right corner
            cv2.ellipse(highlight, (segment_width - corner_radius, corner_radius), (corner_radius, corner_radius), 270, 0, 90, (255, 255, 255), -1)
            
            # Apply stronger blur for more glow
            highlight = cv2.GaussianBlur(highlight, (0, 0), 60)
            # Increase highlight intensity significantly
            segment = cv2.addWeighted(segment, 2.0, highlight, 0.4, 0)
            
            # Draw the border after the highlight effect
            # Draw the straight edges
            cv2.line(segment, (corner_radius, 0), (segment_width - corner_radius, 0), (0, 0, 0), 2)
            cv2.line(segment, (corner_radius, segment_height), (segment_width - corner_radius, segment_height), (0, 0, 0), 2)
            cv2.line(segment, (0, corner_radius), (0, segment_height - corner_radius), (0, 0, 0), 2)
            cv2.line(segment, (segment_width, corner_radius), (segment_width, segment_height - corner_radius), (0, 0, 0), 2)
            
            # Draw the rounded corners for the border
            # Top-left corner
            cv2.ellipse(segment, (corner_radius, corner_radius), (corner_radius, corner_radius), 180, 0, 90, (0, 0, 0), 2)
            # Top-right corner
            cv2.ellipse(segment, (segment_width - corner_radius, corner_radius), (corner_radius, corner_radius), 270, 0, 90, (0, 0, 0), 2)
            # Bottom-left corner
            cv2.ellipse(segment, (corner_radius, segment_height - corner_radius), (corner_radius, corner_radius), 90, 0, 90, (0, 0, 0), 2)
            # Bottom-right corner
            cv2.ellipse(segment, (segment_width - corner_radius, segment_height - corner_radius), (corner_radius, corner_radius), 0, 0, 90, (0, 0, 0), 2)
            
            # Apply noise if enabled
            if use_noise:
                # Use more noise for low health segments
                noise_variance = 30 if is_low_health else 15
                noise_coverage = 0.4 if is_low_health else 0.2
                segment = apply_noise(segment, variance=noise_variance, coverage=noise_coverage)
            
            # Apply anti-aliasing if enabled
            if use_anti_aliasing:
                segment = apply_anti_aliasing(segment)
            
            # Add segment to canvas with side padding
            canvas[y_start:y_end, padding_sides:padding_sides+segment_width] = segment
    
    # Apply bloom effect with different parameters for low health
    if health_value <= threshold:
        # Extra intense bloom for low health
        canvas = apply_bloom_effect(canvas, threshold=100, blur_size=20, intensity=2.0)
    else:
        # Normal bloom for high health
        canvas = apply_bloom_effect(canvas, threshold=100, blur_size=15, intensity=1.8)
    
    return canvas


def generate_red_tv_static(image_shape, block_size=5):
    """Generate vibrant red TV-style static with large pixel blocks"""
    height, width = image_shape[:2]
    
    # Calculate how many blocks we can fit
    blocks_h = height // block_size
    blocks_w = width // block_size
    
    # Create small random array for blocks
    static_blocks = np.random.choice([0, 1], size=(blocks_h, blocks_w), p=[0.1, 0.9])
    
    # Create output image
    static_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(blocks_h):
        for j in range(blocks_w):
            y_start = i * block_size
            y_end = min((i + 1) * block_size, height)
            x_start = j * block_size
            x_end = min((j + 1) * block_size, width)
            
            if static_blocks[i, j]:
                # Maximum red intensity - mostly red with minimal white
                if np.random.random() < 0.05:  # Only 5% chance of white
                    static_image[y_start:y_end, x_start:x_end] = [255, 255, 255]  # Pure white
                else:  # 95% chance of red
                    static_image[y_start:y_end, x_start:x_end] = [0, 0, 255]  # Pure bright red
            else:
                # Pure black pixels for maximum contrast
                static_image[y_start:y_end, x_start:x_end] = [0, 0, 0]
    
    return static_image


def combine_channels_to_red(image):
    """Combine all BGR channels into red channel only (in-place)
    
    Args:
        image: BGR image to modify (modified in-place)
    """
    image[:,:,2] = np.minimum(image[:,:,0] + image[:,:,1] + image[:,:,2], 255)
    image[:,:,0] = 0
    image[:,:,1] = 0


def display_split_rotated_images(image_shape, image_list):
    """Display up to 2 images rotated 90 degrees clockwise, split top/bottom
    
    Args:
        image_shape: Target output shape (height, width, channels)
        image_list: List of images (0, 1, or 2 images)
    
    Returns:
        Combined image with rotated images in top/bottom halves
    """
    height, width = image_shape[:2]
    output_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    if not image_list:
        return output_image  # Return black screen if no images
    
    # Calculate half height for splitting
    half_height = height // 2
    
    # Process first image (top half)
    if len(image_list) >= 1:
        img1 = image_list[0]
        # Rotate 90 degrees clockwise
        rotated_img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        # Resize maintaining aspect ratio to fit top half
        resized_img1 = get_resized_equalaspect(rotated_img1, (half_height, width))
        # Center the image in the top half
        y_offset = (half_height - resized_img1.shape[0]) // 2
        x_offset = (width - resized_img1.shape[1]) // 2
        output_image[y_offset:y_offset+resized_img1.shape[0], 
                    x_offset:x_offset+resized_img1.shape[1]] = resized_img1
    
    # Process second image (bottom half) if available
    if len(image_list) >= 2:
        img2 = image_list[1]
        # Rotate 90 degrees clockwise
        rotated_img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
        # Resize maintaining aspect ratio to fit bottom half
        resized_img2 = get_resized_equalaspect(rotated_img2, (half_height, width))
        # Center the image in the bottom half
        y_offset = half_height + (half_height - resized_img2.shape[0]) // 2
        x_offset = (width - resized_img2.shape[1]) // 2
        output_image[y_offset:y_offset+resized_img2.shape[0], 
                    x_offset:x_offset+resized_img2.shape[1]] = resized_img2
    
    return output_image
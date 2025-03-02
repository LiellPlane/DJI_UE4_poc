import cv2
import numpy as np
import math
import random
import time
from dataclasses import dataclass

class CardioGramDisplay:
    def __init__(self, pos_x, pos_y, width, height, value_range=(-1, 1), flow_direction=0):
        """
        pos_x, pos_y: Top-left position in the parent image.
        width, height: Size of the display region.
        value_range: Expected numeric range for metric values.
        flow_direction: Direction (in degrees) for history flow:
            0   -> New data at bottom (scrolls upward)
            180 -> New data at top (scrolls downward)
            90  -> New data at right (scrolls leftward)
            270 -> New data at left (scrolls rightward)
        """
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.width = width
        self.height = height
        self.value_range = value_range
        self.flow_direction = flow_direction % 360

        # Create an overlay with an alpha channel (BGRA)
        self.overlay = np.zeros((height, width, 4), dtype=np.uint8)
        self.metrics = {"test": self.Metric("test", 0)}
        
        # Pre-calculate the gradient matrix - OPTIMIZATION
        self._precalculate_gradient()
        
        # Create cache for colors - OPTIMIZATION
        self.color_cache = {}
        
        # Pre-allocate arrays for compositing - OPTIMIZATION
        self._temp_overlay = np.zeros_like(self.overlay)
        
    def _precalculate_gradient(self):
        """Pre-compute the gradient for fade effect - OPTIMIZATION"""
        if self.flow_direction in (0, 180):
            if self.flow_direction == 0:
                self.gradient = np.linspace(0, 1, self.height).reshape(self.height, 1)
            else:
                self.gradient = np.linspace(1, 0, self.height).reshape(self.height, 1)
        else:
            if self.flow_direction == 90:
                self.gradient = np.linspace(0, 1, self.width).reshape(1, self.width)
            else:
                self.gradient = np.linspace(1, 0, self.width).reshape(1, self.width)

    @dataclass
    class MetricUpdate:
        __slots__ = ['metric', 'color', 'slice_row', 'slice_col']
        metric: str
        color: tuple[int, int, int, int]
        slice_row: slice
        slice_col: slice

    @dataclass
    class Metric:
        __slots__ = ['metric', 'pos']
        metric: str
        pos: int

    def apply_image_actions(self, image, image_actions):
        """
        Applies a list of image actions to the given image.
        """
        for action in image_actions:
            image[action.slice_row, action.slice_col] = action.color

    def update_metrics(self, updates: dict[str, float]) -> list[MetricUpdate]:
        """
        updates: Dictionary mapping metric names to values
        """
        min_val, max_val = self.value_range
        offset_range = 40
        bar_thickness = 3
        
        # Shift the overlay along the chosen axis - OPTIMIZED to avoid unnecessary operations
        if self.flow_direction == 0:
            # New data at bottom; shift upward.
            self.overlay[:-1, :, :] = self.overlay[1:, :, :]
            self.overlay[-1, :, :] = 0
            new_edge = [i for i in range(self.height - 1, self.height - offset_range, -1)]
        elif self.flow_direction == 180:
            # New data at top; shift downward.
            self.overlay[1:, :, :] = self.overlay[:-1, :, :]
            self.overlay[0, :, :] = 0
            new_edge = [i for i in range(0, offset_range, 1)]
        elif self.flow_direction == 90:
            # New data at right; shift left.
            self.overlay[:, :-1, :] = self.overlay[:, 1:, :]
            self.overlay[:, -1, :] = 0
            new_edge = [i for i in range(self.width - 1, self.width - offset_range, -1)]
        elif self.flow_direction == 270:
            # New data at left; shift right.
            self.overlay[:, 1:, :] = self.overlay[:, :-1, :]
            self.overlay[:, 0, :] = 0
            new_edge = [i for i in range(0, offset_range, 1)]

        output_actions = []
        # For each metric update, compute the coordinate and set the pixel.
        for cnt, (metric, value) in enumerate(updates.items()):
            if metric not in self.metrics:
                max_pos = max(self.metrics.values(), key=lambda m: m.pos).pos
                self.metrics[metric] = self.Metric(metric, max_pos + bar_thickness)
            pos = self.metrics[metric].pos
            
            # Get color from cache or create new - OPTIMIZATION
            if metric not in self.color_cache:
                # Calculate step based on total number of metrics
                step = 255 // max(len(self.metrics), 1)
                # Generate colors using a simple formula
                red = (255 - cnt * step) % 256
                green = (cnt * step) % 256
                blue = (128 + cnt * step) % 256  # Offset blue for better visibility
                self.color_cache[metric] = (red, green, blue)
            
            color = self.color_cache[metric]

            # Normalize the value to a 0..1 scale - OPTIMIZATION: integer math where possible
            value = min(max(value, min_val), max_val)
            norm_val = (value - min_val) / (max_val - min_val)
            
            if self.flow_direction == 0:
                new_x = int(norm_val * (self.width - 1))
                self.overlay[new_edge[pos], new_x] = (color[0], color[1], color[2], 255)
                row_slice = slice(new_edge[pos]-bar_thickness, new_edge[pos])
                col_slice = slice(0, new_x)
            elif self.flow_direction == 180:
                new_x = int(norm_val * (self.width - 1))
                self.overlay[new_edge[pos], self.width - new_x - 1] = (color[0], color[1], color[2], 255)
                row_slice = slice(new_edge[pos], new_edge[pos]+bar_thickness)
                col_slice = slice(self.width - new_x - 1, self.width)
            elif self.flow_direction == 90:
                new_y = int(norm_val * (self.height - 1))
                self.overlay[new_y, new_edge[pos]] = (color[0], color[1], color[2], 255)
                row_slice = slice(0, new_y)
                col_slice = slice(new_edge[pos]-bar_thickness, new_edge[pos])
            elif self.flow_direction == 270:
                new_y = int(norm_val * (self.height - 1))
                self.overlay[self.height - new_y - 1, new_edge[pos]] = (color[0], color[1], color[2], 255)
                row_slice = slice(self.height - new_y, self.height)
                col_slice = slice(new_edge[pos], new_edge[pos]+bar_thickness)
            else:
                raise ValueError(f"Invalid flow direction: {self.flow_direction}")
            
            output_actions.append(self.MetricUpdate(metric, (color[0], color[1], color[2], 255),
                                                 slice_row=row_slice, slice_col=col_slice))
        return output_actions

    def get_overlay_with_gradient(self):
        """
        Returns a copy of the overlay with a fade gradient applied along the history flow axis.
        OPTIMIZED to use pre-calculated gradient and minimize copies.
        """
        # Reuse pre-allocated array instead of creating a new one
        np.copyto(self._temp_overlay, self.overlay)
        
        # Use the pre-calculated gradient - much faster
        self._temp_overlay[:,:,3] = (self._temp_overlay[:,:,3] * self.gradient).astype(np.uint8)
        
        return self._temp_overlay

    def composite_onto_inplace(self, background, image_actions):
        """
        Composites the overlay onto the background using OpenCV's optimized functions.
        OPTIMIZED to use addWeighted instead of manual per-pixel operations.
        """
        # Get overlay with gradient and apply actions
        overlay = self.get_overlay_with_gradient()
        self.apply_image_actions(overlay, image_actions)
        
        # Extract region of interest
        h, w = self.height, self.width
        roi = background[self.pos_y:self.pos_y+h, self.pos_x:self.pos_x+w]
        
        # Create temporary float arrays for calculation
        alpha = overlay[:,:,3].astype(float) / 255.0
        alpha_inv = 1.0 - alpha
        
        # Create a temporary result array
        temp_result = np.zeros_like(roi, dtype=float)
        
        # Use vectorized operations for each channel
        for c in range(3):
            # Blend overlay and background using alpha
            temp_result[:,:,c] = overlay[:,:,c] * alpha + roi[:,:,c] * alpha_inv
        
        # Convert back to uint8 and store in the ROI
        np.copyto(roi, temp_result.astype(np.uint8))
        
        return background

# ------------------------ Test Script ------------------------

if __name__ == '__main__':
    # Create a test parent background image (vertical grayscale gradient).
    bg_height = 600
    bg_width = 800
    background = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
    for i in range(bg_height):
        shade = int(255 * (i / bg_height))
        background[i, :, :] = (shade, shade, shade)

    # Define display parameters.
    disp_pos_x = 0   # X position in the background
    disp_pos_y = 500   # Y position in the background
    disp_width = 300  # Width of the overlay region
    disp_height = 80 # Height of the overlay region

    # Set desired flow direction (0, 90, 180, or 270).
    flow_direction =90  # For example, 0°: new data appears at the bottom.

    # Create an instance of the display.
    display = CardioGramDisplay(disp_pos_x, disp_pos_y, disp_width, disp_height,
                                value_range=(0, 100), flow_direction=flow_direction)

    t = 0.0
    dt = 0.05

    cv2.namedWindow("CardioGram Test", cv2.WINDOW_AUTOSIZE)
    while True:
        # Build a dictionary of metric updates.
        # Each update provides a value and a color (BGR) for that metric.
        updates = {}
        # Metric "A" (red) is available from the start.
        valueA = 25 + 25 * math.sin(t * 1.0) + random.uniform(-2, 2)
        updates["A"] = (max(0, min(50, valueA)))
        # Metric "B" (green) starts after 1 second.
        if t > 1:
            valueB = 25 + 25 * math.sin(t * 1.2 + math.pi/4) + random.uniform(-2, 2)
            updates["B"] = (max(0, min(50, valueB)))
        # Metric "C" (blue) starts after 2 seconds.
        if t > 2:
            valueC = 25 + 25 * math.sin(t * 0.8 + math.pi/2) + random.uniform(-2, 2)
            updates["C"] = (max(0, min(50, valueC)))
        # updates["D"] = (10, (0, 255, 255), 8)
        # Update the display with the provided metric updates.
        image_actions = display.update_metrics(updates)
        # Composite the overlay (with fade gradient) onto a copy of the background.
        output = display.composite_onto_inplace(background.copy(), image_actions)
        
        cv2.imshow("CardioGram Test", output)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break
        t += dt

    cv2.destroyAllWindows()

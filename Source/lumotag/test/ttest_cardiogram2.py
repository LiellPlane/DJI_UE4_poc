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
        OPTIMIZED with direct indexing and pre-allocated arrays
        """
        min_val, max_val = self.value_range
        offset_range = 40
        bar_thickness = 3
        
        # OPTIMIZATION: Use more efficient array shifting with direct slice assignment
        if self.flow_direction == 0:
            # Shift directly with NumPy's optimized memory handling
            self.overlay[:-1, :] = self.overlay[1:, :]
            self.overlay[-1:, :] = 0  # Clear the last row
            # Pre-calculate positions once
            edge_positions = np.arange(self.height - 1, self.height - offset_range, -1)
        elif self.flow_direction == 180:
            self.overlay[1:, :] = self.overlay[:-1, :]
            self.overlay[:1, :] = 0
            edge_positions = np.arange(0, offset_range)
        elif self.flow_direction == 90:
            self.overlay[:, :-1] = self.overlay[:, 1:]
            self.overlay[:, -1:] = 0
            edge_positions = np.arange(self.width - 1, self.width - offset_range, -1)
        elif self.flow_direction == 270:
            self.overlay[:, 1:] = self.overlay[:, :-1]
            self.overlay[:, :1] = 0
            edge_positions = np.arange(0, offset_range)
        
        # OPTIMIZATION: Pre-calculate 1/(max-min) for faster normalization
        norm_factor = 1.0 / (max_val - min_val)
        
        output_actions = []
        for cnt, (metric, value) in enumerate(updates.items()):
            if metric not in self.metrics:
                max_pos = max(self.metrics.values(), key=lambda m: m.pos).pos
                self.metrics[metric] = self.Metric(metric, max_pos + bar_thickness)
            pos = self.metrics[metric].pos
            
            # if metric not in self.color_cache:
            #     # Calculate colors (same as before)
            #     step = 255 // max(len(self.metrics), 1)
            #     red = (255 - cnt * step) % 256
            #     green = (cnt * step) % 256
            #     blue = (128 + cnt * step) % 256
            #     self.color_cache[metric] = (red, green, blue)
            
            # color = self.color_cache[metric]
            step = 255 // max(len(self.metrics), 1)
            red = (255 - cnt * step) % 256
            green = (cnt * step) % 256
            blue = (128 + cnt * step) % 256
            color = (red, green, blue)
            # OPTIMIZATION: Faster normalization with pre-calculated factor
            value = min(max(value, min_val), max_val)
            norm_val = (value - min_val) * norm_factor
            
            # OPTIMIZATION: Process flow directions with cleaner code
            if self.flow_direction == 0:
                new_x = int(norm_val * (self.width - 1))
                edge_pos = edge_positions[pos]
                # Draw 4 pixels instead of 1 for better visibility
                self.overlay[edge_pos, new_x] = (color[0], color[1], color[2], 255)
                if new_x > 0:
                    self.overlay[edge_pos, new_x-1] = (color[0], color[1], color[2], 255)
                if new_x < self.width-1:
                    self.overlay[edge_pos, new_x+1] = (color[0], color[1], color[2], 255)
                if edge_pos > 0:
                    self.overlay[edge_pos-1, new_x] = (color[0], color[1], color[2], 255)
                row_slice = slice(edge_pos-bar_thickness, edge_pos)
                col_slice = slice(0, new_x)
            elif self.flow_direction == 180:
                new_x = int(norm_val * (self.width - 1))
                edge_pos = edge_positions[pos]
                x_pos = self.width - new_x - 1
                # Draw 4 pixels instead of 1 for better visibility
                self.overlay[edge_pos, x_pos] = (color[0], color[1], color[2], 255)
                if x_pos > 0:
                    self.overlay[edge_pos, x_pos-1] = (color[0], color[1], color[2], 255)
                if x_pos < self.width-1:
                    self.overlay[edge_pos, x_pos+1] = (color[0], color[1], color[2], 255)
                if edge_pos < self.height-1:
                    self.overlay[edge_pos+1, x_pos] = (color[0], color[1], color[2], 255)
                row_slice = slice(edge_pos, edge_pos+bar_thickness)
                col_slice = slice(self.width - new_x - 1, self.width)
            elif self.flow_direction == 90:
                new_y = int(norm_val * (self.height - 1))
                edge_pos = edge_positions[pos]
                # Draw 4 pixels instead of 1 for better visibility
                self.overlay[new_y, edge_pos] = (color[0], color[1], color[2], 255)
                if new_y > 0:
                    self.overlay[new_y-1, edge_pos] = (color[0], color[1], color[2], 255)
                if new_y < self.height-1:
                    self.overlay[new_y+1, edge_pos] = (color[0], color[1], color[2], 255)
                if edge_pos > 0:
                    self.overlay[new_y, edge_pos-1] = (color[0], color[1], color[2], 255)
                row_slice = slice(0, new_y)
                col_slice = slice(edge_pos-bar_thickness, edge_pos)
            elif self.flow_direction == 270:
                new_y = int(norm_val * (self.height - 1))
                edge_pos = edge_positions[pos]
                y_pos = self.height - new_y - 1
                # Draw 4 pixels instead of 1 for better visibility
                self.overlay[y_pos, edge_pos] = (color[0], color[1], color[2], 255)
                if y_pos > 0:
                    self.overlay[y_pos-1, edge_pos] = (color[0], color[1], color[2], 255)
                if y_pos < self.height-1:
                    self.overlay[y_pos+1, edge_pos] = (color[0], color[1], color[2], 255)
                if edge_pos < self.width-1:
                    self.overlay[y_pos, edge_pos+1] = (color[0], color[1], color[2], 255)
                row_slice = slice(self.height - new_y, self.height)
                col_slice = slice(edge_pos, edge_pos+bar_thickness)
            
            output_actions.append(self.MetricUpdate(metric, (color[0], color[1], color[2], 255),
                                               slice_row=row_slice, slice_col=col_slice))
        return output_actions

    def get_overlay_with_gradient(self):
        """
        OPTIMIZED to minimize array operations and type conversions
        """
        # Use provided array reference if no modifications needed
        # OPTIMIZATION: Apply gradient with minimal memory operations
        np.multiply(self.overlay[:,:,3], self.gradient, out=self._temp_overlay[:,:,3], casting='unsafe')
        np.copyto(self._temp_overlay[:,:,0:3], self.overlay[:,:,0:3])
        
        return self._temp_overlay

    def composite_onto_inplace(self, background, image_actions):
        """
        FINAL OPTIMIZATION: Simplified blending with uint8 math instead of float conversions
        """
        # Get overlay with gradient and apply actions
        overlay = self.get_overlay_with_gradient()
        self.apply_image_actions(overlay, image_actions)
        
        # Extract region of interest
        h, w = self.height, self.width
        roi = background[self.pos_y:self.pos_y+h, self.pos_x:self.pos_x+w]
        
        # Find pixels with non-zero alpha
        alpha_mask = overlay[:,:,3] > 0
        
        # Only process if there are visible pixels
        if np.any(alpha_mask):
            # Fast integer-based alpha blending
            for c in range(3):
                # Use integer math (much faster on low-power CPUs like Raspberry Pi)
                # This avoids expensive float conversions
                roi[:,:,c][alpha_mask] = (
                    (overlay[:,:,c][alpha_mask].astype(np.uint16) * overlay[:,:,3][alpha_mask].astype(np.uint16) + 
                     roi[:,:,c][alpha_mask].astype(np.uint16) * (255 - overlay[:,:,3][alpha_mask].astype(np.uint16))) 
                    // 255
                ).astype(np.uint8)
        
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

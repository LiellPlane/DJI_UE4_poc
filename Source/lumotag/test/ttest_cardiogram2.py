import cv2
import numpy as np
import math
import random
import time

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

    def update_metrics(self, updates):
        """
        updates: Dictionary mapping metric names to (value, color, pos)
                 - value: Numeric value within the specified value_range.
                 - color: A BGR tuple (e.g. (0, 0, 255) for red).
        
        This method shifts the overlay in the history direction and then, for each provided
        metric update, maps its normalized value to a coordinate along the full value axis,
        and directly sets that pixel in the overlay.
        """
        min_val, max_val = self.value_range
        offset_range = 20
        # Shift the overlay along the chosen axis.
        if self.flow_direction == 0:
            # New data at bottom; shift upward.
            self.overlay[:-1, :, :] = self.overlay[1:, :, :]
            self.overlay[-1, :, :] = 0
            new_edge = self.height - 1
            new_edge = [i for i in range(self.height - 1, self.height - offset_range, -1)]
        elif self.flow_direction == 180:
            # New data at top; shift downward.
            self.overlay[1:, :, :] = self.overlay[:-1, :, :]
            self.overlay[0, :, :] = 0
            new_edge = 0
            new_edge = [i for i in range(0, offset_range, 1)]
        elif self.flow_direction == 90:
            # New data at right; shift left.
            self.overlay[:, :-1, :] = self.overlay[:, 1:, :]
            self.overlay[:, -1, :] = 0
            new_edge = self.width - 1
            new_edge = [i for i in range(self.width - 1, self.width - offset_range, -1)]
        elif self.flow_direction == 270:
            # New data at left; shift right.
            self.overlay[:, 1:, :] = self.overlay[:, :-1, :]
            self.overlay[:, 0, :] = 0
            new_edge = [i for i in range(0, offset_range, 1)]

        # For each metric update, compute the coordinate and set the pixel.
        for metric, (value, color, pos) in updates.items():
            # Normalize the value to a 0..1 scale.
            value = min(max(value, min_val), max_val)
            norm = (value - min_val) / (max_val - min_val)
            if self.flow_direction in (0, 180):
                # Map normalized value to an x-coordinate (using the full width)
                new_x = int(norm * (self.width - 1))
                # Directly set the pixel at (row=new_edge, col=new_x) to the provided color with full opacity.
                # self.overlay[new_edge[pos], new_x] = (color[0], color[1], color[2], 255)
                self.overlay[new_edge[pos], new_x] = (color[0], color[1], color[2], 255)
                # self.overlay[new_edge[pos], 0: new_x] = (color[0], color[1], color[2], 255)
            else:
                # Map normalized value to a y-coordinate (using the full height)
                new_y = int(norm * (self.height - 1))
                # Directly set the pixel at (row=new_y, col=new_edge) to the provided color with full opacity.
                self.overlay[new_y, new_edge[pos]] = (color[0], color[1], color[2], 255)

    def get_overlay_with_gradient(self):
        """
        Returns a copy of the overlay with a fade gradient applied along the history flow axis.
        New data is fully opaque while older data fades to transparent.
        """
        overlay_copy = self.overlay.copy()
        if self.flow_direction in (0, 180):
            if self.flow_direction == 0:
                gradient = np.linspace(0, 1, self.height).reshape(self.height, 1)
            else:
                gradient = np.linspace(1, 0, self.height).reshape(self.height, 1)
            alpha_float = overlay_copy[:, :, 3].astype(np.float32)
            overlay_copy[:, :, 3] = (alpha_float * gradient).astype(np.uint8)
        else:
            if self.flow_direction == 90:
                gradient = np.linspace(0, 1, self.width).reshape(1, self.width)
            else:
                gradient = np.linspace(1, 0, self.width).reshape(1, self.width)
            alpha_float = overlay_copy[:, :, 3].astype(np.float32)
            overlay_copy[:, :, 3] = (alpha_float * gradient).astype(np.uint8)

        if self.flow_direction == 0:
            overlay_copy[:, :, 3] = (alpha_float * gradient).astype(np.uint8)
        elif self.flow_direction == 90:
            overlay_copy[:, :, 3] = (alpha_float * gradient).astype(np.uint8)
        elif self.flow_direction == 90:
            overlay_copy[:, :, 3] = (alpha_float * gradient).astype(np.uint8)
        elif self.flow_direction == 90:
            overlay_copy[:, :, 3] = (alpha_float * gradient).astype(np.uint8)
        else:
            raise ValueError(f"Invalid flow direction: {self.flow_direction}")


        # Apply a blur to the overlay
        # overlay_copy = cv2.blur(overlay_copy, (1, 1))  # Simple box blur with a 5x5 kernel
        # Alternatively, use GaussianBlur for a smoother effect
        # overlay_copy = cv2.GaussianBlur(overlay_copy, (5, 5), 0)

        return overlay_copy

    def composite_onto_inplace(self, background):
        """
        Composites the (gradient-adjusted) overlay directly onto the given background image,
        modifying the background in place.
        """
        overlay = self.get_overlay_with_gradient()
        h, w = self.height, self.width
        roi = background[self.pos_y:self.pos_y+h, self.pos_x:self.pos_x+w]
        roi_float = roi.astype(float)
        overlay_bgr = overlay[:, :, :3].astype(float)
        overlay_alpha = overlay[:, :, 3].astype(float) / 255.0

        for c in range(3):
            roi_float[:, :, c] = overlay_bgr[:, :, c] * overlay_alpha + roi_float[:, :, c] * (1 - overlay_alpha)
        background[self.pos_y:self.pos_y+h, self.pos_x:self.pos_x+w] = roi_float.astype(np.uint8)
        return background

    def composite_onto(self, background):
        """
        Composites the (gradient-adjusted) overlay onto a copy of the background image.
        """
        overlay = self.get_overlay_with_gradient()
        output = background.copy()
        h, w = self.height, self.width
        roi = output[self.pos_y:self.pos_y+h, self.pos_x:self.pos_x+w]
        overlay_bgr = overlay[:, :, :3].astype(float)
        overlay_alpha = overlay[:, :, 3].astype(float) / 255.0
        roi = roi.astype(float)
        for c in range(3):
            roi[:, :, c] = overlay_bgr[:, :, c] * overlay_alpha + roi[:, :, c] * (1 - overlay_alpha)
        output[self.pos_y:self.pos_y+h, self.pos_x:self.pos_x+w] = roi.astype(np.uint8)
        return output

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
    flow_direction = 0  # For example, 0°: new data appears at the bottom.

    # Create an instance of the display.
    display = CardioGramDisplay(disp_pos_x, disp_pos_y, disp_width, disp_height,
                                value_range=(0, 50), flow_direction=flow_direction)

    t = 0.0
    dt = 0.05

    cv2.namedWindow("CardioGram Test", cv2.WINDOW_AUTOSIZE)
    while True:
        # Build a dictionary of metric updates.
        # Each update provides a value and a color (BGR) for that metric.
        updates = {}
        # Metric "A" (red) is available from the start.
        valueA = 25 + 25 * math.sin(t * 1.0) + random.uniform(-2, 2)
        updates["A"] = (max(0, min(50, valueA)), (0, 0, 255), 0)
        # Metric "B" (green) starts after 1 second.
        if t > 1:
            valueB = 25 + 25 * math.sin(t * 1.2 + math.pi/4) + random.uniform(-2, 2)
            updates["B"] = (max(0, min(50, valueB)), (0, 255, 0), 5)
        # Metric "C" (blue) starts after 2 seconds.
        if t > 2:
            valueC = 25 + 25 * math.sin(t * 0.8 + math.pi/2) + random.uniform(-2, 2)
            updates["C"] = (max(0, min(50, valueC)), (255, 0, 0), 8)

        # Update the display with the provided metric updates.
        display.update_metrics(updates)
        # Composite the overlay (with fade gradient) onto a copy of the background.
        output = display.composite_onto_inplace(background.copy())
        cv2.imshow("CardioGram Test", output)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break
        t += dt

    cv2.destroyAllWindows()

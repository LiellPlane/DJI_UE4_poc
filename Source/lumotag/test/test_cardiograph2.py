import cv2
import numpy as np
import math
import colorsys

class CardioGraph:
    def __init__(self, canvas_pos, canvas_size, metric_names=None,
                 value_range=(0, 100), fade=0.95, vertical_scale=1.0,
                 scroll_direction='rtl'):
        """
        canvas_pos: (x, y) tuple for the top-left corner of the canvas within a background.
        canvas_size: (width, height) of the canvas where the graph is drawn.
        metric_names: list of metric names. Defaults to 3 metrics if None.
        value_range: (min, max) expected measurement values.
        fade: factor to fade the canvas each update (closer to 1 => slower fade).
        vertical_scale: multiplier applied to the value amplitude (pre‑calculated).
        scroll_direction: 'rtl' (right‑to‑left, new data appears on right) or
                          'ltr' (left‑to‑right, new data appears on left).
        """
        self.canvas_x, self.canvas_y = canvas_pos
        self.canvas_width, self.canvas_height = canvas_size
        self.value_range = value_range
        self.fade = fade
        self.vertical_scale = vertical_scale
        self.scroll_direction = scroll_direction.lower()

        # Create the canvas (BGR, black background).
        self.img = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)

        # Default metric names.
        if metric_names is None:
            self.metric_names = [f"Metric {i+1}" for i in range(3)]
        else:
            self.metric_names = metric_names
        self.n_metrics = len(self.metric_names)

        # Pre-calculate the mapping parameters:
        min_val, max_val = self.value_range
        self.mid = (min_val + max_val) / 2.0
        # We want the full range (max_val - min_val) to span the canvas height.
        # Since the mid maps to canvas_height/2, scale factor is:
        self.scale_factor = vertical_scale * (self.canvas_height / (max_val - min_val))

        # Precompute a color for each metric (rainbow-ish).
        self.colors = []
        for i in range(self.n_metrics):
            hue = i / self.n_metrics  # hue from 0 to 1.
            rgb = colorsys.hsv_to_rgb(hue, 1, 1)
            # Convert to BGR (0-255)
            bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
            self.colors.append(bgr)

        # Store the last y positions for each metric.
        self.last_points = [None] * self.n_metrics

    def value_to_y(self, value):
        """
        Map a measurement value to a y-coordinate on the canvas.
        The mapping centers the value_range so that self.mid maps to canvas_height/2.
        """
        y = int(self.canvas_height/2 - (value - self.mid) * self.scale_factor)
        return max(0, min(self.canvas_height - 1, y))

    def update(self, measurements):
        """
        Update the canvas with new measurements (one per metric).
        Shifts the canvas (either left or right), applies a fade, and draws new data.
        """
        # Shift the canvas based on scroll direction.
        if self.scroll_direction == 'rtl':
            # Shift left: copy all columns except the first one to the left.
            self.img[:, :-1] = self.img[:, 1:]
            self.img[:, -1] = 0  # clear the new rightmost column
            new_x, prev_x = self.canvas_width - 1, self.canvas_width - 2
        elif self.scroll_direction == 'ltr':
            # Shift right: copy all columns except the last one to the right.
            self.img[:, 1:] = self.img[:, :-1]
            self.img[:, 0] = 0  # clear the new leftmost column
            new_x, prev_x = 0, 1
        else:
            raise ValueError("scroll_direction must be either 'rtl' or 'ltr'.")

        # Apply fade (this operates only on the small canvas).
        self.img = (self.img.astype(np.float32) * self.fade).astype(np.uint8)

        # For each metric, convert the measurement to a y-coordinate and draw.
        for i, measurement in enumerate(measurements):
            y = self.value_to_y(measurement)
            if self.last_points[i] is not None:
                cv2.line(self.img, (prev_x, self.last_points[i]), (new_x, y),
                         self.colors[i], thickness=2)
            else:
                self.img[y, new_x] = self.colors[i]
            self.last_points[i] = y

    def get_canvas(self):
        """Return the current canvas image."""
        return self.img

    def reset(self):
        """Clear the canvas and reset metric history."""
        self.img.fill(0)
        self.last_points = [None] * self.n_metrics

def overlay_canvas(background, canvas, canvas_pos, mode='mask', alpha=0.5, threshold=10):
    """
    Overlay a small canvas onto the background image at canvas_pos.
    
    background: The target image.
    canvas: The small canvas image from CardioGraph.
    canvas_pos: (x, y) tuple for where to place the top-left of the canvas in the background.
    mode: 'blend' for simple alpha blending; 'mask' to overlay only non-black pixels.
    alpha: Blending factor (only for 'blend' mode).
    threshold: Threshold to determine non-black pixels (only for 'mask' mode).
    """
    x, y = canvas_pos
    ch, cw = canvas.shape[:2]
    # Get a reference to the region of interest in the background.
    roi = background[y:y+ch, x:x+cw]
    if mode == 'blend':
        blended = cv2.addWeighted(roi, 1.0, canvas, alpha, 0)
        background[y:y+ch, x:x+cw] = blended
    elif mode == 'mask':
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)
        canvas_part = cv2.bitwise_and(canvas, canvas, mask=mask)
        background[y:y+ch, x:x+cw] = cv2.add(bg_part, canvas_part)
    else:
        raise ValueError("mode must be either 'blend' or 'mask'.")
    return background

# Test mode: simulate sine waves for the graph and overlay the canvas onto a background.
if __name__ == '__main__':
    # Create an arbitrary background (for example, 800x600 with a color gradient).
    bg_height, bg_width = 600, 800
    background = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
    for i in range(bg_height):
        color = int(255 * (i / bg_height))
        background[i, :] = (color, color, color)
    
    # Define the canvas parameters.
    canvas_size = (300, 150)        # width, height of the graph area
    # For bottom left: let’s say x=20, and y such that the canvas sits at the bottom.
    canvas_pos = (20, bg_height - canvas_size[1])
    
    # Create the CardioGraph (using a small canvas).
    # Change scroll_direction to 'ltr' or 'rtl' as desired.
    graph = CardioGraph(canvas_pos=canvas_pos, canvas_size=canvas_size,
                        metric_names=["Metric 1", "Metric 2", "Metric 3"],
                        value_range=(-1, 1), fade=0.98,
                        vertical_scale=1.0, scroll_direction='rtl')
    
    t = 0.0
    dt = 0.05  # time step between frames
    overlay_mode = 'mask'  # or 'blend'
    alpha_value = 0.6      # used only in 'blend' mode

    while True:
        # Copy the background so we can overlay the canvas each frame.
        frame = background.copy()
        
        # Generate sample sine wave measurements.
        m1 = math.sin(t)
        m2 = math.sin(2 * t + math.pi / 4)
        m3 = math.sin(0.5 * t + math.pi / 2)
        measurements = [m1, m2, m3]

        # Update the graph (only the small canvas is updated).
        graph.update(measurements)
        canvas_img = graph.get_canvas()

        # Overlay the canvas onto the frame at the desired position.
        combined = overlay_canvas(frame, canvas_img, canvas_pos,
                                  mode=overlay_mode, alpha=alpha_value)
        
        cv2.imshow("Background with CardioGraph", combined)
        if cv2.waitKey(20) == 27:  # press ESC to exit
            break
        t += dt

    cv2.destroyAllWindows()

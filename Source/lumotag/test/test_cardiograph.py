import cv2
import numpy as np
import math
import colorsys

class CardioGraph:
    def __init__(self, width=800, height=600, metric_names=None, value_range=(0, 100),
                 fade=0.95, vertical_scale=1.0, scroll_direction='rtl'):
        """
        width, height: dimensions of the display image.
        metric_names: list of metric names. If None, defaults to 3 metrics.
        value_range: tuple (min, max) that defines the expected range of measurements.
        fade: factor by which the image is faded each update (closer to 1 = slower fade).
        vertical_scale: multiplier for the amplitude mapping.
          For a centered mapping, the measurements are scaled so that the midpoint maps to height/2.
        scroll_direction: 'rtl' for right-to-left scrolling (new data on right, history shifts left)
                          'ltr' for left-to-right scrolling (new data on left, history shifts right)
        """
        self.width = width
        self.height = height
        self.value_range = value_range
        self.fade = fade
        self.vertical_scale = vertical_scale
        self.scroll_direction = scroll_direction.lower()

        # If no metric names provided, assume 3 metrics.
        if metric_names is None:
            self.metric_names = [f"Metric {i+1}" for i in range(3)]
        else:
            self.metric_names = metric_names
        self.n_metrics = len(self.metric_names)

        # Create a blank image (BGR, black background).
        self.img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Precompute mapping parameters:
        # We map measurements linearly so that the mid-point of value_range maps to height/2.
        min_val, max_val = self.value_range
        self.mid = (min_val + max_val) / 2.0
        # The factor to scale the difference (value - mid) so that the full range fits in half the height.
        # Without vertical_scale, (max_val - mid) should map to height/2.
        self.scale_factor = vertical_scale * (self.height / (max_val - min_val))

        # Generate a rainbow-esque color for each metric.
        self.colors = []
        for i in range(self.n_metrics):
            hue = i / self.n_metrics  # Hue between 0 and 1.
            rgb = colorsys.hsv_to_rgb(hue, 1, 1)
            # Convert to BGR with values 0-255 (OpenCV uses BGR)
            bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
            self.colors.append(bgr)

        # Store the last drawn y coordinate for each metric.
        # This will be used to draw continuous lines.
        self.last_points = [None] * self.n_metrics

    def value_to_y(self, value):
        """
        Map the measurement value to a y-coordinate.
        The mapping is centered so that self.mid maps to height/2.
        """
        # y = height/2 - (value - mid)*scale_factor
        y = int(self.height/2 - (value - self.mid) * self.scale_factor)
        # Clamp y within image bounds.
        return max(0, min(self.height - 1, y))

    def update(self, measurements):
        """
        Update the graph with a new list of measurements (one per metric).
        Depending on scroll_direction, shifts the image and draws new data at the edge.
        """
        if self.scroll_direction == 'rtl':
            # Right-to-left: shift the image left and add new data at the rightmost column.
            self.img[:, :-1] = self.img[:, 1:]
            self.img[:, -1] = 0  # Clear the new rightmost column.
            new_x = self.width - 1
            prev_x = self.width - 2
        elif self.scroll_direction == 'ltr':
            # Left-to-right: shift the image right and add new data at the leftmost column.
            self.img[:, 1:] = self.img[:, :-1]
            self.img[:, 0] = 0  # Clear the new leftmost column.
            new_x = 0
            prev_x = 1
        else:
            raise ValueError("scroll_direction must be either 'rtl' or 'ltr'.")

        # Apply fade to the entire image.
        self.img = (self.img.astype(np.float32) * self.fade).astype(np.uint8)

        # For each metric, compute y and draw the new segment.
        for i, measurement in enumerate(measurements):
            y = self.value_to_y(measurement)
            if self.last_points[i] is not None:
                # Draw a line from the previous point (shifted) to the new measurement.
                cv2.line(self.img, (prev_x, self.last_points[i]), (new_x, y),
                         self.colors[i], thickness=2)
            else:
                # For the very first point, just draw a dot.
                self.img[y, new_x] = self.colors[i]
            # Update last point for next iteration.
            self.last_points[i] = y

    def _get_image(self):
        """Return the current graph image."""
        return self.img

    def reset(self):
        """Clear the image and reset historical points."""
        self.img.fill(0)
        self.last_points = [None] * self.n_metrics

def overlay_on_background(background, overlay, mode='mask', alpha=0.5, threshold=10):
    """
    Overlay the graph (overlay) on top of the background image.
    
    Parameters:
    - background: The background image.
    - overlay: The CardioGraph image (should be the same dimensions as background).
    - mode: 'blend' uses alpha blending over the entire image.
            'mask' overlays only the non-black parts from overlay.
    - alpha: The blending factor for 'blend' mode.
    - threshold: Threshold to determine non-black pixels in 'mask' mode.
    """
    if mode == 'blend':
        return cv2.addWeighted(background, 1.0, overlay, alpha, 0)
    elif mode == 'mask':
        gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        background_part = cv2.bitwise_and(background, background, mask=mask_inv)
        overlay_part = cv2.bitwise_and(overlay, overlay, mask=mask)
        return cv2.add(background_part, overlay_part)
    else:
        raise ValueError("Mode must be either 'blend' or 'mask'.")

# Test mode: simulate sin waves for each metric and overlay on a background image.
if __name__ == '__main__':
    # Create a background image (for example, a simple gradient).
    bg_height, bg_width = 400, 800
    background = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
    for i in range(bg_height):
        color = int(255 * (i / bg_height))
        background[i, :] = (color, color, color)
    
    # Initialize CardioGraph with desired dimensions and parameters.
    # Change scroll_direction to 'ltr' for left-to-right scrolling.
    graph = CardioGraph(width=bg_width, height=bg_height,
                        metric_names=["Metric 1", "Metric 2", "Metric 3"],
                        value_range=(-1, 1), fade=0.98,
                        vertical_scale=1.0, scroll_direction='ltr')
    
    t = 0.0
    dt = 0.05  # time step between frames
    overlay_mode = 'mask'  # Choose 'mask' or 'blend'
    alpha_value = 0.6     # Only used in blend mode.
    
    while True:
        # Generate three sine waves with different frequencies and phases.
        m1 = math.sin(t)
        m2 = math.sin(2 * t + math.pi / 4)
        m3 = math.sin(0.5 * t + math.pi / 2)
        measurements = [m1, m2, m3]

        graph.update(measurements)
        graph_img = graph.get_image()
        combined = overlay_on_background(background, graph_img,
                                           mode=overlay_mode, alpha=alpha_value)
        cv2.imshow("CardioGraph Overlay", combined)
        if cv2.waitKey(20) == 27:  # ESC key to exit
            break
        t += dt

    cv2.destroyAllWindows()

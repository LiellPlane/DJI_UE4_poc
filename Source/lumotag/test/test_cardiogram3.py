import cv2
import numpy as np
import math
import colorsys
import time

class Cardiogram:
    """
    A class to display metrics in a cardiogram-like visualization that updates in real-time.
    Displays current metric values as bars and historical values flowing in a specified direction.
    """
    
    def __init__(self, x, y, width, height, direction=0, history_length=100, alpha=0.7):
        """
        Initialize the cardiogram display.
        
        Args:
            x: X position of the ROI
            y: Y position of the ROI
            width: Width of the ROI
            height: Height of the ROI
            direction: Direction of history flow in degrees (0, 90, 180, 270)
            history_length: Number of historical points to keep
            alpha: Opacity of overlay (0-1)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.direction = direction
        self.history_length = history_length
        self.alpha = alpha
        
        # Validate direction
        if direction not in [0, 90, 180, 270]:
            raise ValueError("Direction must be one of: 0, 90, 180, 270")
        
        # Initialize metric storage
        self.metrics = {}  # {metric_name: [current_value, [history_values]]}
        self.metric_colors = {}  # {metric_name: color}
        
        # Create buffer for rendering
        self.buffer = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Direction-specific settings
        if direction in [0, 180]:  # Horizontal
            self.bar_thickness = max(1, height // 5)
        else:  # Vertical
            self.bar_thickness = max(1, width // 5)
    
    def _get_color_for_metric(self, metric_name):
        """Generate a consistent, attractive color for a metric."""
        if metric_name not in self.metric_colors:
            # Use the hash of the metric name to generate a consistent hue
            hue = (hash(metric_name) % 1000) / 1000.0
            # Convert HSV to RGB (using full saturation and value)
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            # Convert to BGR for OpenCV (0-255 range)
            color = tuple(int(c * 255) for c in reversed(rgb))  # Reversed for BGR
            self.metric_colors[metric_name] = color
        
        return self.metric_colors[metric_name]
    
    def update(self, metrics_values):
        """
        Update the metrics with new values.
        
        Args:
            metrics_values: List of tuples (metric_name, value)
        """
        # Process each metric
        for metric_name, value in metrics_values:
            # Normalize value to 0-1 range for display
            # Assumes values are typically between 0-1, adjust as needed
            normalized_value = max(0, min(1, float(value)))  # Ensure float conversion
            
            # Initialize metric if not exists
            if metric_name not in self.metrics:
                self.metrics[metric_name] = [normalized_value, []]
                self._get_color_for_metric(metric_name)  # Ensure color is assigned
            else:
                # Update current value and add to history
                current_history = self.metrics[metric_name][1]
                current_history.insert(0, normalized_value)
                # Trim history to max length
                if len(current_history) > self.history_length:
                    current_history = current_history[:self.history_length]
                
                self.metrics[metric_name] = [normalized_value, current_history]
    
    def render(self, frame):
        """
        Render the cardiogram onto the provided frame.
        
        Args:
            frame: The image to render onto (modified in-place in the ROI)
        """
        # Clear buffer with transparency
        self.buffer.fill(0)
        
        # Skip if no metrics
        if not self.metrics:
            return
        
        # Safety check to ensure ROI is within frame boundaries
        if self.y + self.height > frame.shape[0] or self.x + self.width > frame.shape[1]:
            print(f"Warning: Cardiogram ROI extends beyond frame boundaries. Adjusting...")
            actual_height = min(self.height, frame.shape[0] - self.y)
            actual_width = min(self.width, frame.shape[1] - self.x)
            if actual_height <= 0 or actual_width <= 0:
                print("Error: Cardiogram ROI is outside frame boundaries")
                return
            # Resize buffer if needed
            if actual_height != self.height or actual_width != self.width:
                self.buffer = np.zeros((actual_height, actual_width, 4), dtype=np.uint8)
                self.height = actual_height
                self.width = actual_width
        
        # Calculate dimensions based on metrics count
        num_metrics = len(self.metrics)
        if num_metrics == 0:
            return
        
        # Constant for bar thickness
        BAR_THICKNESS = 2  # 2 pixels as requested
        
        # Rendering based on direction
        if self.direction == 0:  # Right direction, history flows upward
            # Calculate metrics spacing
            metrics_spacing = self.width // num_metrics
            
            for i, (metric_name, (current_value, history)) in enumerate(self.metrics.items()):
                color = self._get_color_for_metric(metric_name)
                
                # Calculate center position for this metric
                center_x = (i + 0.5) * metrics_spacing
                
                # Bar graph is horizontal at the bottom
                bar_length = int(current_value * metrics_spacing * 0.8)  # 80% of available space
                bar_y = self.height - BAR_THICKNESS - 1
                
                # Draw horizontal bar
                left_x = int(center_x - bar_length / 2)
                right_x = int(center_x + bar_length / 2)
                cv2.rectangle(self.buffer, 
                             (max(0, left_x), bar_y), 
                             (min(self.width-1, right_x), bar_y + BAR_THICKNESS), 
                             (*color, 255), -1)
                
                # Draw history flowing upward
                for j, hist_val in enumerate(history):
                    if j >= bar_y:  # Don't go beyond the top
                        break
                    
                    # Calculate alpha for fade out
                    alpha = 255 * (1 - j / max(1, len(history)))
                    
                    # Calculate line length based on value
                    hist_length = int(hist_val * metrics_spacing * 0.6)  # 60% of available space
                    
                    # Draw a line that flows upward
                    y_pos = bar_y - j - 1
                    left_hist_x = int(center_x - hist_length / 2)
                    right_hist_x = int(center_x + hist_length / 2)
                    
                    if 0 <= y_pos < self.height:
                        cv2.line(self.buffer, 
                                (max(0, left_hist_x), y_pos), 
                                (min(self.width-1, right_hist_x), y_pos),
                                (*color, int(alpha)), 1)
        
        elif self.direction == 180:  # Left direction, history flows downward
            # Calculate metrics spacing
            metrics_spacing = self.width // num_metrics
            
            for i, (metric_name, (current_value, history)) in enumerate(self.metrics.items()):
                color = self._get_color_for_metric(metric_name)
                
                # Calculate center position for this metric
                center_x = (i + 0.5) * metrics_spacing
                
                # Bar graph is horizontal at the top
                bar_length = int(current_value * metrics_spacing * 0.8)
                bar_y = BAR_THICKNESS
                
                # Draw horizontal bar
                left_x = int(center_x - bar_length / 2)
                right_x = int(center_x + bar_length / 2)
                cv2.rectangle(self.buffer, 
                             (max(0, left_x), 0), 
                             (min(self.width-1, right_x), bar_y), 
                             (*color, 255), -1)
                
                # Draw history flowing downward
                for j, hist_val in enumerate(history):
                    if j >= self.height - bar_y:  # Don't go beyond the bottom
                        break
                    
                    # Calculate alpha for fade out
                    alpha = 255 * (1 - j / max(1, len(history)))
                    
                    # Calculate line length based on value
                    hist_length = int(hist_val * metrics_spacing * 0.6)
                    
                    # Draw a line that flows downward
                    y_pos = bar_y + j + 1
                    left_hist_x = int(center_x - hist_length / 2)
                    right_hist_x = int(center_x + hist_length / 2)
                    
                    if 0 <= y_pos < self.height:
                        cv2.line(self.buffer, 
                                (max(0, left_hist_x), y_pos), 
                                (min(self.width-1, right_hist_x), y_pos),
                                (*color, int(alpha)), 1)
        
        elif self.direction == 90:  # Up direction, history flows leftward
            # Calculate metrics spacing
            metrics_spacing = self.height // num_metrics
            
            for i, (metric_name, (current_value, history)) in enumerate(self.metrics.items()):
                color = self._get_color_for_metric(metric_name)
                
                # Calculate center position for this metric
                center_y = (i + 0.5) * metrics_spacing
                
                # Bar graph is vertical at the right
                bar_length = int(current_value * metrics_spacing * 0.8)
                bar_x = self.width - BAR_THICKNESS - 1
                
                # Draw vertical bar
                top_y = int(center_y - bar_length / 2)
                bottom_y = int(center_y + bar_length / 2)
                cv2.rectangle(self.buffer, 
                             (bar_x, max(0, top_y)), 
                             (bar_x + BAR_THICKNESS, min(self.height-1, bottom_y)), 
                             (*color, 255), -1)
                
                # Draw history flowing leftward
                for j, hist_val in enumerate(history):
                    if j >= bar_x:  # Don't go beyond the left edge
                        break
                    
                    # Calculate alpha for fade out
                    alpha = 255 * (1 - j / max(1, len(history)))
                    
                    # Calculate line height based on value
                    hist_length = int(hist_val * metrics_spacing * 0.6)
                    
                    # Draw a line that flows leftward
                    x_pos = bar_x - j - 1
                    top_hist_y = int(center_y - hist_length / 2)
                    bottom_hist_y = int(center_y + hist_length / 2)
                    
                    if 0 <= x_pos < self.width:
                        cv2.line(self.buffer, 
                                (x_pos, max(0, top_hist_y)), 
                                (x_pos, min(self.height-1, bottom_hist_y)),
                                (*color, int(alpha)), 1)
        
        elif self.direction == 270:  # Down direction, history flows rightward
            # Calculate metrics spacing
            metrics_spacing = self.height // num_metrics
            
            for i, (metric_name, (current_value, history)) in enumerate(self.metrics.items()):
                color = self._get_color_for_metric(metric_name)
                
                # Calculate center position for this metric
                center_y = (i + 0.5) * metrics_spacing
                
                # Bar graph is vertical at the left
                bar_length = int(current_value * metrics_spacing * 0.8)
                bar_x = BAR_THICKNESS
                
                # Draw vertical bar
                top_y = int(center_y - bar_length / 2)
                bottom_y = int(center_y + bar_length / 2)
                cv2.rectangle(self.buffer, 
                             (0, max(0, top_y)), 
                             (bar_x, min(self.height-1, bottom_y)), 
                             (*color, 255), -1)
                
                # Draw history flowing rightward
                for j, hist_val in enumerate(history):
                    if j >= self.width - bar_x:  # Don't go beyond the right edge
                        break
                    
                    # Calculate alpha for fade out
                    alpha = 255 * (1 - j / max(1, len(history)))
                    
                    # Calculate line height based on value
                    hist_length = int(hist_val * metrics_spacing * 0.6)
                    
                    # Draw a line that flows rightward
                    x_pos = bar_x + j + 1
                    top_hist_y = int(center_y - hist_length / 2)
                    bottom_hist_y = int(center_y + hist_length / 2)
                    
                    if 0 <= x_pos < self.width:
                        cv2.line(self.buffer, 
                                (x_pos, max(0, top_hist_y)), 
                                (x_pos, min(self.height-1, bottom_hist_y)),
                                (*color, int(alpha)), 1)
        
        # Apply buffer to ROI
        roi = frame[self.y:self.y+self.height, self.x:self.x+self.width]
        
        # Create mask from alpha channel
        mask = self.buffer[:, :, 3] / 255.0
        
        # Apply overlay with proper alpha blending (only where mask > 0)
        for c in range(3):  # BGR channels
            try:
                roi[:, :, c] = np.where(
                    mask > 0,
                    roi[:, :, c] * (1 - mask * self.alpha) + self.buffer[:, :, c] * mask * self.alpha,
                    roi[:, :, c]
                )
            except ValueError as e:
                print(f"Error in alpha blending: {e}")
                print(f"ROI shape: {roi.shape}, Buffer shape: {self.buffer.shape}, Mask shape: {mask.shape}")
                return


def test_cardiogram():
    """Test function to demonstrate the Cardiogram class using sine waves."""
    # Create a black image for testing
    width, height = 800, 600
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create 4 cardiograms, one for each direction
    cardio_0 = Cardiogram(50, 50, 300, 150, direction=0)
    cardio_90 = Cardiogram(400, 50, 150, 300, direction=90)
    cardio_180 = Cardiogram(50, 350, 300, 150, direction=180)
    cardio_270 = Cardiogram(400, 350, 150, 300, direction=270)
    
    # Generate 4 sine waves with different phases
    metric_names = ["Sine1", "Sine2", "Sine3", "Sine4"]
    
    # Debug flag to print metric values
    debug_metrics = True
    
    t = 0
    try:
        while True:
            # Clear the frame
            frame.fill(20)  # Dark gray background
            
            # Generate sine wave values (0-1 range)
            metrics = []
            for i, name in enumerate(metric_names):
                # Different frequencies and phases
                frequency = 0.1 + i * 0.05
                phase = i * math.pi / 4
                value = (math.sin(t * frequency + phase) + 1) / 2  # Scale to 0-1
                metrics.append((name, value))
                
                # Debug metrics values
                if debug_metrics and t % 1.0 < 0.05:  # Print every ~1 second
                    print(f"{name}: {value:.3f}")
            
            # Update and render each cardiogram
            for cardio in [cardio_0, cardio_90, cardio_180, cardio_270]:
                cardio.update(metrics)
                cardio.render(frame)
            
            # Add direction labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "Direction: 0°", (50, 40), font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Direction: 90°", (400, 40), font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Direction: 180°", (50, 340), font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Direction: 270°", (400, 340), font, 0.7, (255, 255, 255), 2)
            
            # Display metric values on screen
            y_offset = 20
            for i, (name, value) in enumerate(metrics):
                y_pos = height - y_offset - (i * 20)
                cv2.putText(frame, f"{name}: {value:.3f}", (width - 150, y_pos), 
                           font, 0.5, (200, 200, 200), 1)
            
            # Display
            cv2.imshow("Cardiogram Test", frame)
            
            # Exit on 'q' press
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            
            t += 0.05
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_cardiogram()

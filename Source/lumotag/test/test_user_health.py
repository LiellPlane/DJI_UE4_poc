import cv2
import numpy as np


def create_health_bar(health_value, width=400, height=600, num_segments=10):
    # Create a black canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate segment height
    segment_height = height // num_segments
    segment_width = width - 40  # Leave some margin
    
    # Calculate number of active segments
    active_segments = int(health_value * num_segments)
    
    # Define colours
    high_health_colour = (0, 255, 0)  # Pure Green
    low_health_colour = (0, 0, 255)   # Pure Red
    threshold = 0.3
    
    # Draw segments from bottom to top
    for i in range(num_segments):
        if i < active_segments:
            # Determine colour based on health
            colour = low_health_colour if health_value <= threshold else high_health_colour
            
            # Calculate segment position
            y_start = height - (i + 1) * segment_height
            y_end = height - i * segment_height
            
            # Create segment with gradient
            segment = np.zeros((segment_height, segment_width, 3), dtype=np.uint8)
            # Fill with pure colour first
            cv2.rectangle(segment, (0, 0), (segment_width, segment_height), colour, -1)
            # Add black border
            cv2.rectangle(segment, (0, 0), (segment_width, segment_height), (0, 0, 0), 2)
            # Add yellow highlight line
            cv2.line(segment, (0, int(segment_height * 0.3)), 
                    (segment_width, int(segment_height * 0.3)), (0, 255, 255), 2)
            
            # Add enhanced highlight effect
            highlight = np.zeros_like(segment)
            # Create a much larger highlight area with white for more intensity
            cv2.rectangle(highlight, (0, 0), 
                         (segment_width, int(segment_height * 0.7)), (255, 255, 255), -1)
            # Apply stronger blur for more glow
            highlight = cv2.GaussianBlur(highlight, (0, 0), 60)
            # Increase highlight intensity significantly
            segment = cv2.addWeighted(segment, 2.0, highlight, 0.4, 0)
            
            # Add segment to canvas
            canvas[y_start:y_end, 20:20+segment_width] = segment
    
    # Apply overall blur for smoothness
    canvas = cv2.GaussianBlur(canvas, (0, 0), 3)
    
    return canvas


def main():
    # Create window
    cv2.namedWindow('Health Bar', cv2.WINDOW_NORMAL)
    
    # Test animation
    health_values = np.arange(1.0, -0.01, -0.01)
    
    for health in health_values:
        health_bar = create_health_bar(health)
        cv2.imshow('Health Bar', health_bar)
        
        # Break loop on 'q' press
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 
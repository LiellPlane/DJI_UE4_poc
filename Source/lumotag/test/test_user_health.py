import cv2
import numpy as np
import math


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


def main():
    # Create window
    cv2.namedWindow('Health Bar', cv2.WINDOW_NORMAL)
    
    # Test animation
    health_values = np.arange(1.0, -0.01, -0.1)
    
    for health in health_values:
        print(health)
        health_bar = create_health_bar(
            health,
            use_anti_aliasing=True,
            use_noise=True,
            high_health_color='green',
            width=600,
            height=600)  # You can change to 'green' or 'blue'
        cv2.imshow('Health Bar', cv2.resize(health_bar,(60,60)))
        
        # Wait for any key press to continue
        key = cv2.waitKey(0)
        # Break loop on 'q' press
        if key & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 
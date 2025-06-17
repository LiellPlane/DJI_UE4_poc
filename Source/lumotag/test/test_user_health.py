import cv2
import numpy as np


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
    
    # Calculate segment height and add padding
    padding_top = 40  # pixels of padding at top
    padding_bottom = 40  # pixels of padding at bottom
    padding_sides = 40  # pixels of padding on sides
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
        canvas = apply_bloom_effect(canvas, threshold=100, blur_size=15, intensity=2.0)
    else:
        # Normal bloom for high health
        canvas = apply_bloom_effect(canvas, threshold=100, blur_size=10, intensity=1.8)
    
    return canvas


def main():
    # Create window
    cv2.namedWindow('Health Bar', cv2.WINDOW_NORMAL)
    
    # Test animation
    health_values = np.arange(1.0, -0.01, -0.1)
    
    for health in health_values:
        print(health)
        health_bar = create_health_bar(health, use_anti_aliasing=True, use_noise=True, high_health_color='green')  # You can change to 'green' or 'blue'
        cv2.imshow('Health Bar', cv2.resize(health_bar,(80,80)))
        
        # Wait for any key press to continue
        key = cv2.waitKey(0)
        # Break loop on 'q' press
        if key & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 
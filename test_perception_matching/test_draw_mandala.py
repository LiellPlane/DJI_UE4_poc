import cv2
import numpy as np
import math

def draw_ring(img, center, inner_radius, outer_radius, color):
    """Draw a ring using scanline fill approach"""
    x0, y0 = center
    for y in range(int(y0 - outer_radius), int(y0 + outer_radius + 1)):
        for x in range(int(x0 - outer_radius), int(x0 + outer_radius + 1)):
            # Calculate distance from center
            dx = x - x0
            dy = y - y0
            distance = math.sqrt(dx*dx + dy*dy)
            
            # If pixel is within the ring
            if inner_radius <= distance <= outer_radius:
                img[y, x] = color

def draw_concentric_circles(image_size=200, num_circles=20):
    # Create a white image
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    
    # Get center coordinates
    center = (image_size // 2, image_size // 2)
    
    # Draw single pixel center dot
    img[center[1], center[0]] = (0, 0, 0)
    
    # Define 5 different colors
    colors = [
        (0, 0, 0),      # Black
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        
        (128, 0, 128)   # Purple
    ]
    
    # Draw concentric circles starting from radius 1 (just outside the center dot)
    for i in range(num_circles):
        # Each ring is exactly 1 pixel thick
        inner_radius = 1 + i  # Start just outside the center dot
        outer_radius = inner_radius + 1
        # Cycle through the 5 colors
        color = colors[i % len(colors)]
        draw_ring(img, center, inner_radius, outer_radius, color)
    
    return img

def main():
    # Create the image with concentric circles
    img = draw_concentric_circles()
    
    # Zoom the image by 4x
    zoomed_img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    
    # Display the image
    cv2.imshow('Concentric Circles', zoomed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the image
    cv2.imwrite('concentric_circles.png', zoomed_img)

if __name__ == "__main__":
    main()

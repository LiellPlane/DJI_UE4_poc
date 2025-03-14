import os
import random
from PIL import Image
import numpy as np

def generate_random_colour_images(num_images, output_folder):
    """
    Generate a specified number of images with random colours and sizes.
    
    Args:
        num_images (int): Number of images to generate
        output_folder (str): Path to the folder where images will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Reasonable size ranges for images (in pixels)
    min_width, max_width = 100, 800
    min_height, max_height = 100, 600
    
    for i in range(num_images):
        print(f"Generating image {i+1} of {num_images}")
        # Generate random dimensions
        width = random.randint(min_width, max_width)
        height = random.randint(min_height, max_height)
        
        # Generate random RGB colour
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        # Create a solid colour image
        img_array = np.full((height, width, 3), [r, g, b], dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save the image with colour values in the filename
        filename = f"random_colour_r{r:03d}_g{g:03d}_b{b:03d}_{i+1:04d}.jpg"
        img.save(os.path.join(output_folder, filename))
        
    print(f"Generated {num_images} random colour images in {output_folder}")

# Example usage (uncomment to run directly)
if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output folder path by appending the folder name to script directory
    output_folder = os.path.join(script_dir, "test_images_colour_seq")
    
    generate_random_colour_images(100, output_folder)

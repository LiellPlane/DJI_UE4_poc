import os
import random
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter



def generate_random_colour_images(num_images, output_folder, add_noise=False, noise_percentage=0):
    """
    Generate a specified number of images with random colours and sizes.
    
    Args:
        num_images (int): Number of images to generate
        output_folder (str): Path to the folder where images will be saved
        add_noise (bool): Whether to add color noise to the images
        noise_percentage (float): Amount of noise to add (0-100%)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Reasonable size ranges for images (in pixels)
    min_width, max_width = 50, 50
    min_height, max_height = 50, 50
    
    for i in range(num_images):
        print(f"Generating image {i+1} of {num_images}")
        # Generate random dimensions
        width = random.randint(min_width, max_width)
        height = random.randint(min_height, max_height)
        
        # Generate random RGB colour
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        # Create base color array
        img_array = np.full((height, width, 3), [r, g, b], dtype=np.uint8)
        
        # Add noise if specified
        if add_noise and noise_percentage > 0:
            # Calculate max noise value based on percentage (0-255)
            max_noise = int(255 * noise_percentage / 100)
            
            # Generate random noise for each pixel and color channel
            noise = np.random.randint(-max_noise, max_noise + 1, size=(height, width, 3))
            
            # Add noise to the image and clip values to valid range [0, 255]
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        
        # Save the image with colour values in the filename
        filename = f"random_colour_r{r:03d}_g{g:03d}_b{b:03d}_{i+1:04d}.jpg"
        if add_noise:
            filename = f"random_colour_r{r:03d}_g{g:03d}_b{b:03d}_noise{int(noise_percentage)}_{i+1:04d}.jpg"
        
        img.save(os.path.join(output_folder, filename))
        
    print(f"Generated {num_images} random colour images in {output_folder}")

def generate_random_colour_blobs(num_images, output_folder, add_noise=False, noise_percentage=0):
    """
    Generate a specified number of images with random colored blobs (3-7 blobs per image).
    
    Args:
        num_images (int): Number of images to generate
        output_folder (str): Path to the folder where images will be saved
        add_noise (bool): Whether to add color noise to the images
        noise_percentage (float): Amount of noise to add (0-100%)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Fixed size for images (in pixels)
    width, height = 50, 50
    
    for i in range(num_images):
        print(f"Generating blob image {i+1} of {num_images}")
        
        # Create empty image array
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate random number of blobs (3-7)
        num_blobs = random.randint(3, 7)
        
        # Generate vibrant colors for each blob
        blob_colors = []
        for _ in range(num_blobs):
            # Generate a vibrant color by ensuring at least one channel is near max
            channels = np.random.randint(0, 255, size=3)
            # Randomly select one channel to be near maximum
            max_channel = random.randint(0, 2)
            channels[max_channel] = random.randint(200, 255)
            # Ensure at least one other channel is not too low
            other_channels = [i for i in range(3) if i != max_channel]
            channels[random.choice(other_channels)] = random.randint(50, 255)
            blob_colors.append(channels)
        blob_colors = np.array(blob_colors)
        
        # Generate random centers and sizes for blobs
        centers = np.random.randint(0, [width, height], size=(num_blobs, 2))
        sizes = np.random.randint(10, 25, size=num_blobs)  # Random sizes for each blob
        
        # Create distance field for each blob
        for y in range(height):
            for x in range(width):
                # Calculate distance to each blob center
                distances = np.sqrt((x - centers[:, 0])**2 + (y - centers[:, 1])**2)
                
                # Find the closest blob
                closest_blob = np.argmin(distances)
                
                # Only color the pixel if it's within the blob's size
                if distances[closest_blob] < sizes[closest_blob]:
                    img_array[y, x] = blob_colors[closest_blob]
        
        # Add noise if specified
        if add_noise and noise_percentage > 0:
            max_noise = int(255 * noise_percentage / 100)
            noise = np.random.randint(-max_noise, max_noise + 1, size=(height, width, 3))
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        
        # Save the image with number of blobs in the filename
        filename = f"random_blobs_{num_blobs}_blobs_{i+1:04d}.jpg"
        if add_noise:
            filename = f"random_blobs_{num_blobs}_blobs_noise{int(noise_percentage)}_{i+1:04d}.jpg"
        
        img.save(os.path.join(output_folder, filename))
        
    print(f"Generated {num_images} random blob images in {output_folder}")

# Example usage (uncomment to run directly)
if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output folder path by appending the folder name to script directory
    output_folder = os.path.join(script_dir, "test_images_colour_seq")
    
    # Generate both types of images
    generate_random_colour_images(5000, output_folder, add_noise=True, noise_percentage=10)
    # generate_random_colour_blobs(5000, output_folder, add_noise=True, noise_percentage=10)

import os
import random
import shutil
import platform
from pathlib import Path



if platform.system() == "Windows":
    DESTINATION_FOLDER = Path("D:/match_images")
    INPUT_FOLDER = Path("D:/temp_match_imgs\matchable\Furniture")


# Number of random images to select from each subfolder
N_IMAGES_PER_FOLDER = 200

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def get_image_files(folder_path):
    """Get all image files from a folder."""
    image_files = []
    try:
        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_files.append(file_path)
    except Exception as e:
        print(f"Error reading folder {folder_path}: {e}")
    return image_files


def copy_random_images_from_subfolders(DESTINATION_FOLDER, output_folder, n_images):
    """
    Select N random images from each nested subfolder and copy to output folder.
    Handles errors gracefully - skips folders that don't have enough images or have issues.
    """
    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Clear existing files in output folder
    for existing_file in output_folder.glob('*'):
        if existing_file.is_file():
            try:
                existing_file.unlink()
            except Exception as e:
                print(f"Error deleting {existing_file}: {e}")
    
    total_copied = 0
    subfolders_processed = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(DESTINATION_FOLDER):
        root_path = Path(root)
        
        # Skip the root input folder itself
        if root_path == DESTINATION_FOLDER:
            continue
        
        # Get all image files in this subfolder
        image_files = get_image_files(root_path)
        
        if not image_files:
            continue
        
        # Select random images (min of n_images or available images)
        n_to_select = min(n_images, len(image_files))
        
        try:
            selected_images = random.sample(image_files, n_to_select)
            
            # Copy selected images to output folder
            for cnt, img_path in enumerate(selected_images):
                try:
                    # Create unique filename with subfolder name prefix
                    subfolder_name = root_path.name
                    new_filename = f"{cnt}_{img_path.name}"
                    destination = output_folder / new_filename
                    
                    shutil.copy2(img_path, destination)
                    total_copied += 1
                    
                except Exception as e:
                    print(f"Error copying {img_path}: {e}")
                    continue
            
            subfolders_processed += 1
            print(f"Processed {root_path.relative_to(DESTINATION_FOLDER)}: selected {n_to_select} images")
            
        except Exception as e:
            print(f"Error processing subfolder {root_path}: {e}")
            continue
    
    print(f"\nCompleted! Processed {subfolders_processed} subfolders, copied {total_copied} images to {output_folder}")


if __name__ == "__main__":
    print(f"Source folder (with nested subfolders): {INPUT_FOLDER}")
    print(f"Output folder (where images will be copied): {DESTINATION_FOLDER}")
    print(f"Images per subfolder: {N_IMAGES_PER_FOLDER}\n")
    
    if not INPUT_FOLDER.exists():
        print(f"Error: Source folder {INPUT_FOLDER} does not exist!")
    else:
        copy_random_images_from_subfolders(INPUT_FOLDER, DESTINATION_FOLDER, N_IMAGES_PER_FOLDER)

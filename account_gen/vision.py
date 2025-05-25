import cv2
import numpy as np
from interactions import FontConfig
from PIL import Image, ImageDraw, ImageFont
import os


class FindText():
    def __init__(self) -> None:
        font_info = None

    def calibrate(img: np.ndarray, calib_str: str, fontinfo: any):
        pass


def display_image(image: np.ndarray, window_name: str = "Debug View", wait_for_key: bool = False):
    """
    Display an image using OpenCV with configurable wait behavior.
    Args:
        image: The image to display (numpy array)
        window_name: Name of the window to display the image in
        wait_for_key: If True, waits for any key press. If False, waits 20ms
    """
    cv2.imshow(window_name, image)
    # cv2.imshow(window_name, cv2.resize(image,(800,800)))
    if wait_for_key:
        cv2.waitKey(0)  # Wait for any key press
    else:
        cv2.waitKey(20)  # Wait for 20ms


def generate_string_pattern(string: str, fontinfo: FontConfig):
    """
    Convert string to an image to be used in pattern matching.
    Args:
        string: The text to convert to an image
        fontinfo: FontConfig instance containing font parameters
    Returns:
        np.ndarray: Image containing the text
    """
    # Use FontConfig members directly - MUST match exactly with interactions.py
    font_family = fontinfo.FONT_FAMILY.split(',')[0].strip()  # Get 'Arial' from 'Arial, sans-serif'
    font_size = int(fontinfo.FONT_SIZE.replace('px', ''))  # Convert '16px' to 16
    text_transform = fontinfo.TEXT_TRANSFORM
    font_weight = fontinfo.FONT_WEIGHT.lower()
    font_style = fontinfo.FONT_STYLE.lower()
    
    # Apply text transform
    if text_transform == 'uppercase':
        string = string.upper()
    elif text_transform == 'lowercase':
        string = string.lower()
    elif text_transform == 'capitalize':
        string = string.title()
    
    # Try to load Arial font - this MUST match the font used in interactions.py
    try:
        # On Windows, Arial is typically in this location
        font_path = "C:\\Windows\\Fonts\\arial.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        # If we can't find Arial, we should fail explicitly rather than silently using a different font
        raise OSError("Could not load Arial font. This is required for pattern matching to work correctly with interactions.py")
    
    # Create an absolutely massive image (8K size)
    img = Image.new('L', (7680, 4320), 0)  # 8K resolution
    draw = ImageDraw.Draw(img)
    
    # Draw text in the center of this massive space
    draw.text((3840, 2160), string, font=font, fill=255, anchor="mm")  # Center text using anchor
    
    # Convert to numpy array
    img_np = np.array(img)
    
    # Convert to pure binary (0 or 255) using a higher threshold
    img_np = (img_np > 200).astype(np.uint8) * 255
    
    # Find non-zero elements and get their bounding box
    coords = np.nonzero(img_np)
    if len(coords[0]) == 0:  # If no text was found
        return img_np
    
    top = np.min(coords[0])
    bottom = np.max(coords[0])
    left = np.min(coords[1])
    right = np.max(coords[1])
    
    # Add a small safety margin, but ensure we stay in bounds
    margin = 5
    top = max(0, top - margin)
    bottom = min(img_np.shape[0] - 1, bottom + margin)
    left = max(0, left - margin)
    right = min(img_np.shape[1] - 1, right + margin)
    
    cropped = img_np[top:bottom+1, left:right+1]
    
    # Debug: Save the image to see what's happening
    debug_img = Image.fromarray(cropped)
    debug_img.save('debug_crop.png')
    
    return cropped

    
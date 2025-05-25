import cv2
import numpy as np
from .interactions import FontConfig
from PIL import Image, ImageDraw, ImageFont
import os


class FindText():
    def __init__(self) -> None:
        font_info = None

    def calibrate(img: np.ndarray, calib_str: str, fontinfo: any):
        pass



def generate_string_pattern(string: str, fontinfo: dict[str, any]):
    """
    Convert string to an image to be used in pattern matching.
    Args:
        string: The text to convert to an image
        fontinfo: Dictionary containing font parameters:
            - font_family: Font family (e.g., 'Arial')
            - font_size: Font size in pixels
            - text_transform: Text transformation (e.g., 'uppercase')
            - font_weight: Font weight (e.g., 'normal', 'bold')
            - font_style: Font style (e.g., 'normal', 'italic')
    Returns:
        np.ndarray: Image containing the text with 1-pixel padding
    """
    # Map CSS font properties to PIL parameters
    font_family = fontinfo.get('font_family', FontConfig.FONT_FAMILY).split(',')[0].strip()
    font_size = int(fontinfo.get('font_size', FontConfig.FONT_SIZE).replace('px', ''))
    text_transform = fontinfo.get('text_transform', FontConfig.TEXT_TRANSFORM)
    font_weight = fontinfo.get('font_weight', FontConfig.FONT_WEIGHT).lower()
    font_style = fontinfo.get('font_style', FontConfig.FONT_STYLE).lower()
    
    # Apply text transform
    if text_transform == 'uppercase':
        string = string.upper()
    elif text_transform == 'lowercase':
        string = string.lower()
    elif text_transform == 'capitalize':
        string = string.title()
    
    # Create PIL font

    # Try to load the exact font
    font = ImageFont.truetype(font_family, font_size)

    
    # Get text size
    bbox = font.getbbox(string)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Create image with padding
    img = Image.new('L', (text_width + 2, text_height + 2), 0)
    draw = ImageDraw.Draw(img)
    
    # Draw text
    draw.text((1, 1), string, font=font, fill=255)
    
    # Convert PIL image to numpy array
    img_np = np.array(img)
    
    # Find non-zero pixels to get tight bounds
    coords = cv2.findNonZero(img_np)
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop image to tight bounds
    cropped = img_np[y:y+h, x:x+w]
    
    return cropped

    
import cv2
import numpy as np
from interactions import FontConfig
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass

import os


@dataclass
class PatternMatch:
    debug_image: np.ndarray | None
    x: int
    y: int
    width: int
    height: int
    score: float


@dataclass
class PatternMatchScale(PatternMatch):
    scale: int

    def __post_init__(self):
        # Validate score is between 0 and 1
        if not 0 <= self.score <= 1:
            raise ValueError("Score must be between 0 and 1")


@dataclass
class PatternMatchScroll(PatternMatchScale):
    scroll_position_Y: int


class FindText:
    def __init__(self) -> None:
        font_info = None

    def calibrate(img: np.ndarray, calib_str: str, fontinfo: any):
        pass


def display_image(
    image: np.ndarray, window_name: str = "Debug View", wait_for_key: bool = False
):
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
    font_family = fontinfo.FONT_FAMILY.split(",")[
        0
    ].strip()  # Get 'Arial' from 'Arial, sans-serif'
    font_size = int(fontinfo.FONT_SIZE.replace("px", ""))  # Convert '16px' to 16
    text_transform = fontinfo.TEXT_TRANSFORM
    font_weight = fontinfo.FONT_WEIGHT.lower()
    font_style = fontinfo.FONT_STYLE.lower()

    # Apply text transform
    if text_transform == "uppercase":
        string = string.upper()
    elif text_transform == "lowercase":
        string = string.lower()
    elif text_transform == "capitalize":
        string = string.title()

    # Try to load Arial font - this MUST match the font used in interactions.py
    try:
        # On Windows, Arial is typically in this location
        font_path = "C:\\Windows\\Fonts\\arial.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        # If we can't find Arial, we should fail explicitly rather than silently using a different font
        raise OSError(
            "Could not load Arial font. This is required for pattern matching to work correctly with interactions.py"
        )

    # Create an absolutely massive image (8K size)
    img = Image.new("L", (7680, 4320), 0)  # 8K resolution
    draw = ImageDraw.Draw(img)

    # Draw text in the center of this massive space
    draw.text(
        (3840, 2160), string, font=font, fill=255, anchor="mm"
    )  # Center text using anchor

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

    cropped = img_np[top : bottom + 1, left : right + 1]

    # Debug: Save the image to see what's happening
    debug_img = Image.fromarray(cropped)
    debug_img.save("debug_crop.png")

    return cropped


def gaussian_blur(
    img: np.ndarray, kernel_size: tuple = (5, 5), sigma: float = 0
) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    Args:
        img: Input image as numpy array
        kernel_size: Tuple of (width, height) for the Gaussian kernel
        sigma: Standard deviation in X direction. If 0, calculated from kernel size
    Returns:
        Blurred image as numpy array
    """
    return cv2.GaussianBlur(img, kernel_size, sigma)


def pattern_match(img: np.ndarray, pattern: np.ndarray) -> PatternMatch:
    """
    Find the best match for a pattern in an input image using OpenCV template matching.
    Args:
        img: Input image to search in
        pattern: Pattern to search for
    Returns:
        PatternMatch: Object containing match location and confidence score
    """
    # Convert images to grayscale if they aren't already
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    if len(pattern.shape) == 3:
        pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(img_gray, pattern, cv2.TM_CCOEFF_NORMED)

    # Get the best match location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Get the pattern dimensions
    h, w = pattern.shape

    print(f"Pattern dimensions: {w}x{h}")
    print(f"Match location: {max_loc}")
    print(f"Input image shape: {img.shape}")
    print(f"Match score: {max_val}")

    # Create a copy of the input image for visualization
    if len(img.shape) == 3:
        vis_img = img.copy()
    else:
        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Create PatternMatch object first
    match = PatternMatch(
        x=max_loc[0],
        y=max_loc[1],
        width=w,
        height=h,
        score=max_val,
        debug_image=None,  # We'll set this after cropping
    )

    # Extract the matched region using the exact coordinates from PatternMatch
    matched_region = vis_img[
        match.y : match.y + match.height, match.x : match.x + match.width
    ]
    print(
        f"Cropping region: x={match.x}:{match.x + match.width}, y={match.y}:{match.y + match.height}"
    )
    print(f"Matched region shape: {matched_region.shape}")

    # Convert pattern to BGR if it's grayscale
    if len(pattern.shape) == 2:
        pattern_bgr = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
    else:
        pattern_bgr = pattern.copy()

    # Resize pattern to match the matched region's dimensions
    pattern_resized = cv2.resize(
        pattern_bgr, (matched_region.shape[1], matched_region.shape[0])
    )

    # Stack them side by side
    comparison = np.hstack((pattern_resized, matched_region))

    # Set the debug image in the PatternMatch object
    match.debug_image = comparison

    return match


def find_searchpattern_scale(
    img, pattern, save_debug_imgs=False, known_scale: int | None = None
) -> PatternMatchScale:
    """
    Find the best scale for pattern matching using binary search
    """
    best_score: int = 0
    best_match: PatternMatchScale | None = None

    # Define the scale range to search
    min_scale = 50
    max_scale = 150
    step = 5

    # override the scale range if we know the scale
    if known_scale:
        min_scale = known_scale
        max_scale = known_scale
        step = 1

    # First pass: try all scales to find approximate best region
    for scale_percent in range(min_scale, max_scale + 1, step):
        print(f"calculating for {scale_percent}% scale")
        # Calculate new dimensions based on percentage
        width = int(pattern.shape[1] * scale_percent / 100)
        height = int(pattern.shape[0] * scale_percent / 100)

        # Resize pattern
        scaled_pattern = cv2.resize(pattern, (width, height))

        # Try to match
        match = pattern_match(img, scaled_pattern)
        print(f"Match score: {match.score}")

        # Keep track of best match
        if match.score > best_score:
            best_score = match.score
            # best_match = match
            best_match = PatternMatchScale(**vars(match), scale=scale_percent)
            print(
                f"New best match found at {scale_percent}% scale with score {best_score}"
            )
            print(f"Match location: ({best_match.x}, {best_match.y})")
            print(f"Match dimensions: {best_match.width}x{best_match.height}")

    # If we found a good match, do a finer search around that scale
    if (best_match and best_score > 0.5) and not known_scale:
        best_scale = (best_match.width / pattern.shape[1]) * 100
        print(f"\nDoing fine search around best scale: {best_scale:.1f}%")

        # Search in smaller steps around the best scale
        fine_min = max(min_scale, best_scale - 10)
        fine_max = min(max_scale, best_scale + 10)

        for scale_percent in range(int(fine_min), int(fine_max) + 1, 1):
            print(f"Fine search at {scale_percent}% scale")
            width = int(pattern.shape[1] * scale_percent / 100)
            height = int(pattern.shape[0] * scale_percent / 100)

            scaled_pattern = cv2.resize(pattern, (width, height))
            match = pattern_match(img, scaled_pattern)
            print(f"Match score: {match.score}")

            if match.score > best_score:
                best_score = match.score
                best_match = PatternMatchScale(**vars(match), scale=scale_percent)
                print(
                    f"New best match found at {scale_percent}% scale with score {best_score}"
                )
                print(f"Match location: ({best_match.x}, {best_match.y})")
                print(f"Match dimensions: {best_match.width}x{best_match.height}")

    # Convert the final match coordinates back to original pattern scale
    if best_match:
        if not known_scale:
            original_scale = 100 / (best_match.width / pattern.shape[1])
            best_match.x = int(best_match.x * original_scale / 100)
            best_match.y = int(best_match.y * original_scale / 100)
            best_match.width = pattern.shape[1]
            best_match.height = pattern.shape[0]
        print(f"\nFinal match after scale conversion:")
        print(f"Location: ({best_match.x}, {best_match.y})")
        print(f"Dimensions: {best_match.width}x{best_match.height}")

        if save_debug_imgs:
            # Save debug images for the best match
            if len(pattern.shape) == 2:
                pattern_bgr = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
            else:
                pattern_bgr = pattern.copy()

            if len(img.shape) == 3:
                vis_img = img.copy()
            else:
                vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            matched_region = vis_img[
                best_match.y : best_match.y + best_match.height,
                best_match.x : best_match.x + best_match.width,
            ]

            cv2.imwrite("debug_pattern.png", pattern_bgr)
            cv2.imwrite("debug_matched.png", matched_region)

    return best_match

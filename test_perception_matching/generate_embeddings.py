import numpy as np
import cv2
import random
from typing import List, Tuple, Literal, Optional, Union, Any
from dataclasses import dataclass, asdict
import json
from enum import Enum

@dataclass
class ImageEmbeddingParams:
    """Parameters for image embedding generation, designed to be serializable."""
    vertical: int
    horizontal: int
    overlap: int
    bins_per_channel: int
    center_histograms: bool
    mask: bool
    
 
    def to_dict(self):
        """Convert parameters to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self):
        """Convert parameters to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data):
        """Create parameters from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str):
        """Create parameters from JSON string."""
        return cls.from_dict(json.loads(json_str))

def create_image_slices(image, vertical=10, horizontal=9, overlap=10) -> list[tuple[slice, slice, str]]:
    """
    Create a list of slice objects for vertical and horizontal stripes with overlap.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    vertical : int
        Number of vertical stripes
    horizontal : int
        Number of horizontal stripes
    overlap : int
        Overlap percentage between adjacent stripes
    
    Returns:
    --------
    list of tuples
        Each tuple contains (slice_y, slice_x, 'v'|'h') where:
        - slice_y is the vertical slice object
        - slice_x is the horizontal slice object
        - 'v' indicates vertical stripe, 'h' indicates horizontal stripe
    """
    height, width = image.shape[:2]
    slices = []
    
    # Calculate stripe dimensions with overlap
    vert_stripe_width = width / ((vertical * (100 - overlap) / 100) + (overlap / 100))
    horiz_stripe_height = height / ((horizontal * (100 - overlap) / 100) + (overlap / 100))
    
    overlap_width = vert_stripe_width * (overlap / 100)
    overlap_height = horiz_stripe_height * (overlap / 100)
    
    # Create vertical stripes (columns)
    for i in range(vertical):
        start_x = int(i * (vert_stripe_width - overlap_width))
        end_x = int(min(start_x + vert_stripe_width, width))
        slices.append((slice(0, height), slice(start_x, end_x), 'v'))
    
    # Create horizontal stripes (rows)
    for i in range(horizontal):
        start_y = int(i * (horiz_stripe_height - overlap_height))
        end_y = int(min(start_y + horiz_stripe_height, height))
        slices.append((slice(start_y, end_y), slice(0, width), 'h'))
    
    return slices

def visualize_slices(image, slices):
    """
    Visualize the image slices as colored outlines on a grayscale background.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    slices : list of tuples
        List of slice objects (as returned by create_image_slices)
    
    Returns:
    --------
    numpy.ndarray
        Grayscale image with colored box outlines for each slice
    """
    # Convert the input image to grayscale
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert back to 3 channels for colorization
        vis_image = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    else:
        # Already grayscale
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Count vertical slices
    vertical_count = sum(1 for _, _, direction in slices if direction == 'v')
    
    # Color palette for better distinction
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Light Blue
        (128, 255, 0)   # Lime
    ]
    
    # Draw each slice as a simple colored box
    for i, (slice_y, slice_x, direction) in enumerate(slices):
        # Use colors from the palette
        color = colors[i % len(colors)]
        
        # Get slice boundaries
        y_start, y_end = slice_y.start, slice_y.stop
        x_start, x_end = slice_x.start, slice_x.stop
        
        # Draw a simple rectangle outline
        cv2.rectangle(vis_image, (x_start, y_start), (x_end-1, y_end-1), color, 2)
        
        # Add a simple label
        if direction == 'v':
            label = f"V{i}"
            text_pos = (x_start + 5, 20)
        else:
            label = f"H{i-vertical_count}"
            text_pos = (5, y_start + 20)
        
        # Add the slice label in matching color
        cv2.putText(vis_image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, color, 2, cv2.LINE_AA)
    
    return vis_image

def demo_image_slicing(image_path, vertical=10, horizontal=9, overlap=10):
    """
    Demonstration function to test the image slicing and visualization.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    vertical : int
        Number of vertical stripes
    horizontal : int
        Number of horizontal stripes
    overlap : int
        Overlap percentage between adjacent stripes
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    slices = create_image_slices(image, vertical, horizontal, overlap)
    vis_image = visualize_slices(image, slices)
    
    cv2.imshow("Image Slices", vis_image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return vis_image

def create_test_image(width=800, height=600, num_shapes=50, bg_color=(240, 240, 240)):
    """
    Create a test image with random shapes of different colors.
    
    Parameters:
    -----------
    width : int
        Base width of the image
    height : int
        Base height of the image
    num_shapes : int
        Number of random shapes to draw
    bg_color : tuple
        Background color (B, G, R)
    
    Returns:
    --------
    numpy.ndarray
        Generated image with random shapes
    """
    # Randomize aspect ratio (between 0.5 and 2.0)
    aspect_ratio = random.uniform(0.5, 2.0)
    adjusted_width = int(width * aspect_ratio)
    adjusted_height = int(height / aspect_ratio)
    
    # Create a blank image
    image = np.ones((adjusted_height, adjusted_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
    
    # Shape types
    shape_types = ['circle', 'rectangle', 'line', 'polygon']
    
    for _ in range(num_shapes):
        # Random color (BGR)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        
        # Random shape type
        shape_type = random.choice(shape_types)
        
        if shape_type == 'circle':
            # Random circle
            center = (
                random.randint(0, adjusted_width),
                random.randint(0, adjusted_height)
            )
            radius = random.randint(10, min(100, adjusted_width//4, adjusted_height//4))
            thickness = random.randint(-1, 10)  # -1 for filled
            cv2.circle(image, center, radius, color, thickness)
            
        elif shape_type == 'rectangle':
            # Random rectangle
            pt1 = (
                random.randint(0, adjusted_width - 10),
                random.randint(0, adjusted_height - 10)
            )
            pt2 = (
                random.randint(pt1[0] + 10, min(adjusted_width, pt1[0] + 200)),
                random.randint(pt1[1] + 10, min(adjusted_height, pt1[1] + 200))
            )
            thickness = random.randint(-1, 10)  # -1 for filled
            cv2.rectangle(image, pt1, pt2, color, thickness)
            
        elif shape_type == 'line':
            # Random line
            pt1 = (
                random.randint(0, adjusted_width),
                random.randint(0, adjusted_height)
            )
            pt2 = (
                random.randint(0, adjusted_width),
                random.randint(0, adjusted_height)
            )
            thickness = random.randint(1, 10)
            cv2.line(image, pt1, pt2, color, thickness)
            
        else:  # polygon
            try:
                # Random polygon with safety checks
                num_points = random.randint(3, 8)
                if adjusted_width <= 0 or adjusted_height <= 0:
                    continue  # Skip this shape if dimensions are invalid
                
                # Generate points ensuring we have at least 3 valid points
                points = []
                for _ in range(num_points):
                    x = min(max(0, random.randint(0, adjusted_width-1)), adjusted_width-1)
                    y = min(max(0, random.randint(0, adjusted_height-1)), adjusted_height-1)
                    points.append([x, y])
                
                # Safety check - make sure we have enough points
                if len(points) < 3:
                    continue
                
                points = np.array(points, np.int32)
                points = points.reshape((-1, 1, 2))
                
                # Use a safer thickness range
                thickness = random.randint(1, 5)  # Avoid -1 for now
                
                # Draw the polygon outline
                cv2.polylines(image, [points], True, color, thickness)
                
                # Maybe fill the polygon
                if random.random() > 0.5:
                    cv2.fillPoly(image, [points], color)
            except Exception:
                # If any errors occur, just skip this shape
                continue
    
    return image

def demo_with_random_image(params: ImageEmbeddingParams = None):
    """
    Demonstration function that creates two random test images,
    shows the slicing visualization, calculates histogram embeddings,
    and compares them.
    
    Parameters:
    -----------
    params : ImageEmbeddingParams, optional
        Parameters for embedding generation (uses default values if None)
        
    Returns:
    --------
    tuple
        (original_image1, original_image2, visualization1, visualization2, similarity_score)
    """
    # Use default parameters if none provided
    if params is None:
        params = ImageEmbeddingParams()
    
    # Create two random test images
    test_image1 = create_test_image()
    test_image2 = create_test_image()
    
    # Get slices for both images
    slices1 = create_image_slices(test_image1, params.vertical, params.horizontal, params.overlap)
    slices2 = create_image_slices(test_image2, params.vertical, params.horizontal, params.overlap)
    
    # Visualize the slices
    vis_image1 = visualize_slices(test_image1, slices1)
    vis_image2 = visualize_slices(test_image2, slices2)
    
    # Calculate embeddings for both images
    embedding1 = create_image_embedding(test_image1, params=params)
    embedding2 = create_image_embedding(test_image2, params=params)
    
    # Compare the embeddings
    similarity = compare_images_for_vector_db(test_image1, test_image2, params=params)
    
    # Display results
    cv2.imshow("Image 1", test_image1)
    cv2.imshow("Image 1 Slices", vis_image1)
    cv2.imshow("Image 2", test_image2)
    cv2.imshow("Image 2 Slices", vis_image2)
    
    # Create a text image displaying the comparison result
    result_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    # Add text showing the similarity score
    cv2.putText(
        result_img, 
        f"Similarity Score: {similarity:.4f}", 
        (20, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (0, 0, 0), 
        2
    )
    
    # Add text explaining what the score means
    cv2.putText(
        result_img, 
        "1.0 = identical, 0.0 = unrelated", 
        (20, 90), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 0, 0), 
        1
    )
    
    # Add text showing histogram parameters
    cv2.putText(
        result_img, 
        f"Bins: {params.bins_per_channel}, Centered: {params.center_histograms}", 
        (20, 130), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        (0, 0, 255), 
        1
    )
    
    # Display the result image
    cv2.imshow("Comparison Result", result_img)
    
    # Wait for key press and clean up
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Similarity score: {similarity:.4f}")
    print(f"Embedding size: {len(embedding1)}")
    
    return test_image1, test_image2, vis_image1, vis_image2, similarity

def demo_similar_images(params: ImageEmbeddingParams = None):
    """
    Demonstrates comparison between a random image and a slightly modified version of itself.
    
    Parameters:
    -----------
    params : ImageEmbeddingParams, optional
        Parameters for embedding generation (uses default values if None)
    
    Returns:
    --------
    tuple
        (original_image, modified_image, similarity_score)
    """
    # Use default parameters if none provided
    if params is None:
        params = ImageEmbeddingParams()
    
    # Create a random test image
    original_image = create_test_image()
    
    # Create a slightly modified version (add noise)
    modified_image = original_image.copy()
    noise = np.random.normal(0, 15, modified_image.shape).astype(np.uint8)
    modified_image = cv2.add(modified_image, noise)
    
    # Compare the images
    similarity = compare_images_for_vector_db(original_image, modified_image, params=params)
    
    # Display results
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Modified Image (with noise)", modified_image)
    
    # Create a text image displaying the comparison result
    result_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(
        result_img, 
        f"Similarity Score: {similarity:.4f}", 
        (20, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (0, 0, 0), 
        2
    )
    cv2.putText(
        result_img, 
        "Similar images with added noise", 
        (20, 90), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 0, 0), 
        1
    )
    
    # Display the result image
    cv2.imshow("Similarity Result", result_img)
    
    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Similarity between original and noisy version: {similarity:.4f}")
    
    return original_image, modified_image, similarity

def calculate_slice_histograms(
    image, 
    slices: List[Tuple[slice, slice, Literal['v', 'h']]], 
    bins_per_channel: int = 8, 
    mask: Optional[np.ndarray] = None,
    center_histograms: bool = False
) -> List[np.ndarray]:
    """
    Calculate normalized color histograms for each image slice.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (3-channel BGR)
    slices : List[Tuple[slice, slice, str]]
        List of slice objects as returned by create_image_slices
    bins_per_channel : int
        Number of bins per color channel
    mask : numpy.ndarray, optional
        Optional mask for the entire image
    center_histograms : bool
        If True, center histograms by subtracting the mean from each bin,
        making cosine similarity behave more like correlation
        
    Returns:
    --------
    List[np.ndarray]
        List of normalized, flattened histograms for each slice
    """
    # Ensure the image is in BGR format
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a 3-channel BGR image")
    
    histogram_list = []
    
    for slice_y, slice_x, direction in slices:
        # Extract the slice from the image
        slice_image = image[slice_y, slice_x]
        
        # Apply the mask to this slice if provided
        slice_mask = None
        if mask is not None:
            slice_mask = mask[slice_y, slice_x]
        
        # Calculate histogram for all 3 channels - always apply the mask (which might be None)
        hist = cv2.calcHist(
            [slice_image], 
            [0, 1, 2], 
            slice_mask, 
            [bins_per_channel, bins_per_channel, bins_per_channel],
            [0, 256, 0, 256, 0, 256]
        )
        
        # Check if the histogram is empty (all zeros)
        if np.sum(hist) == 0:
            hist = hist.flatten()  # Just flatten without normalization
        else:
            # Normalize and flatten if the histogram has values
            hist = cv2.normalize(hist, hist).flatten()
        
        # Center the histogram if requested (only if it has non-zero values)
        if center_histograms and np.sum(hist) > 0:
            hist_mean = np.mean(hist)
            hist = hist - hist_mean
        
        # Add to our collection
        histogram_list.append(hist)
    
    return histogram_list

def create_image_embedding(
    image, 
    params: ImageEmbeddingParams,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create an image embedding by calculating histograms for slices of the image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    params : ImageEmbeddingParams
        Parameters for the embedding generation (uses default values if None)
    mask : numpy.ndarray, optional
        Optional mask for the image
        
    Returns:
    --------
    numpy.ndarray
        Feature vector representing the image as concatenated histograms
    """

    # Create the slices
    slices = create_image_slices(image, params.vertical, params.horizontal, params.overlap)
    

    # Calculate histograms for each slice
    histograms = calculate_slice_histograms(
        image, slices, params.bins_per_channel, mask, params.center_histograms
    )

    # for histogram in histograms:
    #     print(histogram.shape)
    # Always concatenate into a single feature vector
    return np.concatenate(histograms) if histograms else np.array([])

def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Parameters:
    -----------
    vector1 : np.ndarray
        First vector
    vector2 : np.ndarray
        Second vector
        
    Returns:
    --------
    float
        Cosine similarity between the vectors (between -1 and 1)
        Returns 0.0 if either vector has zero norm
    """
    # Check for zero vectors
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    return np.dot(vector1, vector2) / (norm1 * norm2)

def compare_images_for_vector_db(
    image1, 
    image2, 
    params: ImageEmbeddingParams,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compare two images using cosine similarity for vector database compatibility.
    
    Parameters:
    -----------
    image1 : numpy.ndarray
        First input image
    image2 : numpy.ndarray
        Second input image
    params : ImageEmbeddingParams, optional
        Parameters for embedding generation (uses default values if None)
    mask : Optional[np.ndarray], optional
        Optional mask to apply to both images
        
    Returns:
    --------
    float
        Similarity score between the two images (higher means more similar)
    """

    
    # Get embeddings for both images
    embedding1 = create_image_embedding(image1, params=params, mask=mask)
    embedding2 = create_image_embedding(image2, params=params, mask=mask)
    
    # Handle empty embeddings
    if embedding1.size == 0 or embedding2.size == 0:
        return 0.0
    
    # Use the utility function for cosine similarity
    return cosine_similarity(embedding1, embedding2)

def create_circular_mask(image_shape, center=None, radius=None):
    """
    Create a circular mask for an image.
    
    Parameters:
    -----------
    image_shape : tuple
        Shape of the image (height, width) or (height, width, channels)
    center : tuple, optional
        (x, y) coordinates of the center. Default is the center of the image.
    radius : int, optional
        Radius of the circle. Default is the largest possible radius.
        
    Returns:
    --------
    numpy.ndarray
        Binary mask where the circle is 1 and the rest is 0
    """
    # Get height and width
    if len(image_shape) == 3:
        height, width = image_shape[:2]
    else:
        height, width = image_shape
    
    # Set default center if not provided
    if center is None:
        center = (width // 2, height // 2)
    
    # Set default radius if not provided
    if radius is None:
        radius = min(width, height) // 2
    
    # Create coordinate arrays
    Y, X = np.ogrid[:height, :width]
    
    # Calculate squared distance from center
    dist_from_center = (X - center[0])**2 + (Y - center[1])**2
    
    # Create the mask
    mask = dist_from_center <= radius**2
    
    return mask.astype(np.uint8)

def visualize_mask(image, mask):
    """
    Visualize a mask on an image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original image
    mask : numpy.ndarray
        Binary mask
        
    Returns:
    --------
    numpy.ndarray
        Image with the mask highlighted
    """
    # Make a copy of the image
    if len(image.shape) == 2:  # Grayscale
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Convert the mask to BGR for better visualization
    mask_vis = np.zeros_like(vis_image)
    mask_vis[:,:,1] = mask * 255  # Green channel
    
    # Blend the mask with the image
    alpha = 0.5
    result = cv2.addWeighted(vis_image, 1.0, mask_vis, alpha, 0)
    
    # Draw the contour of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    return result

def test_circular_mask():
    """
    Test function to visualize a circular mask on an image.
    """
    # Create a test image
    test_image = create_test_image(width=600, height=400)
    
    # Create a circular mask
    mask = create_circular_mask(test_image.shape)
    
    # Visualize the mask
    mask_vis = visualize_mask(test_image, mask)
    
    # Calculate some stats about the mask
    mask_coverage = np.sum(mask) / (mask.shape[0] * mask.shape[1]) * 100
    
    # Create an information image
    info_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(info_img, f"Image shape: {test_image.shape[:2]}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(info_img, f"Mask radius: {min(test_image.shape[:2]) // 2} px", 
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(info_img, f"Mask coverage: {mask_coverage:.1f}%", 
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(info_img, f"Mask pixels: {np.sum(mask)}", 
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Display the results
    cv2.imshow("Original Image", test_image)
    cv2.imshow("Image with Mask", mask_vis)
    cv2.imshow("Mask Information", info_img)
    
    # Just show the mask itself
    mask_only = mask.copy() * 255
    cv2.imshow("Mask Only", mask_only)
    
    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return test_image, mask, mask_vis

def test_masked_embedding():
    """
    Test function to demonstrate how a mask affects image embeddings.
    """
    # Create a test image
    test_image = create_test_image(width=600, height=400)
    
    # Create a circular mask
    mask = create_circular_mask(test_image.shape)
    
    # Visualize the mask
    mask_vis = visualize_mask(test_image, mask)
    
    # Create embedding parameters
    params = ImageEmbeddingParams(
        vertical=5,
        horizontal=4,
        overlap=10,
        bins_per_channel=8,
        center_histograms=True
    )
    
    # Create slices for visualization
    slices = create_image_slices(test_image, params.vertical, params.horizontal, params.overlap)
    
    # Visualize the slices (on grayscale to see better)
    gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    slices_vis = visualize_slices(gray_image, slices)
    
    # Calculate embeddings with and without mask
    embedding_with_mask = create_image_embedding(test_image, params=params, mask=mask)
    embedding_without_mask = create_image_embedding(test_image, params=params, mask=None)
    
    # Compare the embeddings
    similarity = cosine_similarity(embedding_with_mask, embedding_without_mask)
    
    # Verify embedding sizes are identical
    embedding_with_mask_size = len(embedding_with_mask)
    embedding_without_mask_size = len(embedding_without_mask)
    sizes_match = embedding_with_mask_size == embedding_without_mask_size
    
    # Calculate how many slices are affected by the mask
    affected_slices = 0
    for slice_y, slice_x, _ in slices:
        slice_mask = mask[slice_y, slice_x]
        # If the slice is partially masked (between 1% and 99% coverage)
        coverage = np.sum(slice_mask) / (slice_mask.shape[0] * slice_mask.shape[1])
        if 0.01 < coverage < 0.99:
            affected_slices += 1
    
    # Create an information image
    info_img = np.ones((350, 500, 3), dtype=np.uint8) * 255
    cv2.putText(info_img, f"Masked vs Unmasked Similarity: {similarity:.4f}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(info_img, f"Embedding size with mask: {embedding_with_mask_size}", 
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(info_img, f"Embedding size without mask: {embedding_without_mask_size}", 
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(info_img, f"Sizes match: {sizes_match}", 
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100 if sizes_match else 255), 2)
    cv2.putText(info_img, f"Slices: {len(slices)}", 
                (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(info_img, f"Partially masked slices: {affected_slices}", 
                (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(info_img, f"Mask coverage: {np.sum(mask) / mask.size * 100:.1f}%", 
                (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(info_img, "Lower similarity = bigger mask impact", 
                (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
    
    # Display the results
    cv2.imshow("Original Image", test_image)
    cv2.imshow("Image with Mask", mask_vis)
    cv2.imshow("Image with Slices", slices_vis)
    cv2.imshow("Embedding Information", info_img)
    
    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Masked vs Unmasked Similarity: {similarity:.4f}")
    print(f"Embedding size with mask: {embedding_with_mask_size}")
    print(f"Embedding size without mask: {embedding_without_mask_size}")
    print(f"Sizes match: {sizes_match}")
    
    return test_image, mask, mask_vis, embedding_with_mask, embedding_without_mask, similarity

def verify_mask_effect():
    """
    Verify that mask has the expected effect on embeddings while maintaining consistent size.
    """
    test_image = create_test_image(width=600, height=400)
    mask = create_circular_mask(test_image.shape)
    
    # Try different parameter combinations
    param_variations = [
        ImageEmbeddingParams(vertical=5, horizontal=4, bins_per_channel=4, center_histograms=False),
        ImageEmbeddingParams(vertical=10, horizontal=8, bins_per_channel=8, center_histograms=True),
        ImageEmbeddingParams(vertical=3, horizontal=3, bins_per_channel=16, center_histograms=False)
    ]
    
    results = []
    
    for i, params in enumerate(param_variations):
        # Generate embeddings with and without mask
        embedding_with_mask = create_image_embedding(test_image, params=params, mask=mask)
        embedding_without_mask = create_image_embedding(test_image, params=params, mask=None)
        
        # Check if sizes match
        sizes_match = len(embedding_with_mask) == len(embedding_without_mask)
        
        # Calculate similarity
        similarity = cosine_similarity(embedding_with_mask, embedding_without_mask)
        
        # Store results
        results.append({
            "params": params,
            "embedding_size": len(embedding_with_mask),
            "sizes_match": sizes_match,
            "similarity": similarity
        })
        
        print(f"Test {i+1}:")
        print(f"  Params: v={params.vertical}, h={params.horizontal}, bins={params.bins_per_channel}, center={params.center_histograms}")
        print(f"  Embedding size: {len(embedding_with_mask)}")
        print(f"  Sizes match: {sizes_match}")
        print(f"  Similarity: {similarity:.4f}")
        print()
    
    return results

def stress_test_embeddings(
    num_images=100, 
    params: ImageEmbeddingParams = None,
    show_visualizations=False,
    use_plasma=False
):
    """
    Run a stress test by processing multiple random images and comparing embeddings.
    
    Parameters:
    -----------
    num_images : int
        Number of images to process
    params : ImageEmbeddingParams, optional
        Parameters for embedding generation (uses default values if None)
    show_visualizations : bool
        Whether to show visualizations during processing
    use_plasma : bool
        Use plasma images instead of random shapes
    """
    # Use default parameters if none provided
    if params is None:
        params = ImageEmbeddingParams()
    
    import time
    start_time = time.time()
    
    # Statistics to track
    stats = {
        "total_images": num_images,
        "masked_images": 0,
        "embedding_sizes": [],
        "processing_times": [],
        "similarities": [],
        "masked_vs_unmasked_similarities": [],
        "consecutive_similarities": []
    }
    
    # Keep track of previous embedding for comparison
    prev_embedding = None
    
    # Process multiple random images
    for i in range(num_images):
        print(i)
        img_start_time = time.time()
        
        # Create a random test image with random dimensions
        width = random.randint(400, 800)
        height = random.randint(300, 600)
        
        # Use plasma or shape images based on parameter
        if use_plasma:
            # Random parameters for plasma
            blob_count = random.randint(5, 12)
            min_blob_size = random.randint(30, 70)
            max_blob_size = random.randint(100, 250)
            smoothness = random.randint(20, 50)
            
            test_image = create_plasma_image(
                width=width, 
                height=height,
                blob_count=blob_count,
                min_blob_size=min_blob_size,
                max_blob_size=max_blob_size,
                smoothness=smoothness
            )
        else:
            test_image = create_test_image(width=width, height=height)
        
        # Randomly decide whether to use a mask (50% chance)
        use_mask = random.random() > 0.5
        mask = None
        
        if use_mask:
            mask = create_circular_mask(test_image.shape)
            stats["masked_images"] += 1
        
        # Calculate embedding
        embedding = create_image_embedding(test_image, params=params, mask=mask)
        
        # Record embedding size
        stats["embedding_sizes"].append(len(embedding))
        
        # If mask was used, also calculate unmasked embedding and compare
        if use_mask:
            unmasked_embedding = create_image_embedding(test_image, params=params, mask=None)
            
            # Calculate similarity between masked and unmasked versions
            similarity = cosine_similarity(embedding, unmasked_embedding)
            if similarity != 0.0:  # Only add non-zero similarities
                stats["masked_vs_unmasked_similarities"].append(similarity)
        
        # Compare to previous embedding if available
        if prev_embedding is not None:
            similarity = cosine_similarity(embedding, prev_embedding)
            if similarity != 0.0:  # Only add non-zero similarities
                stats["consecutive_similarities"].append(similarity)
        
        # Store this embedding for the next comparison
        prev_embedding = embedding
        
        # Record processing time
        img_processing_time = time.time() - img_start_time
        stats["processing_times"].append(img_processing_time)
        
        # Optional visualization
        if show_visualizations:
            # Create slice visualization
            slices = create_image_slices(test_image, params.vertical, params.horizontal, params.overlap)
            vis_image = visualize_slices(test_image, slices)
            
            # Show mask if used
            if use_mask:
                mask_vis = visualize_mask(test_image, mask)
                cv2.imshow("Current Mask", mask_vis)
            
            # Show images
            cv2.imshow("Current Image", test_image)
            cv2.imshow("Current Slices", vis_image)
            
            # Create progress display
            progress_img = np.ones((100, 500, 3), dtype=np.uint8) * 255
            cv2.putText(
                progress_img,
                f"Processing image {i+1}/{num_images} ({'Plasma' if use_plasma else 'Shapes'})", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
            )
            cv2.putText(
                progress_img,
                f"Embedding size: {len(embedding)}", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
            )
            cv2.imshow("Progress", progress_img)
            
            # Brief delay - just 1ms as requested
            cv2.waitKey(1)
    
    # Calculate summary statistics
    total_time = time.time() - start_time
    
    if stats["embedding_sizes"]:
        stats["mean_embedding_size"] = np.mean(stats["embedding_sizes"])
        stats["min_embedding_size"] = min(stats["embedding_sizes"])
        stats["max_embedding_size"] = max(stats["embedding_sizes"])
    
    if stats["processing_times"]:
        stats["mean_processing_time"] = np.mean(stats["processing_times"])
        stats["total_processing_time"] = total_time
        stats["images_per_second"] = num_images / total_time
    
    if stats["consecutive_similarities"]:
        stats["mean_consecutive_similarity"] = np.mean(stats["consecutive_similarities"])
        stats["min_consecutive_similarity"] = min(stats["consecutive_similarities"])
        stats["max_consecutive_similarity"] = max(stats["consecutive_similarities"])
    
    if stats["masked_vs_unmasked_similarities"]:
        stats["mean_masked_vs_unmasked"] = np.mean(stats["masked_vs_unmasked_similarities"])
    
    # Print summary
    print("\n--- Stress Test Results ---")
    print(f"Processed {num_images} {'plasma' if use_plasma else 'shape'} images ({stats['masked_images']} with masks)")
    print(f"Total time: {total_time:.2f} seconds ({stats['images_per_second']:.2f} images/sec)")
    print(f"Embedding size: {stats['mean_embedding_size']:.1f} dimensions")
    print(f"Average processing time: {stats['mean_processing_time']*1000:.2f} ms per image")
    
    if stats["consecutive_similarities"]:
        print(f"\nConsecutive image similarity: {stats['mean_consecutive_similarity']:.4f}")
        print(f"  Range: {stats['min_consecutive_similarity']:.4f} to {stats['max_consecutive_similarity']:.4f}")
    
    if stats["masked_vs_unmasked_similarities"]:
        print(f"\nMasked vs unmasked similarity: {stats['mean_masked_vs_unmasked']:.4f}")
    
    print("\nAll embeddings were consistent in dimensionality")
    
    # Clean up windows
    cv2.destroyAllWindows()
    
    return stats

# Fixed the stress_test_with_plasma function
def stress_test_with_plasma(num_images=100, bins_per_channel=4, show_visualizations=True):
    """Run the stress test using plasma images instead of random shapes"""
    params = ImageEmbeddingParams(
        vertical=10, 
        horizontal=9, 
        overlap=10, 
        bins_per_channel=bins_per_channel,
        center_histograms=True
    )
    
    return stress_test_embeddings(
        num_images=num_images,
        params=params,
        show_visualizations=show_visualizations,
        use_plasma=True  # Use plasma images
    )


def create_plasma_image(
    width=800, 
    height=600, 
    blob_count=8,
    min_blob_size=50,
    max_blob_size=200,
    smoothness=30,
    seed=None,
    colorful=True
) -> np.ndarray:
    """
    Create a lava lamp-like plasma image with large colorful blobs.
    
    Parameters:
    -----------
    width : int
        Width of the image
    height : int
        Height of the image
    blob_count : int
        Approximate number of blobs to generate
    min_blob_size : int
        Minimum radius of blobs
    max_blob_size : int
        Maximum radius of blobs
    smoothness : int
        Controls the smoothness of transitions between blobs (higher = smoother)
    seed : int, optional
        Random seed for reproducible images
    colorful : bool
        Whether to use multiple vibrant colors (True) or a more uniform colormap (False)
        
    Returns:
    --------
    numpy.ndarray
        Generated plasma image as a 3-channel BGR array
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Create a blank canvas
    base_image = np.zeros((height, width), dtype=np.float32)
    
    # Generate metaballs (centers and sizes)
    metaballs = []
    for _ in range(blob_count):
        # Random center position
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        # Random size
        radius = np.random.randint(min_blob_size, max_blob_size)
        # Random strength
        strength = np.random.uniform(0.5, 1.5)
        
        metaballs.append((cx, cy, radius, strength))
    
    # Add a few large background blobs for more variety
    for _ in range(3):
        cx = np.random.randint(-100, width+100)
        cy = np.random.randint(-100, height+100)
        radius = np.random.randint(max_blob_size, max_blob_size*3)
        strength = np.random.uniform(0.3, 0.7)
        metaballs.append((cx, cy, radius, strength))
    
    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Calculate metaball field
    field = np.zeros((height, width), dtype=np.float32)
    
    for cx, cy, radius, strength in metaballs:
        # Calculate squared distance to center
        sq_dist = (x_coords - cx)**2 + (y_coords - cy)**2
        # Convert to metaball field value (inverse squared distance)
        radius_sq = radius**2
        # Add contribution from this metaball
        field += radius_sq * strength / (sq_dist + smoothness)
    
    # Normalize field to 0-1 range
    field_min = np.min(field)
    field_max = np.max(field)
    field = (field - field_min) / (field_max - field_min)
    
    # Apply some mild distortion for more organic look
    distortion = np.zeros_like(field)
    
    # Add several waves of different frequencies
    freqs = [2, 5, 15]
    for freq in freqs:
        angle = np.random.uniform(0, 2*np.pi)
        phase_x = np.random.uniform(0, 2*np.pi)
        phase_y = np.random.uniform(0, 2*np.pi)
        
        wave_x = np.sin(x_coords/width*freq*np.pi*2 + phase_x)
        wave_y = np.sin(y_coords/height*freq*np.pi*2 + phase_y)
        
        distortion += (wave_x + wave_y) * (0.05 / len(freqs))
    
    # Apply distortion
    field = np.clip(field + distortion, 0, 1)
    
    # Create colored image
    if colorful:
        # Generate vibrant color palette (3-5 colors)
        num_colors = np.random.randint(3, 6)
        palette = []
        
        # Generate vibrant colors with good separation
        hues = np.linspace(0, 1, num_colors, endpoint=False)
        hues = [h + np.random.uniform(-0.05, 0.05) for h in hues]  # Add slight randomness
        
        for hue in hues:
            # Convert HSV to BGR
            hue = hue % 1.0  # Ensure in range [0,1]
            sat = np.random.uniform(0.7, 1.0)
            val = np.random.uniform(0.8, 1.0)
            
            # Manual HSV to BGR conversion for vibrant colors
            c = val * sat
            x = c * (1 - abs((hue*6) % 2 - 1))
            m = val - c
            
            if hue < 1/6:
                r, g, b = c, x, 0
            elif hue < 2/6:
                r, g, b = x, c, 0
            elif hue < 3/6:
                r, g, b = 0, c, x
            elif hue < 4/6:
                r, g, b = 0, x, c
            elif hue < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            # Convert to BGR color
            bgr = (
                int(255 * (b + m)),
                int(255 * (g + m)),
                int(255 * (r + m))
            )
            palette.append(bgr)
        
        # Create gradient map between colors
        color_stops = np.linspace(0, 1, num_colors)
        
        # Map field values to colors
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        # For each pixel, interpolate between colors
        field_flat = field.flatten()
        
        # Find which color segment each pixel belongs to
        segment_indices = np.searchsorted(color_stops, field_flat) - 1
        segment_indices = np.clip(segment_indices, 0, num_colors - 2)
        
        # Calculate interpolation factor within segment
        lower_bounds = color_stops[segment_indices]
        upper_bounds = color_stops[segment_indices + 1]
        factors = (field_flat - lower_bounds) / (upper_bounds - lower_bounds)
        
        # Interpolate colors
        for i in range(len(field_flat)):
            idx = segment_indices[i]
            t = factors[i]
            color1 = np.array(palette[idx])
            color2 = np.array(palette[idx + 1])
            blended_color = color1 * (1 - t) + color2 * t
            
            # Flat index to 2D coordinates
            y, x = i // width, i % width
            result[y, x] = blended_color.astype(np.uint8)
        
        # Apply a small amount of blur for smoother transitions
        result = cv2.GaussianBlur(result, (5, 5), 0)
        
        return result
    else:
        # Use OpenCV colormap for simpler approach
        gray = (field * 255).astype(np.uint8)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_PLASMA)
        return colored

def demo_plasma_images(num_images=3, delay=1000):
    """
    Demonstrate the plasma image generator by showing multiple examples.
    
    Parameters:
    -----------
    num_images : int
        Number of images to generate
    delay : int
        Delay between images in milliseconds (use 0 to wait for key press)
    """
    for i in range(num_images):
        # Create a plasma image with random parameters
        blob_count = np.random.randint(5, 12)
        min_blob_size = np.random.randint(30, 70)
        max_blob_size = np.random.randint(100, 250)
        smoothness = np.random.randint(20, 50)
        
        plasma = create_plasma_image(
            width=800, 
            height=600, 
            blob_count=blob_count,
            min_blob_size=min_blob_size,
            max_blob_size=max_blob_size,
            smoothness=smoothness
        )
        
        # Create slices
        slices = create_image_slices(plasma, vertical=5, horizontal=4, overlap=10)
        vis_image = visualize_slices(plasma, slices)
        
        # Create a mask for demonstration
        mask = create_circular_mask(plasma.shape)
        mask_vis = visualize_mask(plasma, mask)
        
        # Show the images
        cv2.imshow("Plasma Image", plasma)
        cv2.imshow("Plasma with Slices", vis_image)
        cv2.imshow("Plasma with Mask", mask_vis)
        
        # Create an info display
        info_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(info_img, f"Image {i+1}/{num_images}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(info_img, f"Blobs: {blob_count}, Smoothness: {smoothness}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(info_img, f"Size range: {min_blob_size}-{max_blob_size}", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow("Plasma Info", info_img)
        
        # Wait
        key = cv2.waitKey(delay)
        if key == 27:  # ESC key
            break
            
    cv2.destroyAllWindows()



def crop_inward_slices(image_shape, total_percent_crop, steps, min_dimension=10):
    """
    Generator that yields slices for progressively cropping an image inward.
    
    Args:
        image_shape: Shape of the image or the image itself
        total_percent_crop: Total percentage to crop from each edge by the final step (0-50)
        steps: Number of cropping steps to perform
        min_dimension: Minimum allowed dimension (width/height) for any crop
        
    Yields:
        tuple: (y_slice, x_slice) for each cropping step
        
    Raises:
        ValueError: If parameters would result in invalid crops
    """
    # Get image dimensions
    if isinstance(image_shape, np.ndarray):
        height, width = image_shape.shape[:2]
    else:
        height, width = image_shape[:2]
    
    # Validate parameters
    if total_percent_crop <= 0 or total_percent_crop >= 50:
        raise ValueError(f"total_percent_crop must be between 0 and 50, got {total_percent_crop}")
    if steps <= 0:
        raise ValueError(f"steps must be greater than 0, got {steps}")
    
    # Calculate per-step crop percentage
    per_step_percent = total_percent_crop / steps
    
    # Check if any step would create too small crops
    smallest_height = height * (1 - 2 * (total_percent_crop / 100))
    smallest_width = width * (1 - 2 * (total_percent_crop / 100))
    
    if smallest_height < min_dimension or smallest_width < min_dimension:
        raise ValueError(f"Cropping parameters would result in dimensions smaller than {min_dimension}px " 
                         f"(would be {smallest_width:.1f}x{smallest_height:.1f})")
    
    # Start with the full image (step 0)
    yield slice(0, height), slice(0, width)
    
    # Generate progressive crops
    for step in range(1, steps + 1):
        # Calculate the crop amount for this step (as a fraction of image size)
        crop_fraction = (step * per_step_percent) / 100.0
        
        # Ensure we're not cropping beyond the image bounds
        if crop_fraction >= 0.5:  # Can't crop more than half from each side
            break
        
        # Calculate slice boundaries
        top = int(height * crop_fraction)
        bottom = int(height * (1 - crop_fraction))
        left = int(width * crop_fraction)
        right = int(width * (1 - crop_fraction))
        
        # Validate the crop dimensions
        crop_height = bottom - top
        crop_width = right - left
        
        if crop_height <= 0 or crop_width <= 0:
            raise ValueError(f"Invalid crop dimensions at step {step}: {crop_width}x{crop_height}")
        
        if crop_height < min_dimension or crop_width < min_dimension:
            break  # Stop generating crops when they become too small
        
        # Create and yield slices
        y_slice = slice(top, bottom)
        x_slice = slice(left, right)
        
        yield y_slice, x_slice

def apply_crop_slice(image, slice_pair):
    """
    Apply a crop slice to an image.
    
    Args:
        image: Input image
        slice_pair: Tuple of (y_slice, x_slice)
        
    Returns:
        numpy.ndarray: Cropped image
    """
    y_slice, x_slice = slice_pair
    return image[y_slice, x_slice]

def test_cropping_animation(num_test_cases=5):
    """
    Test function that demonstrates the crop_inward_slices generator with animated visualization.
    Runs multiple test cases with random parameters.
    
    Args:
        num_test_cases: Number of different parameter combinations to test
    """
    import random
    import time
    
    print("Starting crop animation test...")
    print("Press 'ESC' at any time to exit, any other key to proceed to next test case")
    
    # Counter for successful tests
    successful_tests = 0
    
    for test_case in range(num_test_cases):
        try:
            # Create a test image with random dimensions
            width = random.randint(400, 800)
            height = random.randint(300, 600)
            
            # Randomly choose between test_image and plasma_image
            if random.choice([True, False]):
                test_image = create_test_image(width, height)
                image_type ="Shape"
            else:
                test_image = create_plasma_image(width, height)
                image_type ="Plasma"
            
            # Random cropping parameters
            total_percent_crop = random.uniform(10, 45)  # Between 10% and 45%
            steps = random.randint(5, 15)  # Between 5 and 15 steps
            
            # Create window with parameters displayed in title
            window_name = f"Test {test_case+1}/{num_test_cases}: {image_type} Image, {width}x{height}, Crop {total_percent_crop:.1f}%, {steps} steps"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Display original image first
            cv2.imshow(window_name, test_image)
            if cv2.waitKey(500) == 27:  # Wait 500ms, exit on ESC
                break
            
            # Get crop generator
            crop_gen = crop_inward_slices(test_image, total_percent_crop, steps)
            
            # Skip the first yield (full image) since we already displayed it
            next(crop_gen)
            
            # Display each crop with timing
            for i, (y_slice, x_slice) in enumerate(crop_gen, 1):
                # Apply crop
                cropped = test_image[y_slice, x_slice]
                
                # Create info image with black border
                h, w = cropped.shape[:2]
                info_img = np.ones((h + 60, w, 3), dtype=np.uint8) * 255
                info_img[60:, :] = cropped
                
                # Add text with crop information
                current_percent = (total_percent_crop * i) / steps
                cv2.putText(
                    info_img, 
                    f"Step {i}/{steps}: Cropped {current_percent:.1f}% from each side", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 0), 
                    2
                )
                
                # Display
                cv2.imshow(window_name, info_img)
                
                # Wait 100ms between frames, break if ESC pressed
                key = cv2.waitKey(100)
                if key == 27:  # ESC
                    break
            
            # Final pause after sequence completes
            key = cv2.waitKey(1000)
            if key == 27:  # ESC
                break
                
            # Clean up
            cv2.destroyWindow(window_name)
            successful_tests += 1
            
        except Exception as e:
            print(f"Error in test case {test_case+1}: {str(e)}")
    
    # Final cleanup
    cv2.destroyAllWindows()
    print(f"\nCompleted {successful_tests} of {num_test_cases} test cases.")
    
    return successful_tests == num_test_cases

def test_crop_stress(num_iterations=20):
    """
    Stress test for the crop_inward_slices generator with extreme parameters.
    
    Args:
        num_iterations: Number of iterations to run
    """
    import random
    
    print("Starting crop stress test...")
    
    # Edge cases to test
    edge_cases = [
        # (total_percent_crop, steps)
        (0.1, 1),        # Minimal crop, minimal steps
        (49.9, 100),     # Maximum crop, many steps
        (5, 50),         # Small crop, many steps
        (45, 3),         # Large crop, few steps 
        (25, 25),        # Medium values
    ]
    
    # Track failures
    failures = []
    
    # Test edge cases first
    for i, (percent, step_count) in enumerate(edge_cases):
        try:
            # Create a test image with random dimensions
            test_image = create_test_image(600, 400)
            
            # Get slices
            slice_count = 0
            for y_slice, x_slice in crop_inward_slices(test_image, percent, step_count):
                # Apply crop to verify
                cropped = test_image[y_slice, x_slice]
                slice_count += 1
                
                # Basic validation
                assert cropped.shape[0] > 0 and cropped.shape[1] > 0, "Empty crop generated"
            
            print(f"Edge case {i+1}: {percent}%, {step_count} steps - Generated {slice_count} valid slices")
            
        except Exception as e:
            failures.append(f"Edge case {i+1} failed: {str(e)}")
    
    # Random parameter tests
    for i in range(num_iterations):
        try:
            # Create random image
            width = random.randint(10, 2000)   # Test tiny and large images
            height = random.randint(10, 2000)
            test_shape = (height, width, 3)
            
            # Random parameters, including edge values
            percent = random.uniform(0.001, 49.999)
            steps = random.randint(1, 100)
            
            # Get slices
            slice_count = 0
            for y_slice, x_slice in crop_inward_slices(test_shape, percent, steps):
                # Validate slices
                assert 0 <= y_slice.start < y_slice.stop <= height, f"Invalid y_slice: {y_slice}"
                assert 0 <= x_slice.start < x_slice.stop <= width, f"Invalid x_slice: {x_slice}"
                slice_count += 1
            
            if i % 5 == 0:  # Print only every 5th iteration to reduce output
                print(f"Random test {i+1}: {width}x{height}, {percent:.2f}%, {steps} steps - Generated {slice_count} valid slices")
            
        except Exception as e:
            failures.append(f"Random test {i+1} failed: {str(e)}")
    
    # Report results
    if failures:
        print("\nFailures detected:")
        for failure in failures:
            print(f"  - {failure}")
        print(f"\n{len(failures)} tests failed out of {len(edge_cases) + num_iterations}")
    else:
        print(f"\nAll {len(edge_cases) + num_iterations} tests passed successfully!")
    
    return len(failures) == 0


def run_crop_tests():
    """Run both animation and stress tests for the cropping generator"""
    print("=== Starting Crop Generator Tests ===\n")
    
    # Run animation test
    print("\n--- Animation Test ---")
    animation_success = test_cropping_animation(num_test_cases=3)
    
    # Run stress test
    print("\n--- Stress Test ---")
    stress_success = test_crop_stress(num_iterations=20)
    
    # Overall result
    print("\n=== Test Results ===")
    print(f"Animation Test: {'PASSED' if animation_success else 'FAILED'}")
    print(f"Stress Test: {'PASSED' if stress_success else 'FAILED'}")
    print(f"Overall: {'PASSED' if (animation_success and stress_success) else 'FAILED'}")
    
    return animation_success and stress_success



# Run the plasma image demo
# demo_plasma_images(num_images=5, delay=1000)  # 1 second delay between images

# Run the stress test with plasma images
# Uncomment to run:
# stress_test_with_plasma(num_images=100, bins_per_channel=4, show_visualizations=True)

# Run the stress test with shape images
# # Uncomment to run:
# Run the stress test with shape images
# # Uncomment to run:
# params = ImageEmbeddingParams(bins_per_channel=8)
# stress_test_embeddings(num_images=50, params=params, show_visualizations=False)

# # Verify mask functionality
# test_masked_embedding()
# # verify_mask_effect()  # Uncomment for more detailed testing

# # Run stress test
# params = ImageEmbeddingParams(bins_per_channel=8)
# stress_test_embeddings(num_images=50, params=params, show_visualizations=True)

# if __name__ == "__main__":
#     run_crop_tests()


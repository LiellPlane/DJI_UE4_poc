import cv2
import numpy as np
import math
from dataclasses import dataclass
from qdrant_utils import get_qdrant_client, get_random_item

@dataclass(frozen=True)
class ColourPoint:
    x: int
    y: int
    visual_test_colour: tuple[int, int, int]


@dataclass(frozen=True)
class EmbeddedPoint(ColourPoint):
    embedding_id: str = None


def draw_ring(img, center, inner_radius, outer_radius, color)->ColourPoint:
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
                # img[y, x] = 
                yield ColourPoint(x, y, color)


def generate_touching_pxls_coords(
    colourpoint: ColourPoint, 
    embedding_ids: dict[tuple[int, int], EmbeddedPoint]
) -> list[tuple[int, int]]:
    """Generate the coordinates of the touching pixels (8-connected neighbourhood)"""
    x, y = colourpoint.x, colourpoint.y
    touching_pixels = []
    
    # Generate the 8-connected neighbourhood (excluding the pixel itself)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            # Skip the centre pixel (the pixel itself)
            if dx == 0 and dy == 0:
                continue
            yield ((x + dx, y + dy))
            

def get_ids_in_radius(colourpoint: ColourPoint, embedding_ids: dict[tuple[int, int], EmbeddedPoint])->list[str]:
    """Get the ids of the points in the radius of the colourpoint"""
    ids = []
    for pxl in generate_touching_pxls_coords(colourpoint, embedding_ids):
        if pxl in embedding_ids:
            ids.append(embedding_ids[pxl].embedding_id)
    return ids


def draw_concentric_circles(image_size=200, num_circles=20)->tuple[np.ndarray, dict[tuple[int, int], EmbeddedPoint]]:
    # Create a white image
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    
    embedding_ids = {}
    # for testing - load the embeddings into the embedding_ids dict
    client = get_qdrant_client()
    for x,y in np.ndindex(img.shape[0:1]):
        embedding_ids[(x,y)] = get_random_item_with_closest_match(client, "colours", (x,y), 1)

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

        ring_gen = draw_ring(img, center, inner_radius, outer_radius, color)
        for colourpoint in ring_gen:
            # check if any touching points already exist
            # if so, get the average embedding of the touching points
            get_ids_in_radius(colourpoint, embedding_ids)
            

            img[colourpoint.y, colourpoint.x] = colourpoint.visual_test_colour
            # embedding_ids[(colourpoint.x, colourpoint.y)] = colourpoint


    
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

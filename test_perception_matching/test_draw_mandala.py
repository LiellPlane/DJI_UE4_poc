import cv2
import numpy as np
import math
from dataclasses import dataclass
from qdrant_utils import get_point_by_id, clone_collection, get_qdrant_client, get_random_item, get_closest_match, delete_point

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
            
def get_embedding_average(client, neighbour_ids: list[str], collection_name) -> np.ndarray:
    """Get the average embedding of the neighbour ids"""
    # Retrieve points by their IDs
    results = client.retrieve(
        collection_name=collection_name,
        ids=neighbour_ids,
        with_vectors=True
    )
    
    # Extract vectors from the retrieved points
    embeddings = [point.vector for point in results]
    
    # Return the mean of embeddings if any exist, otherwise return None
    if embeddings:
        # this will work for histogram embeddings - but
        # potentially not as well for other embedding types
        return np.mean(embeddings, axis=0)
    return None

def get_ids_in_radius(colourpoint: ColourPoint, 
                      embedding_ids: dict[tuple[int, int], EmbeddedPoint]) -> list[str]:
    """Get the ids of the points in the radius of the colourpoint"""
    ids = []
    for pxl in generate_touching_pxls_coords(colourpoint, embedding_ids):
        if pxl in embedding_ids:
            ids.append(embedding_ids[pxl].embedding_id)
    return ids


def draw_concentric_circles(client, collection_name, image_size=20, num_circles=10)->tuple[np.ndarray, dict[tuple[int, int], EmbeddedPoint]]:
    # Create a white image
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    
    embedding_ids = {}
    # for testing - load the embeddings into the embedding_ids dict
    
    # for x, y in np.ndindex(img.shape[0:2]):
    #     print(x, y)
    #     embedding_ids[(x, y)] = EmbeddedPoint(
    #         embedding_id=   get_random_item(client, "colours").id,
    #         x=x,
    #         y=y,
    #         visual_test_colour=(0, 0, 0)
    #     )

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
        print(f"Drawing circle {i+1} of {num_circles}")
        # Each ring is exactly 1 pixel thick
        inner_radius = 1 + i  # Start just outside the center dot
        outer_radius = inner_radius + 1
        # Cycle through the 5 colors
        color = colors[i % len(colors)]

        ring_gen = draw_ring(img, center, inner_radius, outer_radius, color)
        id = None
        for colourpoint in ring_gen:
            # check if any touching points already exist
            # if so, get the average embedding of the touching points
            neighbour_ids = get_ids_in_radius(colourpoint, embedding_ids)
            if len(neighbour_ids) == 0:
                print("No neighbour ids found")
                id = get_random_item(client=client, collection_name=collection_name).id
            elif len(neighbour_ids) == 1:
                print("One neighbour id found")
                point = get_point_by_id(client=client, collection_name=collection_name, point_id=neighbour_ids[0])
                
                res = get_closest_match(
                    client=client,
                    collection_name=collection_name,
                    vector=neighbour_ids[0],
                    limit=1,
                    with_payload=False,
                    with_vectors=False
                    )
                id = res[0].id
            elif len(neighbour_ids) > 1:
                print(f"{len(neighbour_ids)} neighbour ids found")
                embedding_average = get_embedding_average(client, neighbour_ids, collection_name)
                res =get_closest_match(
                    client=client,
                    collection_name=collection_name,
                    vector=embedding_average,
                    limit=1,
                    with_payload=False,
                    with_vectors=False
                    )
                id = res[0].id
            else:
                raise ValueError(f"Unexpected number of neighbour ids: {len(neighbour_ids)}")
                
            delete_point(client=client, collection_name=collection_name, point_id=id)
            
            embedding_ids[(colourpoint.x, colourpoint.y)] = EmbeddedPoint(
                embedding_id=id,
                x=colourpoint.x,
                y=colourpoint.y,
                visual_test_colour=color
            )

            if colourpoint.x < img.shape[0] and colourpoint.y < img.shape[1]:
                img[colourpoint.y, colourpoint.x] = colourpoint.visual_test_colour
            else:
                break
            # embedding_ids[(colourpoint.x, colourpoint.y)] = colourpoint


    
    return img

def main():

    client = get_qdrant_client()
    clone_collection(client, collection_name="colours", new_collection_name="colours_clone")

    # Create the image with concentric circles
    img = draw_concentric_circles(client, "colours_clone")
    
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

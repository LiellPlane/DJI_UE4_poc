import cv2
import numpy as np
import math
from dataclasses import dataclass
from itertools import chain
# Extremely aggressive Python path handling
import os
import sys
import site
from pathlib import Path
import platform
from qdrant_utils import wait_for_collection_ready, get_point_by_id, clone_collection, get_qdrant_client, get_random_item, get_closest_match, delete_point, async_delete_point
import get_batch_embeddings
import random
import threading
import queue
import time
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, field_validator
import asyncio
import test_async_qdrant
import get_image_embedding
@dataclass(frozen=True)
class ColourPoint:
    x: int
    y: int
    visual_test_colour: tuple[int, int, int]


@dataclass(frozen=True)
class EmbeddedPoint(ColourPoint):
    embedding_id: str = None
    local_file_path: str = None

class SeedEmbedding(BaseModel):
    """
    if we want to seed the mandala with a specific image
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    embedding: np.ndarray
    flipped: bool
    image_path: str

    @field_validator('embedding')
    @classmethod
    def validate_numpy_array(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError('Must be a numpy array')
        return v


def draw_ring(center, inner_radius, outer_radius, color)->ColourPoint:
    """Draw a ring using scanline fill approach"""
    x0, y0 = center
    if inner_radius == 0:
        # special case - draw a single pixel
        yield ColourPoint(x0, y0, color)
 
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
    # dx = 0
    # for dy in [-1, 0, 1]:
    #     # Skip the centre pixel (the pixel itself)
    #     if dx == 0 and dy == 0:
    #         continue
    #     yield ((x + dx, y + dy))


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
                      embedding_ids: dict[tuple[int, int], EmbeddedPoint | SeedEmbedding]) -> list[str | np.ndarray]:
    """Get the ids of the points in the radius of the colourpoint"""
    ids = []
    for pxl in generate_touching_pxls_coords(colourpoint, embedding_ids):
        if pxl in embedding_ids: # if there is data there, use it
            if isinstance(embedding_ids[pxl], EmbeddedPoint):
                ids.append(embedding_ids[pxl].embedding_id)
            elif isinstance(embedding_ids[pxl], SeedEmbedding):
                ids.append(embedding_ids[pxl].embedding)
            else:
                raise ValueError(f"Unexpected type: {type(embedding_ids[pxl])}")
                
    return ids


def backup_draw_concentric_circles(client, collection_name, read_only_collection_name, image_size=300, num_circles=9)->tuple[np.ndarray, dict[tuple[int, int], EmbeddedPoint]]:
    # Create a white image
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    
    embedding_ids = {}

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
    images_exhausted: bool = False
    for i in range(-1, num_circles): # we use -1 as special case to draw the centre dot
        print(f"Drawing circle {i+1} of {num_circles}")
        # Each ring is exactly 1 pixel thick
        inner_radius = 1 + i  # Start just outside the center dot
        outer_radius = inner_radius + 1
        # Cycle through the 5 colors
        color = colors[i % len(colors)]

        ring_gen = draw_ring(center, inner_radius, outer_radius, color)
        id = None
        filepath = None
        score =None
        for colourpoint in ring_gen:
            # check if any touching points already exist
            # if so, get the average embedding of the touching points
            neighbour_ids = get_ids_in_radius(colourpoint, embedding_ids)
            if len(neighbour_ids) == 0:
                print("No neighbour ids found")
                try:
                    point = get_random_item(client=client, collection_name=collection_name)
                    id = point.id
                    filepath = point.payload["filename"]
                    score = point.score
                except ValueError as e:
                    print(f"Error getting random item: {e}")
                    images_exhausted = True
                    break
            elif len(neighbour_ids) > 0:
                # print(f"{len(neighbour_ids)} neighbour ids found")
                embedding_average = get_embedding_average(client, neighbour_ids, read_only_collection_name)
                res = get_closest_match(
                    client=client,
                    collection_name=collection_name,
                    vector=embedding_average,
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                    )

                if len(res) == 0:
                    images_exhausted = True
                    break
                id = res[0].id
                filepath = res[0].payload["filename"]
                score = res[0].score
            else:
                raise ValueError(f"Unexpected number of neighbour ids: {len(neighbour_ids)}")
                
            delete_point(client=client, collection_name=collection_name, point_id=id)
            
            if (colourpoint.x, colourpoint.y) not in embedding_ids:
                embedding_ids[(colourpoint.x, colourpoint.y)] = EmbeddedPoint(
                    embedding_id=id,
                    local_file_path=filepath,
                    x=colourpoint.x,
                    y=colourpoint.y,
                    visual_test_colour=color
                )
            else:
                print(f"Skipping pixel at ({colourpoint.x}, {colourpoint.y})")

            if colourpoint.x < img.shape[0] and colourpoint.y < img.shape[1]:
                # anomaly due to circle drawing errors - which may cause poor matching/artifacts

                # Create color gradient: RED (0,0,255) for score=0 to GREEN (0,255,0) for score=1
                clamped_score = max(0, min(1, score))
                green = int(255 * clamped_score)
                red = int(255 * (1 - clamped_score))
                similarity_colour = (0, green, red)  # BGR format in OpenCV
                img[colourpoint.y, colourpoint.x] = similarity_colour
                
                if random.random() < 0.01:
                    # Zoom the image by 4x
                    zoomed_img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                    
                    # Crop to non-default pixels (white)
                    # Find all non-white pixels
                    non_default_pixels = np.where(np.any(zoomed_img != 255, axis=2))
                    
                    # Check if there are any non-default pixels
                    if len(non_default_pixels[0]) > 0:
                        # Get the bounding box
                        min_y, max_y = np.min(non_default_pixels[0]), np.max(non_default_pixels[0])
                        min_x, max_x = np.min(non_default_pixels[1]), np.max(non_default_pixels[1])
                        
                        # Add padding (10 pixels)
                        padding = 10
                        min_y = max(0, min_y - padding)
                        min_x = max(0, min_x - padding)
                        max_y = min(zoomed_img.shape[0] - 1, max_y + padding)
                        max_x = min(zoomed_img.shape[1] - 1, max_x + padding)
                        
                        # Crop the image
                        zoomed_img = zoomed_img[min_y:max_y+1, min_x:max_x+1]
                    
                    # Display the image
                    cv2.imshow('Concentric Circles', zoomed_img)
                    cv2.waitKey(1)

            else:
                break
            # embedding_ids[(colourpoint.x, colourpoint.y)] = colourpoint

        if images_exhausted:
            print("Images exhausted")
            break

    return img, embedding_ids




async def draw_concentric_circles(client, collection_name, read_only_collection_name, image_size=300, num_circles=7, seeds: list[SeedEmbedding]=[])->tuple[np.ndarray, dict[tuple[int, int], EmbeddedPoint]]:
    
    
    client = test_async_qdrant.FakeQdrantClient(collection_name="test_vectors")
    real_client = test_async_qdrant.qdrant_client.AsyncQdrantClient("localhost")
    # Create task handler
    handler = test_async_qdrant.AsyncTaskHandler(
        depleting_collection_name=collection_name,
        read_only_collection_name=read_only_collection_name,
        real_client=real_client, 
        fake_client=client# Add a small delay to make sequential processing more visible
    )    
    


    # Create a white image
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    
    embedding_ids = {}

    # Get center coordinates
    center = (image_size // 2, image_size // 2)
    

    if seeds:
        # plop in the seed if they exist. space them out if more than one but for now just one
        embedding_ids[center] = seeds[0]

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
    janky_start= -2 # to bodyswerve issues with the first couple circles

    for i in range(janky_start, num_circles): # we use -1 as special case to draw the centre dot
        start_time = time.time()
        print(f"Drawing circle {i+1} of {num_circles}")
        # Each ring is exactly 1 pixel thick
        inner_radius = 1 + i  # Start just outside the center dot
        outer_radius = inner_radius + 1
        # Cycle through the 5 colors
        color = colors[i % len(colors)]

        if i == janky_start:
            ring_gen = draw_ring(center, 0, 0, color)
        else:
            ring_gen = draw_ring(center, inner_radius, outer_radius, color)

        # get the ids of the points in the radius of the colourpoint
        # this will be used to pass to the async worker to get the average embedding
        # bear in mind that this should be alternating or we will miss averages from neighbours on the same circle
        ids_per_circle_point = {
            (colourpoint.x, colourpoint.y): get_ids_in_radius(colourpoint, embedding_ids)
            for colourpoint in ring_gen
            if (colourpoint.x, colourpoint.y) not in embedding_ids
            }
        # filter out already calculated points
        # ids_per_circle_point = {id:val for id,val in ids_per_circle_point.items() if id not in embedding_ids}

        # Choose whether to process each point one-by-one sequentially or in parallel
        # use_sequential_processing = True  # Set to True for sequential processing, False for parallel

        if i > 8:# for first circles we want best matches - so strictly sequential to
            # avoid complications with duplicate ids. Once farther apart it should in theory be less
            # of an issue
            use_sequential_processing = False
            sequence = []
            sequence.append({key: ids_per_circle_point[key] for key in list(ids_per_circle_point.keys())[::2]})
            sequence.append({key: ids_per_circle_point[key] for key in list(ids_per_circle_point.keys())[1::2]})
        else:
            use_sequential_processing = True
            sequence = [{key_:val_} for key_, val_ in ids_per_circle_point.items()]

        # if len(ids_per_circle_point) != len(sequence[0]) + len(sequence[1]):
        #     raise ValueError(f"Odd and evens do not match: {len(ids_per_circle_point)} != {len(odds_and_evens[0])} + {len(odds_and_evens[1])}") 
        neighbours: dict[tuple[int, int], list[str]]
        for neighbours in sequence:
            # Pass force_sequential parameter to process embeddings
            results: list[test_async_qdrant.TaskResult | Exception] = await handler.process_embeddings(
                results_limit=len(neighbours),
                neighbour_ids=neighbours,
                force_sequential=use_sequential_processing,
                delete_after_processing=True  # Delete each point immediately after processing - sequential only
            )
            
            # We now have a bunch of candidate ids for each point around circle (or whatever the interval is)
            # Now remove duplicates

            used_ids = set()
            results_deduped: list[ test_async_qdrant.TaskResult] = []
            for result in results:
                if isinstance(result, test_async_qdrant.TaskResult):
                    for index, id in enumerate(result.embedding_id):
                        if id not in used_ids:
                            results_deduped.append(test_async_qdrant.TaskResult(
                                coord=result.coord,
                                embedding_id=[result.embedding_id[index]],
                                local_file_path=[result.local_file_path[index]],
                                score=[result.score[index]]
                            ))  
                            used_ids.add(id)
                            break
                else:
                    results_deduped.append(result)
            

            results = results_deduped
            # we should now have the coordinate and embedding details for that coordinate. load it into the object
            flattened_embedding_ids = [id for result in results if isinstance(result, test_async_qdrant.TaskResult) for id in result.embedding_id]
            if len(set(flattened_embedding_ids)) != len(flattened_embedding_ids):
                print(f" {len(flattened_embedding_ids) - len(set(flattened_embedding_ids))} Duplicate ids")

            # here we have results, each result has a list of ids (n closest match)
            # but each seperate result may have duplicates
            # we want an algorithm that will let each result have only one id with no duplicates
            # we request closest matches to handle duplicates so we can eliminate them
            for result in results:
                if isinstance(result, test_async_qdrant.TaskResult):
                    # embedding_ids[result.coord] = result
                    embedding_ids[result.coord] = EmbeddedPoint(
                        embedding_id=result.embedding_id[0],
                        local_file_path=result.local_file_path[0],
                        x=result.coord[0],
                        y=result.coord[1],
                        visual_test_colour=color
                    )

                if not [i for i in results if isinstance(i, Exception)]:
                    if result.coord[0] < img.shape[0] and result.coord[1] < img.shape[1]:
                        # anomaly due to circle drawing errors - which may cause poor matching/artifacts

                        # Create color gradient: RED (0,0,255) for score=0 to GREEN (0,255,0) for score=1
                        clamped_score = max(0, min(1, result.score[0]))
                        green = int(255 * clamped_score)
                        red = int(255 * (1 - clamped_score))
                        similarity_colour = (0, green, red)  # BGR format in OpenCV
                        img[result.coord[1], result.coord[0]] = similarity_colour
                        
                        if random.random() < 0.01:
                            # Zoom the image by 4x
                            zoomed_img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                            
                            # Crop to non-default pixels (white)
                            # Find all non-white pixels
                            non_default_pixels = np.where(np.any(zoomed_img != 255, axis=2))
                            
                            # Check if there are any non-default pixels
                            if len(non_default_pixels[0]) > 0:
                                # Get the bounding box
                                min_y, max_y = np.min(non_default_pixels[0]), np.max(non_default_pixels[0])
                                min_x, max_x = np.min(non_default_pixels[1]), np.max(non_default_pixels[1])
                                
                                # Add padding (10 pixels)
                                padding = 10
                                min_y = max(0, min_y - padding)
                                min_x = max(0, min_x - padding)
                                max_y = min(zoomed_img.shape[0] - 1, max_y + padding)
                                max_x = min(zoomed_img.shape[1] - 1, max_x + padding)
                                
                                # Crop the image
                                zoomed_img = zoomed_img[min_y:max_y+1, min_x:max_x+1]
                            
                            # Display the image
                            cv2.imshow('Concentric Circles', zoomed_img)
                            cv2.waitKey(1)

            # Check for exceptions
            if [i for i in results if isinstance(i, Exception)]:
                print(f"Circle {i+1} failed: probably ran out of images")
                print(f"Results: {results}")
                break
            break_ = False
            for result in results:
                if break_:
                    break
                if isinstance(result, Exception):
                    print(f"Circle {i+1} failed: probably ran out of images")
                    print(f"result: {result}")
                    break_ = True
                
            # In parallel mode, we need to handle deletion separately
            # Sequential mode already handles deletion during processing
            if not use_sequential_processing:
                # Only do bulk deletion for parallel processing
                # point_ids = [res.embedding_id for res in results if isinstance(res, test_async_qdrant.TaskResult)]
                point_ids = [id for result in results if isinstance(result, test_async_qdrant.TaskResult) for id in result.embedding_id]
                if point_ids:
                    print(f"Parallel mode: Deleting {len(point_ids)} points in bulk")
                    await async_delete_point(
                        client=real_client, 
                        collection_name=collection_name, 
                        point_ids=point_ids
                    )

        end_time = time.time()
        elapsed_time = end_time - start_time
        seconds = int(elapsed_time)
        milliseconds = int((elapsed_time - seconds) * 1000)
        print(f"Circle {i+1} completed in {seconds} seconds and {milliseconds} milliseconds")

    return img, embedding_ids



def embedding_worker(task_queue, result_queue):
    while True:
        try:
            task = task_queue.get(timeout=0.5)
            try:
                # Process task (placeholder)
                print(f"Processing task {task.task_id}")

                # Placeholder for actual processing
                result = ResultData(
                    task_id=task.task_id,
                    result={"processed": True}
                )
                
                # Add to result queue
                result_queue.put(result)
                
            except Exception as e:
                print(f"Error processing task {task.task_id}: {e}")
            finally:
                task_queue.task_done()
                
        except queue.Empty:
            # No tasks available
            continue


def draw_concentric_circles_multithreaded(client, collection_name, read_only_collection_name, image_size=300, num_circles=9)->tuple[np.ndarray, dict[tuple[int, int], EmbeddedPoint]]:


    # Pydantic models for input and output data
    class TaskData(BaseModel):
        """Simple placeholder for task data"""
        task_id: int
        data: dict

    class ResultData(BaseModel):
        """Simple placeholder for result data"""
        task_id: int
        result: dict
        success: bool = True

    # Create task queue and result queue
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Flag to signal workers to stop
    stop_event = threading.Event()
    
    # Worker function
    def worker():
        while not stop_event.is_set():
            try:
                # Get task with timeout
                task = task_queue.get(timeout=0.5)
                
                try:
                    # Process task (placeholder)
                    print(f"Processing task {task.task_id}")
                    import time
                    time.sleep(1)
                    # Placeholder for actual processing
                    result = ResultData(
                        task_id=task.task_id,
                        result={"processed": True}
                    )
                    
                    # Add to result queue
                    result_queue.put(result)
                    
                except Exception as e:
                    print(f"Error processing task {task.task_id}: {e}")
                finally:
                    task_queue.task_done()
                    
            except queue.Empty:
                # No tasks available
                continue
    
    # Automatically determine optimal number of threads for I/O bound tasks
    cpu_count = os.cpu_count() or 4  # Default to 4 if detection fails
    num_workers = min(32, cpu_count * 2)  # Use 2x CPU cores, capped at 32
    
    print(f"Using {num_workers} worker threads")
    threads = []
    
    for i in range(num_workers):
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Placeholder: Add tasks to queue
    for i in range(10):  # Example: 10 tasks
        task = TaskData(task_id=i, data={"example": "data"})
        task_queue.put(task)
    
    # Wait for all tasks to complete
    task_queue.join()
    
    # Stop all workers
    stop_event.set()
    for thread in threads:
        thread.join()
    
    # Process results (placeholder)
    results = []
    while not result_queue.empty():
        result = result_queue.get()
        results.append(result)
    
    print(f"Processed {len(results)} tasks")
    
    # Placeholder return values
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    embedding_ids = {}
    
    return img, embedding_ids


def create_mandala_from_similarity_matrix(
    similarity_matrix: dict[tuple[int, int], EmbeddedPoint],
    tile_size: int = 30
) -> np.ndarray:
    """Create a mandala from a similarity matrix"""
    # Find boundaries of the grid
    x_coords = [k[0] for k in similarity_matrix.keys()]
    y_coords = [k[1] for k in similarity_matrix.keys()]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    print(f"Grid boundaries: X: {min_x} to {max_x}, Y: {min_y} to {max_y}")
    
    # Calculate canvas dimensions
    width = (max_x - min_x + 1) * tile_size
    height = (max_y - min_y + 1) * tile_size
    
    print(f"Mandala dimensions: {width} x {height} pixels")
    
    # Create canvas for the mandala
    mandala = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Place each tile on the canvas
    total_tiles = len(similarity_matrix)
    processed_tiles = 0
    
    print(f"Processing {total_tiles} tiles...")
    
    for (x, y), point in similarity_matrix.items():
        processed_tiles += 1
        if processed_tiles % 100 == 0:
            print(f"Processed {processed_tiles}/{total_tiles} tiles")
            

        if isinstance(point, SeedEmbedding):
            img_path = point.image_path
        if isinstance(point, EmbeddedPoint):
            img_path = point.local_file_path
        # Skip if the file path is not available
        # if not point.local_file_path:
        #     print(f"Skipping tile at ({x}, {y}): No file path")
        #     continue
            
        try:
            # Load image
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
                
            # Resize to tile_size x tile_size
            resized_img = cv2.resize(img, (tile_size, tile_size), 
                                    interpolation=cv2.INTER_AREA)
            
            # Calculate position in the mandala
            pos_x = (x - min_x) * tile_size
            pos_y = (y - min_y) * tile_size
            
            # Place the tile
            mandala[pos_y:pos_y+tile_size, pos_x:pos_x+tile_size] = resized_img
            
        except Exception as e:
            print(f"Error processing image at ({x}, {y}): {e}")
    
    print(f"Completed processing {processed_tiles}/{total_tiles} tiles")
    
    # Resize if necessary to fit within 2000 x 2000
    max_dimension = 2000
    actual_height, actual_width = mandala.shape[:2]
    
    if actual_height > max_dimension or actual_width > max_dimension:
        print(f"Resizing mandala to fit within {max_dimension}x{max_dimension}")
        
        # Calculate scaling factor to fit within max dimensions
        scale = min(max_dimension / actual_width, max_dimension / actual_height)
        
        new_width = int(actual_width * scale)
        new_height = int(actual_height * scale)
        
        print(f"New dimensions: {new_width} x {new_height}")
        
        # Resize the mandala
        mandala = cv2.resize(mandala, (new_width, new_height), 
                           interpolation=cv2.INTER_AREA)
    
    # Create a temporary display image (scaled down version for quick viewing)
    display_max_dim = 800
    display_scale = min(display_max_dim / mandala.shape[1], display_max_dim / mandala.shape[0])
    
    if display_scale < 1:
        display_width = int(mandala.shape[1] * display_scale)
        display_height = int(mandala.shape[0] * display_scale)
        display_image = cv2.resize(mandala, (display_width, display_height), 
                                 interpolation=cv2.INTER_AREA)
        print(f"Created display image: {display_width} x {display_height}")
    else:
        display_image = mandala.copy()
        print("Using original size for display image")
    
    # Save the display image
    cv2.imwrite('mandala_display.png', display_image)
    print("Saved display image as 'mandala_display.png'")
    
    return mandala


    
async def async_main():
    read_only_collection_name = "everything_s_liellvector"
    clone_collection_name = f"{read_only_collection_name}_clone"
    client = get_qdrant_client()
    # Detect operating system and set appropriate paths
    if platform.system() == "Darwin":  # macOS
        image_paths = get_batch_embeddings.get_image_filepaths_from_folders([Path("D:/match_images_mandala")])
    elif platform.system() == "Windows":
        image_paths = get_batch_embeddings.get_image_filepaths_from_folders([Path("D:/match_images_mandala")])

    if len(image_paths) > 0:
        image_path=image_paths[0]
        embedding, embedding_flipped = get_image_embedding.get_image_embedding(client, image_path, read_only_collection_name)
        # get closest matches so we can see if using a flipped image gives better results
        sorted_files = get_image_embedding.get_closest_match(client, embedding, embedding_flipped, read_only_collection_name)

        seed_embedding = [SeedEmbedding(embedding=embedding_flipped if sorted_files[0].is_flipped else embedding, flipped=sorted_files[0].is_flipped, image_path=image_path)]
    else:
        seed_embedding = []

    clone_collection(client, collection_name=read_only_collection_name, new_collection_name=clone_collection_name)
    wait_for_collection_ready(client, clone_collection_name)

    # img, similarity_matrix = draw_concentric_circles_multithreaded(client, collection_name=clone_collection_name, read_only_collection_name=read_only_collection_name)
    1/0
    # 1/0
    # Create the image with concentric circles
    img, similarity_matrix = await draw_concentric_circles(client, collection_name=clone_collection_name, read_only_collection_name=read_only_collection_name, num_circles=14, seeds=seed_embedding)
    
    mandala = create_mandala_from_similarity_matrix(similarity_matrix)
    
    # Display the mandala
    try:
        print("Displaying mandala (press any key to continue)...")
        cv2.imshow('Mandala', cv2.resize(mandala, (800, 800), 
                                       interpolation=cv2.INTER_AREA))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error displaying mandala: {e}")
        
    print("Saving mandala as 'mandala.png'...")
    cv2.imwrite('mandala.png', mandala)
    print("Mandala saved successfully")
    
    plop=1


def main():
    import asyncio
    asyncio.run(async_main())

if __name__ == "__main__":
    main()

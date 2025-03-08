import os
import pathlib
from typing import List, Optional
import multiprocessing as mp
import cv2
import time
import generate_embeddings
import numpy as np
from dataclasses import dataclass

@dataclass
class HSEmbeddingResult:
    """Data class for storing successful embedding results"""
    filename: str
    shape: str
    embedding: np.ndarray
    mask: bool
    params: generate_embeddings.ImageEmbeddingParams


def get_image_filepaths() -> List[str]:
    """
    Get absolute filepaths of all images in the 'test_images' directory
    which is located in the same directory as this script.
    
    Returns:
        List[str]: List of absolute paths to image files
    """
    # Get the absolute path of the current script
    current_file_path = pathlib.Path(__file__).resolve()
    # Get the directory containing the script
    current_dir = current_file_path.parent
    # Build path to the test_images directory
    test_images_dir = current_dir / "test_images"
    
    # Check if the directory exists
    if not test_images_dir.exists():
        raise FileNotFoundError(f"The directory {test_images_dir} does not exist")
    
    # Common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    # Get all image files in the directory
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(test_images_dir.glob(f"*{ext}")))
        image_files.extend(list(test_images_dir.glob(f"*{ext.upper()}")))  # For uppercase extensions
    
    # Convert Path objects to strings
    return [str(file_path) for file_path in image_files]


def load_image(filepath):
    """
    Load an image from a filepath into memory using OpenCV.
    
    Args:
        filepath (str): Path to the image file
        
    Returns:
        tuple: (filepath, image_object)
    """
    try:
        # Using OpenCV to load the image
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError(f"Failed to load image, returned None")
        return filepath, img
    except Exception as e:
        return filepath, None


def worker(queue_in, queue_out):
    """
    Worker process that pulls image paths from the input queue,
    loads the images, and puts a message on the output queue.
    
    Args:
        queue_in: Queue to get image paths from
        queue_out: Queue to put processed results to
    """
    while True:
        filepath = queue_in.get()
        if filepath is None:  # None is our signal to stop
            break
        
        try:
            # Process the image
            _, img = load_image(filepath)
            if img is not None:
                # Include image shape in the output message
                filename = os.path.basename(filepath)
                height, width, channels = img.shape  # OpenCV images have shape (height, width, channels)
                mask = generate_embeddings.create_circular_mask(img.shape)
                embedding = generate_embeddings.create_image_embedding(
                    img, 
                    generate_embeddings.ImageEmbeddingParams(),
                    mask
                )
                
                # Create an EmbeddingResult with the numpy array
                result = HSEmbeddingResult(
                    filename=filename,
                    shape=f"{height}x{width}x{channels}",
                    embedding=embedding,
                    mask=True,
                    params=generate_embeddings.ImageEmbeddingParams()
                )
                queue_out.put(result)
            else:
                # Send the actual exception for failed images
                queue_out.put(ValueError(f"Failed to load image: {os.path.basename(filepath)}"))
        except Exception as e:
            # Send the actual exception that occurred
            queue_out.put(e)
    
    # Let the output queue know this worker is done
    queue_out.put(None)


def producer(image_paths, queue_in, num_processes):
    """
    Producer function that feeds image paths into the input queue.
    
    Args:
        image_paths: List of image file paths
        queue_in: Queue to put paths into
        num_processes: Number of worker processes (for termination signals)
    """
    for path in image_paths:
        queue_in.put(path)
    
    # Send termination signal to each worker
    for _ in range(num_processes):
        queue_in.put(None)


def main():
    """
    Main function to demonstrate parallel image loading with a queue
    """
    try:
        # Get all image filepaths
        image_paths = get_image_filepaths()
        original_count = len(image_paths)
        print(f"Found {original_count} images in test_images directory")
        
        # Simple replication to create ~500,000 paths (all pointing to real files)
        if original_count > 0:
            image_paths = image_paths * (9000 // original_count + 1)
        
        total_images = len(image_paths)
        print(f"Created test dataset with {total_images} image paths")
        
        # Determine number of processes to use (leave one core for the main process)
        num_processes = max(1, mp.cpu_count() - 3)
        print(f"Starting {num_processes} worker processes")
        
        # Create input and output queues, both with maximum size of 5
        queue_in = mp.Queue(maxsize=5)
        queue_out = mp.Queue(maxsize=5)  # Output queue also blocks at size 5
        
        # Start worker processes
        processes = []
        for _ in range(num_processes):
            p = mp.Process(target=worker, args=(queue_in, queue_out))
            p.daemon = True  # Process will terminate when main process exits
            p.start()
            processes.append(p)
        
        # Start producer in a separate process to avoid blocking
        producer_process = mp.Process(
            target=producer, 
            args=(image_paths, queue_in, num_processes)
        )
        producer_process.start()
        
        # Collect results from the output queue with progress tracking
        results = []
        completed_workers = 0
        start_time = time.time()
        last_update_time = start_time
        update_interval = 1.0  # Update progress every second
        
        while completed_workers < num_processes:
            try:
                # Use a small timeout to allow for regular progress updates
                result = queue_out.get(timeout=0.1)
                
                if result is None:
                    completed_workers += 1
                elif isinstance(result, Exception):
                    # Simply handle exceptions by type
                    print(f"\nError: {type(result).__name__}: {str(result)}")
                    results.append(f"error: {type(result).__name__}")
                elif isinstance(result, HSEmbeddingResult):
                    # Handle successful results
                    results.append(result)
                else:
                    # Handle any other type (like legacy string messages)
                    results.append(result)
                
                # Update progress periodically
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    elapsed_time = current_time - start_time
                    processed_count = len(results)
                    percent_complete = (processed_count / total_images) * 100
                    
                    if elapsed_time > 0:
                        rate = processed_count / elapsed_time
                        eta = (total_images - processed_count) / rate if rate > 0 else 0
                        
                        # Clear line and print updated progress
                        print(f"\rProgress: {percent_complete:.1f}% ({processed_count}/{total_images}) " 
                              f"| Rate: {rate:.1f} img/s | ETA: {eta:.1f}s ", end="")
                    
                    last_update_time = current_time
            
            except mp.queues.Empty:
                # Queue is empty but workers might still be processing
                continue
        
        # Print newline after progress updates
        print()
        
        # Wait for all processes to finish
        producer_process.join()
        for p in processes:
            p.join()
        
        # Final statistics
        total_time = time.time() - start_time
        print(f"\nCompleted processing {len(results)} images in {total_time:.2f} seconds")
        print(f"Average processing rate: {len(results)/total_time:.1f} images/second")
        
        # Print a sample of results
        if results:
            print("\nSample of processed files:")
            for result in results[:5]:
                if isinstance(result, HSEmbeddingResult):
                    print(f"  - Success: {result.filename} | Shape: {result.shape} | "
                          f"Embedding shape: {result.embedding.shape}")
                elif isinstance(result, Exception):
                    print(f"  - Error: {type(result).__name__}: {str(result)}")
                else:
                    print(f"  - {result}")
        
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()



import os
import pathlib
import multiprocessing as mp
import threading
import queue
import cv2
import time
import gc
import numpy as np
from dataclasses import dataclass
import generate_embeddings
import psutil
import pickle
import uuid  # Added UUID library
from pathlib import Path
import random

USE_DINO = False

if USE_DINO is True:
    import dinvo2_embeddings as dino_embeddings

@dataclass
class HSEmbeddingResult:
    """Data class for storing successful embedding results"""
    filename: str
    embedding: np.ndarray
    mask: bool
    params: generate_embeddings.ImageEmbeddingParams
    uuid: str
    

def get_image_filepaths_from_folders(target_folders: list[str]) -> list[str]:
    """
    Find all image files in the specified folders and their subfolders.
    
    Args:
        target_folders (list[str]): List of folder paths to search for images
    
    Returns:
        list[str]: List of absolute paths to all found image files
    """
    # Common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    # Final list of all image paths
    all_image_files = []
    
    for folder in target_folders:
        folder_path = pathlib.Path(folder).resolve()
        
        # Check if the folder exists
        if not folder_path.exists():
            print(f"Warning: The directory {folder_path} does not exist, skipping")
            continue
        
        if not folder_path.is_dir():
            print(f"Warning: {folder_path} is not a directory, skipping")
            continue
            
        # Use a set to prevent duplicate files
        found_image_files = set()
        for ext in image_extensions:
            # Search for extensions in a case-insensitive way (once per extension)
            files = list(folder_path.glob(f"**/*{ext}")) + list(folder_path.glob(f"**/*{ext.upper()}"))
            # Convert to absolute paths and add to set to eliminate duplicates
            found_image_files.update([str(f.absolute()) for f in files])

        # Add the files from this folder to our total list
        all_image_files.extend([Path(f) for f in found_image_files])
    
    # Convert Path objects to strings and return
    return [str(file_path) for file_path in all_image_files]

def get_image_filepaths_localtest() -> list[str]:
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
    test_images_dir = current_dir / "test_images_colour_seq"
    
    # Check if the directory exists
    if not test_images_dir.exists():
        raise FileNotFoundError(f"The directory {test_images_dir} does not exist")
    
    # Common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    # Get all image files in the directory
    image_files = []
    for ext in image_extensions:
        # Use a case-insensitive approach
        lowercase_files = set(str(f) for f in test_images_dir.glob(f"*{ext}"))
        uppercase_files = set(str(f) for f in test_images_dir.glob(f"*{ext.upper()}"))
        # Combine both sets to eliminate duplicates
        unique_files = lowercase_files.union(uppercase_files)
        image_files.extend([Path(f) for f in unique_files])
    
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
    # Create image embedding parameters - explicitly define all parameters
    params = generate_embeddings.ImageEmbeddingParams(
        vertical=5,
        horizontal=5,
        overlap=10,
        bins_per_channel=6,
        center_histograms=True,
        mask=False
    )
    
    while True:
        filepath = queue_in.get()
        if filepath is None:  # None is our signal to stop
            break
        
        try:
            # Process the image
            _, img = load_image(filepath)
            if img is not None:
                # Include image shape in the output message
                height, width, channels = img.shape
                
                # Set mask to None for now (can be customized if needed)
                mask = None
                if params.mask is True:
                    mask = generate_embeddings.create_circular_mask(img.shape)

                # Create embedding using explicit parameters
                embedding = generate_embeddings.create_image_embedding(
                    img, 
                    params=params,
                    mask=mask
                )
                # print(len(embedding))
                # Create an EmbeddingResult with the numpy array
                result = HSEmbeddingResult(
                    filename=filepath,
                    embedding=embedding,
                    mask=mask is not None,
                    params=params.to_json(),
                    uuid=str(uuid.uuid4())
                )
                queue_out.put(result)
            else:
                # Send the actual exception for failed images
                queue_out.put(ValueError(f"Failed to load image: {filepath}"))
        except Exception as e:
            # Send the actual exception that occurred
            queue_out.put(e)
    
    # Let the output queue know this worker is done
    queue_out.put(None)

def worker_dino(queue_in, queue_out):
    """
    Worker process that pulls image paths from the input queue,
    loads the images, and puts a message on the output queue.
    
    Args:
        queue_in: Queue to get image paths from
        queue_out: Queue to put processed results to
    """
    # Create image embedding parameters - explicitly define all parameters
    params = generate_embeddings.ImageEmbeddingParams(
        vertical=5,
        horizontal=5,
        overlap=10,
        bins_per_channel=6,
        center_histograms=True,
        mask=False
    )
    
    while True:
        filepath = queue_in.get()
        if filepath is None:  # None is our signal to stop
            break
        
        try:
            # Process the image
            _, img = load_image(filepath)
            if img is not None:
                # Include image shape in the output message
                height, width, channels = img.shape
                
                # Create embedding using explicit parameters
                embedding = dino_embeddings.create_image_embedding(img)
                # print(len(embedding))
                # Create an EmbeddingResult with the numpy array
                result = HSEmbeddingResult(
                    filename=filepath,
                    embedding=embedding,
                    mask=None,
                    params=params.to_json(),
                    uuid=str(uuid.uuid4())
                )
                queue_out.put(result)
            else:
                # Send the actual exception for failed images
                queue_out.put(ValueError(f"Failed to load image: {filepath}"))
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


def process_batch(image_paths):
    """
    Process a single batch of images using multiprocessing.
    
    Args:
        image_paths: List of image paths for this batch
        
    Returns:
        list: List of processing results
    """
    # Set up multiprocessing resources
    num_processes = max(1, mp.cpu_count() - 3)
    queue_in = mp.Queue(maxsize=5)
    queue_out = mp.Queue(maxsize=5)
    
    # Start worker processes
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=worker, args=(queue_in, queue_out))
        p.daemon = True
        p.start()
        processes.append(p)
    
    # Start producer in separate process
    producer_process = mp.Process(
        target=producer, 
        args=(image_paths, queue_in, num_processes)
    )
    producer_process.start()
    
    # Collect results with progress tracking
    results = []
    completed_workers = 0
    start_time = time.time()
    last_update_time = start_time
    update_interval = 1.0  # Update progress every second
    total_images = len(image_paths)
    
    while completed_workers < num_processes:
        try:
            result = queue_out.get(timeout=0.1)
            if result is None:
                completed_workers += 1
            elif isinstance(result, Exception):
                # Log exceptions but continue processing
                print(f"\n process_batch Error: {type(result).__name__}: {str(result)}")
            elif isinstance(result, HSEmbeddingResult):
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
                    
                    # Report current memory usage
                    process = psutil.Process()
                    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Clear line and print updated progress
                    print(f"\rProgress: {percent_complete:.1f}% ({processed_count}/{total_images}) " 
                          f"| Rate: {rate:.1f} img/s | ETA: {eta:.1f}s | Mem: {memory_usage:.1f}MB ", end="")
                
                last_update_time = current_time
            
        except mp.queues.Empty:
            # Queue is empty but workers might still be processing
            continue
    
    # Print newline after progress updates
    print()
    
    # Clean up processes
    producer_process.join()
    for p in processes:
        p.join()
    
    # Empty any remaining items in queues (cleanup)
    while not queue_in.empty():
        try:
            queue_in.get_nowait()
        except:
            pass
            
    while not queue_out.empty():
        try:
            queue_out.get_nowait()
        except:
            pass
    
    return results


def worker_dinov2_threaded(queue_in, queue_out):
    """
    Worker thread for DINOv2 embeddings (uses threading instead of multiprocessing).
    
    Args:
        queue_in: Queue to get image paths from
        queue_out: Queue to put processed results to
    """
    while True:
        filepath = queue_in.get()
        if filepath is None:  # None is our signal to stop
            queue_out.put(None)
            break
        
        try:
            # Process the image
            _, img = load_image(filepath)
            if img is not None:
                # Create embedding using DINOv2
                embedding = dino_embeddings.create_image_embedding(img)
                
                # Create an EmbeddingResult
                result = HSEmbeddingResult(
                    filename=filepath,
                    embedding=embedding,
                    mask=False,
                    params=None,  # DINOv2 doesn't use params
                    uuid=str(uuid.uuid4())
                )
                queue_out.put(result)
            else:
                queue_out.put(ValueError(f"Failed to load image: {filepath}"))
        except Exception as e:
            queue_out.put(e)


def producer_threaded(image_paths, queue_in, num_threads):
    """
    Producer function for threading (feeds image paths into the input queue).
    
    Args:
        image_paths: List of image file paths
        queue_in: Queue to put paths into
        num_threads: Number of worker threads (for termination signals)
    """
    for path in image_paths:
        queue_in.put(path)
    
    # Send termination signal to each worker
    for _ in range(num_threads):
        queue_in.put(None)


def process_batch_threaded(image_paths, num_threads=None):
    """
    Process a batch of images using threading (better for CPU-bound DINOv2).
    
    Args:
        image_paths: List of image paths for this batch
        num_threads: Number of threads to use (default: CPU count - 2)
        
    Returns:
        list: List of processing results
    """
    # Set up threading resources
    if num_threads is None:
        num_threads = max(1, mp.cpu_count() - 2)
    
    queue_in = queue.Queue(maxsize=10)
    queue_out = queue.Queue(maxsize=10)
    
    # Start worker threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker_dinov2_threaded, args=(queue_in, queue_out))
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Start producer in separate thread
    producer_thread = threading.Thread(
        target=producer_threaded, 
        args=(image_paths, queue_in, num_threads)
    )
    producer_thread.start()
    
    # Collect results with progress tracking
    results = []
    completed_workers = 0
    start_time = time.time()
    last_update_time = start_time
    update_interval = 1.0  # Update progress every second
    total_images = len(image_paths)
    
    while completed_workers < num_threads:
        try:
            result = queue_out.get(timeout=0.1)
            
            if result is None:
                completed_workers += 1
            elif isinstance(result, Exception):
                # Log exceptions but continue processing
                print(f"\nError: {type(result).__name__}: {str(result)}")
            elif isinstance(result, HSEmbeddingResult):
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
                    
                    # Report current memory usage
                    process = psutil.Process()
                    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Clear line and print updated progress
                    print(f"\rProgress: {percent_complete:.1f}% ({processed_count}/{total_images}) " 
                          f"| Rate: {rate:.1f} img/s | ETA: {eta:.1f}s | Mem: {memory_usage:.1f}MB ", end="")
                
                last_update_time = current_time
            
        except queue.Empty:
            # Queue is empty but workers might still be processing
            continue
    
    # Print newline after progress updates
    print()
    
    # Clean up threads
    producer_thread.join()
    for t in threads:
        t.join()
    
    return results


def process_in_batches(all_image_paths, batch_size=50000, output_path="embeddings"):
    """
    Process images in batches to prevent memory issues.
    
    Args:
        all_image_paths: List of all image paths to process
        batch_size: Number of images to process in each batch
        output_path: Directory to save batch results
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    total_images = len(all_image_paths)
    batch_results_meta = []
    
    # Calculate number of batches
    num_batches = (total_images + batch_size - 1) // batch_size
    
    overall_start_time = time.time()
    
    for batch_num in range(num_batches):
        batch_start_time = time.time()
        
        # Calculate start and end indices for this batch
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        current_batch_size = end_idx - start_idx
        
        print(f"\n=== Processing Batch {batch_num+1}/{num_batches} ({current_batch_size} images) ===")
        print(f"Batch range: {start_idx} to {end_idx-1}\n")
        
        # Track memory usage before batch
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Get the paths for this batch
            batch_paths = all_image_paths[start_idx:end_idx]
            
            # Process this batch
            if USE_DINO is True:
                batch_results = process_batch_threaded(batch_paths)
            else:
                batch_results = process_batch(batch_paths)
            
            # Generate batch output filename
            batch_filename = f"batch_{batch_num+1}_of_{num_batches}.pickle"
            batch_filepath = os.path.join(output_path, batch_filename)
            
            # ===== Persistence logic =====
            # This is where you would add your code to save the results
            # Example:
            # save_embeddings_to_file(batch_results, batch_filepath)
            print(f"\nSaving {len(batch_results)} embeddings to {batch_filepath}")
            # Uncomment and implement your saving logic:
            with open(batch_filepath, 'wb') as f:
                pickle.dump(batch_results, f)
            
            # Add metadata about this batch
            batch_results_meta.append({
                "batch_num": batch_num + 1,
                "filename": batch_filename,
                "images_processed": len(batch_results),
                "start_idx": start_idx,
                "end_idx": end_idx - 1
            })
        
        except Exception as e:
            print(f"\nERROR processing batch {batch_num+1}: {str(e)}")
        
        # Track memory and timing after batch
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        batch_duration = time.time() - batch_start_time
        
        # Report on batch completion
        print(f"\nBatch {batch_num+1} completed in {batch_duration:.1f} seconds")
        print(f"Memory: {mem_before:.1f}MB → {mem_after:.1f}MB (Δ{mem_after-mem_before:.1f}MB)")
        
        # Force garbage collection
        del batch_results
        gc.collect()
        
        # Report memory after garbage collection
        mem_after_gc = process.memory_info().rss / 1024 / 1024
        print(f"Memory after GC: {mem_after_gc:.1f}MB (Δ{mem_after_gc-mem_after:.1f}MB)")
        
        # If there are more batches, add a small
        #  delay to ensure resources are released
        if batch_num < num_batches - 1:
            time.sleep(1)
    
    # Report overall completion
    total_duration = time.time() - overall_start_time
    print(f"\n=== All batches completed in {total_duration:.1f} seconds ===")
    print(f"Processed {total_images} images in {num_batches} batches")
    
    # Create a summary file with metadata about all batches
    # summary_path = os.path.join(output_path, "batch_summary.json")
    # with open(summary_path, 'w') as f:
    #     json.dump(batch_results_meta, f, indent=2)
    
    return batch_results_meta


def process_in_batches_threaded(all_image_paths, batch_size=50000, output_path="embeddings", num_threads=None):
    """
    Process images in batches using threading (for DINOv2 on CPU).
    
    Args:
        all_image_paths: List of all image paths to process
        batch_size: Number of images to process in each batch
        output_path: Directory to save batch results
        num_threads: Number of threads to use (default: CPU count - 2)
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Delete all existing files in the output folder
    output_dir = Path(output_path)
    for file in output_dir.glob("*"):
        if file.is_file():
            file.unlink()
            print(f"Deleted existing file: {file}")
    
    total_images = len(all_image_paths)
    batch_results_meta = []
    
    # Calculate number of batches
    num_batches = (total_images + batch_size - 1) // batch_size
    
    overall_start_time = time.time()
    
    for batch_num in range(num_batches):
        batch_start_time = time.time()
        
        # Calculate start and end indices for this batch
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        current_batch_size = end_idx - start_idx
        
        print(f"\n=== Processing Batch {batch_num+1}/{num_batches} ({current_batch_size} images) [THREADED] ===")
        print(f"Batch range: {start_idx} to {end_idx-1}\n")
        
        # Track memory usage before batch
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Get the paths for this batch
            batch_paths = all_image_paths[start_idx:end_idx]
            
            # Process this batch using threading
            batch_results = process_batch_threaded(batch_paths, num_threads=num_threads)
            
            # Generate batch output filename
            batch_filename = f"batch_{batch_num+1}_of_{num_batches}_dinov2.pickle"
            batch_filepath = os.path.join(output_path, batch_filename)
            
            print(f"\nSaving {len(batch_results)} embeddings to {batch_filepath}")
            with open(batch_filepath, 'wb') as f:
                pickle.dump(batch_results, f)
            
            # Add metadata about this batch
            batch_results_meta.append({
                "batch_num": batch_num + 1,
                "filename": batch_filename,
                "images_processed": len(batch_results),
                "start_idx": start_idx,
                "end_idx": end_idx - 1
            })
        
        except Exception as e:
            print(f"\nERROR processing batch {batch_num+1}: {str(e)}")
        
        # Track memory and timing after batch
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        batch_duration = time.time() - batch_start_time
        
        # Report on batch completion
        print(f"\nBatch {batch_num+1} completed in {batch_duration:.1f} seconds")
        print(f"Memory: {mem_before:.1f}MB → {mem_after:.1f}MB (Δ{mem_after-mem_before:.1f}MB)")
        
        # Force garbage collection
        del batch_results
        gc.collect()
        
        # Report memory after garbage collection
        mem_after_gc = process.memory_info().rss / 1024 / 1024
        print(f"Memory after GC: {mem_after_gc:.1f}MB (Δ{mem_after_gc-mem_after:.1f}MB)")
        
        # If there are more batches, add a small delay to ensure resources are released
        if batch_num < num_batches - 1:
            time.sleep(1)
    
    # Report overall completion
    total_duration = time.time() - overall_start_time
    print(f"\n=== All batches completed in {total_duration:.1f} seconds ===")
    print(f"Processed {total_images} images in {num_batches} batches")
    
    return batch_results_meta


def main():
    """
    Main function to process images in batches
    """
    try:
        # # Get all image filepaths
        
        image_paths = get_image_filepaths_from_folders([r"D:\temp_match_imgs\ToScan"])
        # image_paths = get_image_filepaths_from_folders(
        #     [
        #         r"D:\temp_match_imgs\matchable",
        #         r"D:\temp_match_imgs\butterflys",
        #         r"D:\temp_match_imgs\Flowers",
        #         ]
        #         )
        # image_paths = get_image_filepaths_from_folders(
        #     [r"D:\phantom_menance_frames"])
        # image_paths = get_image_filepaths_localtest()
        random.shuffle(image_paths)
        # image_paths = image_paths[:10000]
        original_count = len(image_paths)
        print(f"Found {original_count} images in test_images directory")
        
        # # Simple replication to create test dataset (can be removed for real use)
        # if original_count > 0 and original_count < 500:
        #     image_paths = image_paths * (500 // original_count + 1)
        #     image_paths = image_paths[:500]  # Limit to 500k for testing
        
        total_images = len(image_paths)
        print(f"creating test dataset with {total_images} image paths")
        

        if USE_DINO is True:
                
            # Process all images in batches of 50,000
            batch_results = process_in_batches_threaded(
                image_paths,
                batch_size=5000,
                output_path="embeddings_output"
            )
        else:
            # Process all images in batches of 50,000
            batch_results = process_in_batches(
                image_paths,
                batch_size=5000,
                output_path="embeddings_output"
            )

        print(f"\nProcessed {len(batch_results)} batches successfully")
        
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()



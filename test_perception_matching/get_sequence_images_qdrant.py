#!/usr/bin/env python3
"""
Module for Qdrant vector database connection and similarity search.
"""

import os
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, ScoredPoint
from typing import Optional, Tuple, List, Dict, Any
import shutil
from pathlib import Path
import time
import random

COLLECTION_NAME = "pokemon"
def get_qdrant_client():
    """
    Get a connection to the Qdrant vector database.
    
    Returns:
        Qdrant client object
    """
    # Get Qdrant connection parameters from environment variables
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    
    # Connect to Qdrant server
    client = QdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        api_key=qdrant_api_key
    )
    
    return client


def get_random_item_with_closest_match(
    client, 
    collection_name: str,
    limit: int = 1
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get a random item from the Qdrant collection and find its closest matches.
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection containing vector data
        limit: Number of closest matches to return
        
    Returns:
        Tuple containing (random_item, closest_matches)
    """
    # Get information about the collection
    collection_info = client.get_collection(collection_name=collection_name)
    
    sampled = client.query_points(
        collection_name=collection_name,
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        with_payload=True,
        with_vectors=True
    )

    random_item = sampled.points[0].payload
    random_filename = random_item.get("filename")
    random_id = sampled.points[0].id
    vector = sampled.points[0].vector
    # Find closest matches (excluding the random item itself)

    zz = client.query_points(
        collection_name=collection_name,
        query=vector, # <--- Dense vector
    )

    matched_item: ScoredPoint = zz.points[0]

    
  
    return vector, (random_id, random_filename), (matched_item.id, matched_item.payload.get("filename")), matched_item.payload


def get_sequence_of_closest_matches(
    client, 
    
    collection_name: str, 
    vector: List[float],
    limit: int | None = None
) -> List[Dict[str, Any]]:
    """
    Get a sequence of closest matches to a vector, depleting the collection.
    
    This function repeatedly:
    1. Finds the closest point to the input vector
    2. Adds its payload to the result list
    3. Deletes the point from the collection
    4. Repeats until the collection is empty
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection containing vector data
        vector: Query vector to find similarities against
        
    Returns:
        List of payloads sorted by similarity to the input vector
    """
    results = []
    
    # Get total count of points for progress reporting
    collection_info = client.get_collection(collection_name=collection_name)
    total_points = collection_info.points_count
    print(f"Starting similarity sequence processing for {total_points} points")
    
    # Track progress and timing
    start_time = time.time()
    last_update_time = start_time
    processed_count = 0
    scores=[]
    cnt = 0
    while True:
        if limit is not None and cnt >= limit:
            break
        cnt += 1
        # Find the closest matching point
        search_result = client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=1,
            with_payload=True,
            with_vectors=True
        )
        
        # Check if we found any points
        if not search_result:
            break
        scores.append(search_result[0].score)
        # Get the closest point
        point = search_result[0]
        vector = point.vector
        # Add the payload to our results
        results.append(point.payload)
        
        # Delete the point so it won't be found in the next iteration
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=[point.id]
            ),
            wait=True
        )
        
        # Update progress tracking
        processed_count += 1
        current_time = time.time()
        
        # Print progress update every 10 seconds
        if current_time - last_update_time >= 10:
            elapsed_time = current_time - start_time
            points_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
            
            # Estimate remaining time
            remaining_points = total_points - processed_count
            estimated_time_remaining = remaining_points / points_per_second if points_per_second > 0 else float('inf')
            
            # Format time remaining nicely
            if estimated_time_remaining == float('inf'):
                time_remaining_str = "unknown"
            else:
                hours, remainder = divmod(estimated_time_remaining, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_remaining_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            print(f"Progress: {processed_count}/{total_points} points ({processed_count/total_points*100:.1f}%) | "
                  f"Speed: {points_per_second:.2f} points/sec | "
                  f"Elapsed: {elapsed_time:.1f}s | "
                  f"Est. remaining: {time_remaining_str}")
            
            # Update the last update time
            last_update_time = current_time
    
    # Final progress report
    total_time = time.time() - start_time
    print(f"Completed processing {processed_count} points in {total_time:.1f} seconds "
          f"({processed_count/total_time:.2f} points/sec)")
    
    return results, scores

def clone_collection(client, collection_name: str, new_collection_name: str, batch_size: int = 1000):
    """
    Clone a collection including all vectors and payloads to a new collection.
    
    Args:
        client: Qdrant client
        collection_name: Source collection name
        new_collection_name: Destination collection name
        batch_size: Number of points to process in each batch
    """
    # Check if destination collection already exists and delete it if it does
    try:
        client.get_collection(collection_name=new_collection_name)
        print(f"Collection '{new_collection_name}' already exists. Deleting it...")
        client.delete_collection(collection_name=new_collection_name)
        print(f"Deleted existing collection: '{new_collection_name}'")
    except Exception:
        # Collection doesn't exist, which is fine
        pass
        
    # Get source collection configuration
    source_collection = client.get_collection(collection_name=collection_name)
    vector_size = source_collection.config.params.vectors.size
    vector_distance = source_collection.config.params.vectors.distance
    
    # Create new collection with the same vector configuration
    print(f"Creating new collection '{new_collection_name}' with same configuration as '{collection_name}'")
    client.create_collection(
        collection_name=new_collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=vector_distance),
        on_disk_payload=True
    )
    
    # We'll determine total count during processing
    total_copied = 0
    start_time = time.time()
    offset = None  # Start with None for first batch
    
    print(f"Starting clone from '{collection_name}' to '{new_collection_name}'")
    
    while True:
        batch_start_time = time.time()
        
        # Scroll through records in batches
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,  # This will be None for the first request
            with_payload=True,
            with_vectors=True
        )
        
        if not points:
            break  # No more points to process
            
        # Prepare batch of points to upload
        points_to_upsert = [
            models.PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload
            ) for point in points
        ]
        
        # Upload batch to new collection
        client.upsert(
            collection_name=new_collection_name,
            points=points_to_upsert,
            wait=True  # Wait for this batch to be processed before continuing
        )
        
        total_copied += len(points)
        
        # Print progress without percentages that might be misleading
        elapsed = time.time() - start_time
        batch_time = time.time() - batch_start_time
        points_per_second = total_copied / elapsed if elapsed > 0 else 0
        
        print(f"Progress: {total_copied} points copied | " 
              f"Speed: {points_per_second:.1f} points/sec | "
              f"Batch time: {batch_time:.2f}s | "
              f"Elapsed: {elapsed:.1f}s")
        
        # Use the returned next_offset value for the next iteration
        offset = next_offset
        
        # If there is no next offset, we're done
        if offset is None:
            break
    
    # Final status
    print(f"Successfully cloned {total_copied} points from '{collection_name}' to '{new_collection_name}'")

def main():
    # Create image_sequence directory in the same location as this script
    script_dir = Path(__file__).parent.absolute()
    image_sequence_dir = script_dir / "image_sequence"
    
    # Check if directory already exists
    if image_sequence_dir.exists():
        response = input(f"Directory {image_sequence_dir} already exists. Delete and continue? (y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree(image_sequence_dir)
            print(f"Deleted existing directory: {image_sequence_dir}")
        else:
            print("Operation cancelled by user.")
            return
    
    client = get_qdrant_client()
    
    # Print Qdrant info
    print(f"Connected to Qdrant server")
    
    # Use "embeddings" as the collection name
    vector, random_item, closest_matches, payload = get_random_item_with_closest_match(
        client,
        collection_name=COLLECTION_NAME,
        limit=1
    )

    print(f"Cloning collection {COLLECTION_NAME} to {COLLECTION_NAME}_clone")
    clone_collection(client,collection_name=COLLECTION_NAME, new_collection_name=f"{COLLECTION_NAME}_clone")
    
    sequence, scores = get_sequence_of_closest_matches(
        client,
        limit=None,
        collection_name=f"{COLLECTION_NAME}_clone",
        vector=vector
    )
    
    print(f"Generated sequence of {len(sequence)} items")
    
    # Create the directory if it doesn't exist
    if not image_sequence_dir.exists():
        image_sequence_dir.mkdir(parents=True)
        print(f"Created directory: {image_sequence_dir}")
    
    # Copy files with incrementing filenames
    for i, (item, score) in enumerate(zip(sequence, scores)):
        source_path = item["filename"]
        score = str(score).replace(".", "_")
        # Create destination filename with leading zeros (7 digits)
        dest_filename = f"{i+1:007d}_s{score}.jpg"
        dest_path = image_sequence_dir / dest_filename
        
        try:
            shutil.copy2(source_path, dest_path)
            # print(f"Copied: {source_path} → {dest_path}, Score: {score}")
        except Exception as e:
            print(f"Error copying {source_path}: {e}")
    
    print(f"Successfully copied {len(sequence)} images to {image_sequence_dir}")

if __name__ == "__main__":
    main()

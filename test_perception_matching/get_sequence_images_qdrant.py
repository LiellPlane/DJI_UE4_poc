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
from qdrant_utils import wait_for_collection_ready, delete_point, get_closest_match, clone_collection, get_random_item_with_closest_match, get_qdrant_client
COLLECTION_NAME = "starwars"


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
        search_result = get_closest_match(
            client,
            collection_name=collection_name,
            vector=vector,
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
        delete_point(client=client, collection_name=collection_name, point_ids=[point.id])
        # client.delete(
        #     collection_name=collection_name,
        #     points_selector=models.PointIdsList(
        #         points=[point.id]
        #     ),
        #     wait=True
        # )
        
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


def main():
    # Create image_sequence directory in the same location as this script
    script_dir = Path(__file__).parent.absolute()
    image_sequence_dir = script_dir / "image_sequence"
    # image_sequence_dir = r"D:\MatchOutput"
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
    wait_for_collection_ready(client, f"{COLLECTION_NAME}_clone")
    sequence, scores = get_sequence_of_closest_matches(
        client,
        limit=10000,
        collection_name=f"{COLLECTION_NAME}_clone",
        vector=vector,
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

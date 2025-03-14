#!/usr/bin/env python3
"""
Module for Qdrant vector database connection and similarity search.
"""

import os
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import Optional, Tuple, List, Dict, Any
import shutil
from pathlib import Path
import time
import random


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
    closest_matches_result = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=limit+10,  # Get more results to filter from
        with_payload=True,
        query_filter=Filter(
            must_not=[
                FieldCondition(
                    key="id",
                    match=MatchValue(value=str(random_id))  # Ensure ID is string type
                )
            ]
        )
    )
    

    
    # Add double check to ensure no duplicates
    closest_matches = []
    seen_ids = {random_id}  # Track already seen IDs
    for match in closest_matches_result:
        match_filename = match.payload.get("filename")
        
        # Skip duplicate IDs instead of raising exception
        if match.id == random_id or match.id in seen_ids:
            print(f"Skipping duplicate ID: {match.id}")
            continue
            
        # Skip if this is somehow the same file (different ID but same file)
        if match_filename == random_filename:
            print(f"Skipping duplicate file: {match_filename}")
            continue
            
        seen_ids.add(match.id)  # Track this ID
        match_dict = {
            "id": match.id,
            "filename": match_filename,
            "embedding": match.vector
        }
        closest_matches.append(match_dict)
        
        # Break once we have enough matches
        if len(closest_matches) >= limit:
            break
    
    # Verify we have unique items
    if closest_matches:
        print(f"Random item: {random_filename}")
        print(f"Closest match: {closest_matches[0]['filename']}")
    else:
        print(f"Warning: No closest matches found for {random_filename}")
    
    return random_item_dict, closest_matches


def get_sequence_of_closest_matches(
    client, 
    collection_name: str = "embeddings", 
    sequence_length: int = 100
) -> List[Dict[str, Any]]:
    """
    Get a sequence of closest matches, starting from a random item.
    Each subsequent item is the closest match to the previous item,
    excluding all items already in the sequence.
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection containing vector data (default: embeddings)
        sequence_length: Number of items in the sequence (default: 100)
        
    Returns:
        List of items forming a similarity chain.
    """
    # Provide initial feedback with time estimate
    print(f"Starting sequence generation of {sequence_length} items...")
    print(f"Estimated time: {sequence_length/1000:.1f} to {sequence_length/500:.1f} minutes")
    start_time = time.time()
    
    # Get a random item to start the sequence
    random_items = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=True
    )
    
    if not random_items or not random_items[0]:
        raise ValueError(f"No data found in collection {collection_name}")
    
    # Initialize sequence with the random item
    current_item = random_items[0][0]
    
    # Convert to dictionary format
    current_item_dict = {
        "id": current_item.id,
        "filename": current_item.payload.get("filename"),
        "embedding": current_item.vector
    }
    
    sequence = [current_item_dict]
    excluded_ids = [current_item.id]
    
    # Process one item at a time to maintain the proper chain
    for i in range(sequence_length - 1):
        # Get current vector for similarity search
        current_vector = current_item_dict["embedding"]
        
        # Find the closest match to the current item
        search_results = client.search(
            collection_name=collection_name,
            query_vector=current_vector,
            limit=100,  # Get more than needed to filter through already excluded
            with_payload=True,
            with_vectors=True,
            query_filter=Filter(
                must_not=[
                    FieldCondition(
                        key="id",
                        match=MatchValue(value=str(id_val))
                    ) for id_val in excluded_ids
                ]
            )
        )
        
        if not search_results:
            # No more matches found
            break
            
        # Get the first non-excluded match
        next_item = search_results[0]
        
        # Convert to dictionary format
        next_item_dict = {
            "id": next_item.id,
            "filename": next_item.payload.get("filename"),
            "embedding": next_item.vector
        }
        
        sequence.append(next_item_dict)
        excluded_ids.append(next_item.id)
        
        # Update for next iteration
        current_item_dict = next_item_dict
        
        # Provide progress updates
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            items_per_second = (i + 1) / elapsed if elapsed > 0 else 0
            estimated_total = sequence_length / items_per_second if items_per_second > 0 else 0
            remaining = max(0, estimated_total - elapsed)
            
            print(f"Processed {i + 1}/{sequence_length} items ({(i+1)/sequence_length*100:.1f}%)")
            print(f"Speed: {items_per_second:.1f} items/sec, Est. remaining: {remaining/60:.1f} minutes")
                
    # Final timing information
    total_time = time.time() - start_time
    print(f"Sequence generation complete: {len(sequence)} items in {total_time/60:.1f} minutes")
            
    return sequence


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
    random_item, closest_matches = get_random_item_with_closest_match(
        client,
        collection_name="test_collection",
        limit=1
    )
    
    sequence = get_sequence_of_closest_matches(
        client,
        collection_name="test_collection",
        sequence_length=10
    )
    
    print(f"Generated sequence of {len(sequence)} items")
    
    # Create the directory if it doesn't exist
    if not image_sequence_dir.exists():
        image_sequence_dir.mkdir(parents=True)
        print(f"Created directory: {image_sequence_dir}")
    
    # Copy files with incrementing filenames
    for i, item in enumerate(sequence):
        source_path = item["filename"]
        # Create destination filename with leading zeros (7 digits)
        dest_filename = f"{i+1:07d}.jpg"
        dest_path = image_sequence_dir / dest_filename
        
        try:
            shutil.copy2(source_path, dest_path)
            # print(f"Copied: {source_path} → {dest_path}")
        except Exception as e:
            print(f"Error copying {source_path}: {e}")
    
    print(f"Successfully copied {len(sequence)} images to {image_sequence_dir}")

if __name__ == "__main__":
    main()

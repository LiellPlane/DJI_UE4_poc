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


def get_random_item(
    client, 
    collection_name: str
) -> ScoredPoint:
    """
    Get a random item from the Qdrant collection.
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection containing vector data
        
    Returns:
        A ScoredPoint object containing the random item
    """
    sampled = client.query_points(
        collection_name=collection_name,
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        with_payload=True,
        with_vectors=True
    )
    
    if not sampled.points:
        raise ValueError(f"No data found in collection {collection_name}")
        
    return sampled.points[0]


def get_closest_match(
    client, 
    collection_name: str,
    vector: List[float],
    limit: int = 1,
    with_payload: bool = True,
    with_vectors: bool = True
) -> ScoredPoint:
    """Get the closest match to a vector in a Qdrant collection."""
    search_result = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=limit,
        with_payload=with_payload,
        with_vectors=with_vectors
    )
    return search_result

def delete_point(client, collection_name: str, point_id: str):
    """Delete a point from a Qdrant collection."""

    client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(
            points=[point_id]
        ),
        wait=True
    )


def get_point_by_id(client, collection_name: str, point_id: str, with_payload: bool = True, with_vectors: bool = True) -> ScoredPoint:
    """Get a point from a Qdrant collection by its ID."""
    return client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=with_payload, 
        with_vectors=with_vectors
        )


async def async_get_nearest_neighbors(client, collection_name: str, vector: List[float], limit: int = 1, with_payload: bool = True, with_vectors: bool = True) -> ScoredPoint:
    """Get the nearest neighbors to a vector in a Qdrant collection."""
    res= await client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=limit,
        with_payload=with_payload, with_vectors=with_vectors).points
    return res.points


async def async_get_point_by_id(client, collection_name: str, point_id: str, with_payload: bool = True, with_vectors: bool = True) -> ScoredPoint:
    """Get a point from a Qdrant collection by its ID."""
    return await client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=with_payload, with_vectors=with_vectors)


async def async_get_random_point(client, collection_name: str, with_payload: bool = True, with_vectors: bool = True) -> ScoredPoint:
    """Get a random point from a Qdrant collection."""
    res = await client.query_points(
        collection_name=collection_name,
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        with_payload=with_payload, with_vectors=with_vectors)
    return res.points

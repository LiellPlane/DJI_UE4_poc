#!/usr/bin/env python3
"""
Module for Qdrant vector database connection and similarity search.
"""

import os
import sys
import pathlib
import time
import shutil
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import ScoredPoint, PointStruct
from sklearn.decomposition import PCA
from qdrant_client.http.models.models import ScoredPoint
from dataclasses import dataclass
import asyncio

@dataclass
class ScoredPointWithFlip:
    """
    Simple wrapper for ScoredPoint that adds information about whether
    the match was found with a flipped version of the image.
    """
    point: ScoredPoint
    is_flipped: bool = False


# Add all parent directories to Python path
current_path = pathlib.Path(__file__).parent.absolute()
while current_path != current_path.parent:
    if current_path not in sys.path:
        sys.path.append(str(current_path))
    current_path = current_path.parent



def wait_for_collection_ready(client, collection_name: str):
    """
    Wait until the collection is ready and healthy, showing the count of items.
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection to wait for
    """
    while True:
        time.sleep(5) # TODO do this first as collection will be status green with zero points and not ready
        # this is bad code so fix it if going beyond POC
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            count = client.count(collection_name=collection_name)
            if collection_info.status == "green":
                print(f"Collection '{collection_name}' is ready with {count.count} items")
                return
            else:
                print(f"Collection '{collection_name}' is not ready yet, status: {collection_info.status}, count: {count.count}")
        except Exception:
            print(f"Collection '{collection_name}' is not ready yet")
            
        

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
    


    client.create_collection(
        collection_name=new_collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=vector_distance),
            init_from=models.InitFrom(collection=collection_name),
        )

    return 
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


async def async_get_closest_match(
    client, 
    collection_name: str,
    vector: List[float],
    limit: int = 1,
    with_payload: bool = True,
    with_vectors: bool = True
) -> ScoredPoint:
    """Get the closest match to a vector in a Qdrant collection."""
    res = await client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=limit,
        with_payload=with_payload,
        with_vectors=with_vectors
    )
    return res

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


async def async_delete_point(client, collection_name: str, point_ids: str, max_retries=3):
    """Delete a point from a Qdrant collection with retry logic."""
    for retry in range(max_retries):
        try:
            await client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                ),
                wait=True
            )
            return
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error deleting points: {e}. Retrying ({retry+1}/{max_retries})...")
                await asyncio.sleep(2 * (retry + 1))  # Exponential backoff
            else:
                raise

def delete_point(client, collection_name: str, point_ids: list[str]):
    """Delete a point from a Qdrant collection."""

    client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(
            points=point_ids
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

# Various embedding aggregation methods
def mean_aggregation(embeddings: list) -> np.ndarray:
    """Aggregate embeddings using mean"""
    return np.mean(embeddings, axis=0)

def median_aggregation(embeddings: list) -> np.ndarray:
    """Aggregate embeddings using median (more robust to outliers)"""
    return np.median(embeddings, axis=0)

def normalized_mean_aggregation(embeddings: list) -> np.ndarray:
    """Aggregate embeddings using L2-normalized mean"""
    avg_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        return avg_embedding / norm
    return avg_embedding

def weighted_aggregation(embeddings: list, weights: list = None) -> np.ndarray:
    """Aggregate embeddings using weighted average"""
    if weights is None or len(weights) != len(embeddings):
        # Fall back to mean if weights are invalid
        return mean_aggregation(embeddings)
    
    # Normalize weights to sum to 1
    weights = np.array(weights) / sum(weights)
    # Apply weights and sum
    return np.sum([emb * w for emb, w in zip(embeddings, weights)], axis=0)



def pca_aggregation(embeddings: list, variance_retained=0.95) -> np.ndarray:
    """Aggregate embeddings using PCA-based method"""
    if len(embeddings) > 1:
        embeddings_array = np.array(embeddings)
        # Apply PCA to reduce noise
        pca = PCA(n_components=variance_retained)
        reduced = pca.fit_transform(embeddings_array)
        # Project back to original space and take mean
        reconstructed = pca.inverse_transform(reduced)
        return np.mean(reconstructed, axis=0)
    return embeddings[0]

async def async_get_embedding_average(
    client, 
    neighbour_ids: list[str], 
    collection_name,
    aggregation_method=normalized_mean_aggregation,
    max_retries: int = 5,  # Increased from 3 to 5
    initial_delay: float = 1.0,
    max_delay: float = 30.0,  # Increased from 10 to 30
    batch_size: int = 50,  # Reduced from 100 to 50
    **kwargs
) -> np.ndarray:
    """Get the average embedding of the neighbour ids with retry mechanism
    
    Parameters:
    -----------
    client : Qdrant client
        Client for accessing the vector database
    neighbour_ids : list[str]
        List of IDs to average
    collection_name : str
        Name of the collection
    aggregation_method : function
        Function that takes a list of embeddings and returns an aggregated embedding
    max_retries : int
        Maximum number of retry attempts
    initial_delay : float
        Initial delay between retries in seconds
    max_delay : float
        Maximum delay between retries in seconds
    batch_size : int
        Number of IDs to process in each batch
    **kwargs : dict
        Additional arguments to pass to the aggregation method
    """
    async def process_batch(batch_ids):
        """Process a batch of IDs with retry mechanism"""
        delay = initial_delay
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Retrieve points by their IDs
                results = await client.retrieve(
                    collection_name=collection_name,
                    ids=[id for id in batch_ids if isinstance(id, str)],
                    with_vectors=True
                )
                return results
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    print(f"Error processing batch (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay} seconds...")
                    # Exponential backoff with jitter
                    delay = min(delay * 2, max_delay)
                    jitter = random.uniform(0, delay * 0.1)  # Add 10% jitter
                    await asyncio.sleep(delay + jitter)
                else:
                    print(f"Failed to process batch after {max_retries} attempts. Last error: {e}")
                    raise last_error
    
    # Process IDs in batches
    all_results = []
    for i in range(0, len(neighbour_ids), batch_size):
        batch = neighbour_ids[i:i + batch_size]
        try:
            batch_results = await process_batch(batch)
            all_results.extend(batch_results)
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Continue with next batch even if current batch fails
            continue
    
    # Extract vectors from the retrieved points
    # add the seed embeddings to the list if they exist
    embeddings = [point.vector for point in all_results] + [id for id in neighbour_ids if isinstance(id, np.ndarray)]
    
    # Return the aggregated embeddings if any exist, otherwise return None
    if embeddings:
        return aggregation_method(embeddings, **kwargs)
    return None

def get_embedding_average(
    client, 
    neighbour_ids: list[str], 
    collection_name,
    aggregation_method=mean_aggregation,
    **kwargs
) -> np.ndarray:
    """Get the average embedding of the neighbour ids
    probably should be in another utils but whatever this will do for now
    
    Parameters:
    -----------
    client : Qdrant client
        Client for accessing the vector database
    neighbour_ids : list[str]
        List of IDs to average
    collection_name : str
        Name of the collection
    aggregation_method : function
        Function that takes a list of embeddings and returns an aggregated embedding
    **kwargs : dict
        Additional arguments to pass to the aggregation method
    """
    # Retrieve points by their IDs
    results = client.retrieve(
        collection_name=collection_name,
        ids=neighbour_ids,
        with_vectors=True
    )
    
    # Extract vectors from the retrieved points
    embeddings = [point.vector for point in results]
    
    # Return the aggregated embeddings if any exist, otherwise return None
    if embeddings:
        return aggregation_method(embeddings, **kwargs)
    return None
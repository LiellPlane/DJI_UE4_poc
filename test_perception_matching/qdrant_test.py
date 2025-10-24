from qdrant_client.models import Distance, VectorParams
# //pip install qdrant-client
from get_batch_embeddings import HSEmbeddingResult
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import pickle
import json
import sys
from pathlib import Path


def setup_collection(client, collection_name, vector_size=1024):
    """Set up a Qdrant collection, deleting it first if it already exists."""
    collections = client.get_collections().collections
    if any(collection.name == collection_name for collection in collections):
        print(f"Collection '{collection_name}' already exists. Deleting it...")
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.DOT),
    )
    print(f"Created collection '{collection_name}'")


def insert_test_data(client, collection_name):
    """Insert test data points into the collection."""
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[
            PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
            PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
            PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
            PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
            PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
            PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
        ],
    )
    return operation_info


def load_embeddings_from_pickle(client, collection_name, embeddings_dir=None, max_retries=3, batch_size=10):
    """Load embeddings from pickle files and insert them into Qdrant."""
    # Get the directory one level up from the current file's location if not provided
    if embeddings_dir is None:
        current_file_dir = Path(__file__).parent
        repo_root = current_file_dir.parent
        embeddings_dir = repo_root / "embeddings_output"
    
    if not embeddings_dir.exists():
        print(f"Error: Directory {embeddings_dir} does not exist")
        return False
    
    # Find all pickle files
    pickle_files = list(embeddings_dir.glob("**/*.pickle"))
    
    if not pickle_files:
        print(f"No pickle files found in {embeddings_dir}")
        return False
    
    print(f"Found {len(pickle_files)} pickle files to process")
    
    # Initialize counters
    processed_embeddings = 0
    inserted_embeddings = 0
    skipped_files = 0
    failed_files = 0
    processed_files = 0
    
    # Process each pickle file
    for pickle_path in pickle_files:
        try:
            # Convert Path to string for stability
            pickle_file = str(pickle_path)
            print(f"Processing file: {pickle_file}")
            
            # Load the pickle file
            with open(pickle_file, 'rb') as f:
                embedding_results = pickle.load(f)
            
            # Debug: Print the type and structure of the loaded data
            print(f"Loaded data type: {type(embedding_results)}")
            if isinstance(embedding_results, list):
                print(f"List length: {len(embedding_results)}")
                if embedding_results:
                    print(f"First item type: {type(embedding_results[0])}")
            
            # Handle the list of HSEmbeddingResult objects
            points_to_insert = []
            file_processed_count = 0
            file_inserted_count = 0
            
            if isinstance(embedding_results, list):
                total_in_file = len(embedding_results)
                for idx, result in enumerate(embedding_results):
                    try:
                        # Extract the fields - with error checking
                        if not hasattr(result, 'filename') or not hasattr(result, 'embedding') or not hasattr(result, 'uuid'):
                            print(f"Skipping result: Missing required attributes")
                            continue
                            
                        image_filename = result.filename
                        embedding_vector = result.embedding.tolist() if hasattr(result.embedding, 'tolist') else result.embedding
                        embedding_uuid = result.uuid
                        
                        # Create payload
                        payload = {
                            "filename": image_filename,
                        }
                        
                        # Add params if available
                        if hasattr(result, 'params'):
                            payload["params"] = result.params
                        
                        # Add to batch
                        points_to_insert.append(
                            PointStruct(
                                id=embedding_uuid,  # Use UUID as the point ID
                                vector=embedding_vector,
                                payload=payload
                            )
                        )
                        
                        file_processed_count += 1
                        processed_embeddings += 1
                        
                        # Process in smaller batches
                        if len(points_to_insert) >= batch_size:
                            success = insert_batch(client, collection_name, points_to_insert, max_retries)
                            if success:
                                file_inserted_count += len(points_to_insert)
                                inserted_embeddings += len(points_to_insert)
                            points_to_insert = []  # Reset batch after insertion
                            
                            # Show progress within file
                            print(f"Progress: {idx+1}/{total_in_file} embeddings processed from current file ({(idx+1)/total_in_file*100:.1f}%)")
                            
                    except Exception as e:
                        print(f"Error processing embedding result: {e}")
            else:
                print(f"Skipping {pickle_file}: Expected a list of HSEmbeddingResult objects")
                skipped_files += 1
                continue
                
            # Insert any remaining points
            if points_to_insert:
                print(f"Inserting final batch of {len(points_to_insert)} points from file {pickle_file}")
                success = insert_batch(client, collection_name, points_to_insert, max_retries)
                if success:
                    file_inserted_count += len(points_to_insert)
                    inserted_embeddings += len(points_to_insert)
            
            print(f"File summary: Processed {file_processed_count}/{total_in_file} embeddings, inserted {file_inserted_count}")
                
        except Exception as e:
            print(f"Error processing {pickle_path}: {e}")
            failed_files += 1
        
        processed_files += 1
        print(f"Overall progress: {processed_files}/{len(pickle_files)} files ({processed_files/len(pickle_files)*100:.1f}%)")
        print(f"Total embeddings inserted so far: {inserted_embeddings}")
    
    print("\n=== FINAL SUMMARY ===")
    print(f"Processed {processed_embeddings} embeddings from {len(pickle_files)} files")
    print(f"Successfully inserted {inserted_embeddings} embeddings into Qdrant")
    print(f"Skipped {skipped_files} files (not in expected format)")
    print(f"Failed to process {failed_files} files")
    
    return inserted_embeddings > 0


def insert_batch(client, collection_name, points, max_retries=3):
    """Insert a batch of points with retry logic."""
    batch_size = len(points)
    print(f"Inserting batch of {batch_size} points...")
    
    for retry in range(max_retries):
        try:
            start_time = __import__('time').time()
            operation_info = client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            end_time = __import__('time').time()
            duration = end_time - start_time
            
            print(f"✓ Successfully inserted {batch_size} points in {duration:.2f} seconds ({batch_size/duration:.1f} points/sec)")
            return True
        except Exception as e:
            if retry < max_retries - 1:
                print(f"✗ Error inserting points: {e}. Retrying ({retry+1}/{max_retries})...")
                import time
                time.sleep(2 * (retry + 1))
            else:
                print(f"✗ Failed to insert points after {max_retries} attempts: {e}")
                return False


def main():
    """Main function to run the Qdrant test script."""
    # Connect to Qdrant server
    client = QdrantClient(url="http://localhost:6333", timeout=120)  # Increased client timeout
    collection_name = "test_collection"
    
    # Set up collection with the correct vector size (2160 instead of 1024)
    setup_collection(client, collection_name, vector_size=2160)
    
    # Choose which data to load
    use_test_data = False  # Set to True to use test data, False to load from pickle files
    
    if use_test_data:
        # Insert test data
        operation_info = insert_test_data(client, collection_name)
        print(f"Test data insertion complete: {operation_info}")
    else:
        # Load embeddings from pickle files with smaller batch size and retry logic
        success = load_embeddings_from_pickle(
            client, 
            collection_name,
            batch_size=50,  # Further reduced batch size to 5
            max_retries=3   # Allow retries
        )
        if success:
            print("Successfully loaded embeddings from pickle files into Qdrant")
        else:
            print("Failed to load embeddings from pickle files")


if __name__ == "__main__":
    main()
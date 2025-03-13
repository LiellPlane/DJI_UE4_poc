#!/usr/bin/env python3
"""
Script to load embeddings from pickle files in the @embedding_output folder
into a PostgreSQL database.
"""

import os
import sys
import pickle
import uuid
import json
import glob
from pathlib import Path
import psycopg2
import numpy as np
from psycopg2.extras import execute_values
# Import the HSEmbeddingResult class
from get_batch_embeddings import HSEmbeddingResult

def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "liell_p")
    db_user = os.getenv("DB_USER", "liell_p")
    db_password = os.getenv("DB_PASSWORD", "")
    
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def get_row_count(conn):
    """Get the number of rows in the embeddings table."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    except Exception as e:
        print(f"Error counting rows in embeddings table: {e}")
        return None

def load_pickle_files():
    """Find all pickle files in the embeddings_output folder and load them into the database."""
    # Get the directory one level up from the current file's location
    current_file_dir = Path(__file__).parent
    repo_root = current_file_dir.parent
    embedding_dir = repo_root / "embeddings_output"
    
    if not embedding_dir.exists():
        print(f"Error: Directory {embedding_dir} does not exist")
        sys.exit(1)
    
    # Find all pickle files
    pickle_files = list(embedding_dir.glob("**/*.pickle"))
    
    if not pickle_files:
        print(f"No pickle files found in {embedding_dir}")
        return
    
    print(f"Found {len(pickle_files)} pickle files to process")
    
    # Connect to the database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get initial row count
        initial_count = get_row_count(conn)
        print(f"Initial row count in embeddings table: {initial_count}")
        
        # Initialize counters
        processed_embeddings = 0
        inserted_embeddings = 0
        skipped_files = 0
        failed_files = 0  # Make sure this is initialized
        processed_files = 0  # Add this line to initialize the counter
        batch_size = 100  # Process files in batches of this size
        
        for i in range(0, len(pickle_files), batch_size):
            batch_files = pickle_files[i:i+batch_size]
            data_to_insert = []
            
            for pickle_path in batch_files:
                try:
                    # Convert PosixPath to string for stability
                    pickle_file = str(pickle_path)
                    
                    # Simple file opening with string path
                    with open(pickle_file, 'rb') as f:
                        embedding_results = pickle.load(f)
                    
                    # Handle the list of HSEmbeddingResult objects
                    if isinstance(embedding_results, list):
                        for result in embedding_results:
                            # Extract the fields - will break if any field is missing
                            image_filename = result.filename
                            embedding_vector = result.embedding.tolist()
                            
                            # Create params JSON
                            # params = {
                            #     'mask': result.mask
                            # }
                            
                            # Add params from the result
                            # params.update(result.params)
                            
                            # Use the UUID from the result
                            embedding_uuid = result.uuid
                            
                            # Add to batch
                            data_to_insert.append((
                                embedding_uuid,
                                image_filename,
                                embedding_vector,
                                json.dumps(result.params, default=str)
                            ))
                            
                            processed_embeddings += 1
                    else:
                        print(f"Skipping {pickle_file}: Expected a list of HSEmbeddingResult objects")
                        continue
                        
                except Exception as e:
                    print(f"Error processing {pickle_path}: {e}")
                    failed_files += 1
            
            # Check which filenames already exist in the database
            if data_to_insert:
                image_filenames = [item[1] for item in data_to_insert]
                placeholders = ','.join(['%s'] * len(image_filenames))
                cursor.execute(f"SELECT filename FROM embeddings WHERE filename IN ({placeholders})", 
                              image_filenames)
                existing_filenames = {row[0] for row in cursor.fetchall()}
                
                # Filter out records that already exist in the database
                filtered_data = [item for item in data_to_insert if item[1] not in existing_filenames]
                skipped_files += len(data_to_insert) - len(filtered_data)
                data_to_insert = filtered_data
            
            # Insert the data in batches
            if data_to_insert:
                execute_values(
                    cursor,
                    """
                    INSERT INTO embeddings (uuid, filename, embedding, params)
                    VALUES %s
                    """,
                    data_to_insert
                )
                conn.commit()
                inserted_embeddings += len(data_to_insert)
                print(f"Inserted {len(data_to_insert)} embeddings from batch")
            
            processed_files += len(batch_files)  # Update to count actual files processed, not iterations
            if processed_files % 100 == 0:
                print(f"Processed {processed_files} files...")
        
        print(f"Skipped {skipped_files} files that were already in the database")
        
        # Get final row count for sanity check
        final_count = get_row_count(conn)
        print(f"Final row count in embeddings table: {final_count}")
        if initial_count is not None and final_count is not None:
            print(f"Added {final_count - initial_count} new rows to the database")
            print(f"Processed {processed_embeddings} total embeddings from {processed_files} files")
            
        print(f"Processed {processed_embeddings} embeddings from {len(pickle_files)} files")
        print(f"Inserted {inserted_embeddings} embeddings into the database")
        print(f"Skipped {skipped_files} embeddings (already in database)")
        print(f"Failed to process {failed_files} files")
        
    except Exception as e:
        conn.rollback()
        print(f"Error inserting embeddings into database: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    load_pickle_files()

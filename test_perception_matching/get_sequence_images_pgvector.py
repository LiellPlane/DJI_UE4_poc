#!/usr/bin/env python3
"""
Module for PostgreSQL database connection and vector similarity search.
"""

import os
import psycopg  # pip install psycopg2-binary --force-reinstall --no-cache-dir
from typing import Optional, Tuple, List, Dict, Any
import shutil
from pathlib import Path


def get_postgres_connection(database_name="postgres"):
    """
    Get a connection to the PostgreSQL database.
    
    Args:
        database_name: Name of the database to connect to
        
    Returns:
        PostgreSQL connection object
    """
    # Get database connection parameters from environment variables
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_user = os.getenv("DB_USER", "liell_p")
    db_password = os.getenv("DB_PASSWORD", "")
    
    # Connect to PostgreSQL server
    conn = psycopg.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        dbname=database_name
    )
    
    return conn


def get_random_item_with_closest_match(
    conn, 
    table_name: str, 
    vector_column: str, 
    id_column: str = "id", 
    limit: int = 1
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get a random item from the database and find its closest matches.
    
    Args:
        conn: PostgreSQL connection
        table_name: Name of the table containing vector data
        vector_column: Name of the column containing vector embeddings
        id_column: Name of the ID column
        limit: Number of closest matches to return
        
    Returns:
        Tuple containing (random_item, closest_matches)
    """
    table_name = "embeddings"
    with conn.cursor() as cursor:
        # Get a random item
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1")
        columns = [desc[0] for desc in cursor.description]
        random_item_data = cursor.fetchone()
        
        if not random_item_data:
            raise ValueError(f"No data found in table {table_name}")
            
        random_item = dict(zip(columns, random_item_data))
        
        # Get the vector from the random item
        random_vector = random_item[vector_column]
        random_id = random_item[id_column]
        
        # Find closest matches (excluding the random item itself)
        cursor.execute(
            f"""
            SELECT * FROM {table_name}
            WHERE {id_column} != %s
            ORDER BY {vector_column} <-> %s
            LIMIT %s
            """,
            (random_id, random_vector, limit)
        )
        
        closest_matches = []
        for match_data in cursor.fetchall():
            match_item = dict(zip(columns, match_data))
            closest_matches.append(match_item)
            
    return random_item, closest_matches

def get_sequence_of_closest_matches(
    conn, 
    table_name: str = "embeddings", 
    vector_column: str = "embedding", 
    id_column: str = "id", 
    sequence_length: int = 100
) -> List[Dict[str, Any]]:
    """
    Get a sequence of closest matches, starting from a random item.
    Each subsequent item is the closest match to the previous item,
    excluding all items already in the sequence.
    
    Optimized for very large sequences (hundreds of thousands of items).
    
    Args:
        conn: PostgreSQL connection
        table_name: Name of the table containing vector data (default: embeddings)
        vector_column: Name of the column containing vector embeddings (default: embedding)
        id_column: Name of the ID column (default: id)
        sequence_length: Number of items in the sequence (default: 100)
        
    Returns:
        List of items forming a similarity chain.
    """
    # Create a temporary table to track excluded IDs
    temp_table_name = f"temp_excluded_ids_{os.getpid()}"
    
    # Provide initial feedback with time estimate
    print(f"Starting sequence generation of {sequence_length} items...")
    print(f"Estimated time: {sequence_length/1000:.1f} to {sequence_length/500:.1f} minutes")
    start_time = __import__('time').time()
    
    with conn.cursor() as cursor:
        # Create temporary table with an index for fast lookups
        cursor.execute(f"""
            CREATE TEMPORARY TABLE IF NOT EXISTS {temp_table_name} (
                id_value VARCHAR PRIMARY KEY
            )
        """)
        
        # TRUNCATE TABLE quickly removes all rows from a table without scanning it
        # This is faster than DELETE and resets the table to empty state
        # We do this in case the temporary table already existed from a previous run
        cursor.execute(f"TRUNCATE TABLE {temp_table_name}")
        
        # Get a random item to start the sequence
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1")
        columns = [desc[0] for desc in cursor.description]
        random_item_data = cursor.fetchone()
        
        if not random_item_data:
            raise ValueError(f"No data found in table {table_name}")
        
        # Initialize sequence with the random item
        current_item = dict(zip(columns, random_item_data))
        sequence = [current_item]
        
        # Add the first ID to excluded table
        cursor.execute(f"INSERT INTO {temp_table_name} VALUES (%s)", 
                      (str(current_item[id_column]),))
        
        # Commit to ensure the temporary table is properly initialized
        conn.commit()
        
        # Process one item at a time to maintain the proper chain
        for i in range(sequence_length - 1):
            # Get current vector for similarity search
            current_vector = current_item[vector_column]
            
            # Find the single closest match to the current item
            cursor.execute(f"""
                SELECT * FROM {table_name} t
                WHERE NOT EXISTS (
                    SELECT 1 FROM {temp_table_name} e 
                    WHERE t.{id_column}::text = e.id_value
                )
                ORDER BY {vector_column} <-> %s
                LIMIT 1
            """, (current_vector,))
            
            next_item_data = cursor.fetchone()
            if not next_item_data:
                # No more matches found
                break
                
            # Process the next item
            next_item = dict(zip(columns, next_item_data))
            sequence.append(next_item)
            
            # Add next ID to excluded table
            cursor.execute(f"INSERT INTO {temp_table_name} VALUES (%s)", 
                          (str(next_item[id_column]),))
            
            # Update for next iteration
            current_item = next_item
            
            # Commit periodically to avoid transaction bloat and provide progress updates
            if (i + 1) % 100 == 0:
                conn.commit()
                elapsed = __import__('time').time() - start_time
                items_per_second = (i + 1) / elapsed if elapsed > 0 else 0
                estimated_total = sequence_length / items_per_second if items_per_second > 0 else 0
                remaining = max(0, estimated_total - elapsed)
                
                print(f"Processed {i + 1}/{sequence_length} items ({(i+1)/sequence_length*100:.1f}%)")
                print(f"Speed: {items_per_second:.1f} items/sec, Est. remaining: {remaining/60:.1f} minutes")
                
        # Clean up
        cursor.execute(f"DROP TABLE IF EXISTS {temp_table_name}")
        conn.commit()
        
        # Final timing information
        total_time = __import__('time').time() - start_time
        print(f"Sequence generation complete: {len(sequence)} items in {total_time/60:.1f} minutes")
            
    return sequence

def get_random_item(
    conn, 
    table_name: str, 
    id_column: str = "id"
) -> Dict[str, Any]:
    """
    Get a random item from the database.
    
    Args:
        conn: PostgreSQL connection
        table_name: Name of the table containing data
        id_column: Name of the ID column
        
    Returns:
        A dictionary containing a random item
    """
    with conn.cursor() as cursor:
        # Get a random item
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1")
        columns = [desc[0] for desc in cursor.description]
        random_item_data = cursor.fetchone()
        
        if not random_item_data:
            raise ValueError(f"No data found in table {table_name}")
            
        random_item = dict(zip(columns, random_item_data))
        
    return random_item

def main():
    # Create image_sequence directory in the same location as this script
    script_dir = Path(__file__).parent.absolute()
    image_sequence_dir = script_dir / "image_sequence"
    
    # Check if directory already exists - moved to the start of the function
    if image_sequence_dir.exists():
        response = input(f"Directory {image_sequence_dir} already exists. Delete and continue? (y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree(image_sequence_dir)
            print(f"Deleted existing directory: {image_sequence_dir}")
        else:
            print("Operation cancelled by user.")
            return
    
    conn = get_postgres_connection("liell_p")
    cursor = conn.cursor()
    cursor.execute("SELECT version()")
    db_version = cursor.fetchone()
    print(f"Connected to PostgreSQL: {db_version[0]}")
    
    # Use "embeddings" as the table name instead of "your_embeddings_table"
    random_item, closest_matches = get_random_item_with_closest_match(
        conn,
        table_name="embeddings",  # Changed from "your_embeddings_table"
        vector_column="embedding",
        limit=1
    )
    
    sequence = get_sequence_of_closest_matches(
        conn,
        table_name="embeddings",  # Changed from "your_embeddings_table"
        vector_column="embedding",
        sequence_length=100
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

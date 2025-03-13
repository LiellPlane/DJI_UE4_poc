#!/usr/bin/env python3
"""
Module for PostgreSQL database connection and vector similarity search.
"""

import os
import psycopg  # pip install psycopg2-binary --force-reinstall --no-cache-dir
from typing import Optional, Tuple, List, Dict, Any


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

def main():
    conn = get_postgres_connection("liell_p")
    cursor = conn.cursor()
    cursor.execute("SELECT version()")
    db_version = cursor.fetchone()
    print(f"Connected to PostgreSQL: {db_version[0]}")
    random_item, closest_matches = get_random_item_with_closest_match(
        conn,
        table_name="your_embeddings_table",
        vector_column="embedding",
        limit=5  # Get 5 closest matches
    )
if __name__ == "__main__":
    main()

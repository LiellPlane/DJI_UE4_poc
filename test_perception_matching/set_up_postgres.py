#!/usr/bin/env python3
"""
Script to set up a PostgreSQL table for a vector database.
This script creates the necessary table if it doesn't already exist.
"""

import os
import sys
import psycopg # pip install psycopg2-binary --force-reinstall --no-cache-dir
# from psycopg.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Set this to True to drop and recreate the database
DROP_EXISTING_DATABASE = True

def create_vector_table():
    """Create the vector database table if it doesn't exist."""
    # Get database connection parameters from environment variables
    # Fall back to values from your Postgres.app connection
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "liell_p")
    db_user = os.getenv("DB_USER", "liell_p")
    db_password = os.getenv("DB_PASSWORD", "")
    
    # Connect to PostgreSQL server
    try:
        # First connect to default database to check if our database exists
        conn = psycopg.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            dbname="postgres",
            autocommit=True  # Set autocommit to True for database operations
        )
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
        db_exists = cursor.fetchone()
        
        # Drop database if it exists and DROP_EXISTING_DATABASE is True
        if db_exists and DROP_EXISTING_DATABASE:
            print(f"Dropping existing database '{db_name}'...")
            # Close connections to the database before dropping
            cursor.execute(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{db_name}'
                AND pid <> pg_backend_pid()
            """)
            cursor.execute(f"DROP DATABASE {db_name}")
            db_exists = False
            print(f"Database '{db_name}' dropped successfully.")
        
        # Create database if it doesn't exist
        if not db_exists:
            print(f"Creating database '{db_name}'...")
            cursor.execute(f"CREATE DATABASE {db_name}")
        
        cursor.close()
        conn.close()
        
        # Connect to the vector database
        conn = psycopg.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            dbname=db_name
        )
        cursor = conn.cursor()
        
        # Check if pgvector extension is installed
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create vector table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            uuid UUID NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            embedding halfvec(2160),
            params JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create index for faster similarity search
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS embedding_idx ON embeddings 
        USING ivfflat (embedding halfvec_cosine_ops) WITH (lists = 100)
        """)
        # halfvec_l2_ops
        conn.commit()
        print("Vector database table setup completed successfully.")
        
    except Exception as e:
        print(f"Error setting up vector database: {e}")
        sys.exit(1)
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    create_vector_table()

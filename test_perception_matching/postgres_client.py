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
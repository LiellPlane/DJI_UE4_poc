#pip install psycopg2-binary
import uuid
import psycopg2
from datetime import datetime, timedelta


def partition_table_exists(cursor, schema_name, table_name):
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
            AND table_name = %s
        );
    """
    cursor.execute(query, (schema_name, table_name))
    exists = cursor.fetchone()[0]
    return exists

# Connect to your PostgreSQL server
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="liell",
    host="localhost",
    port="5432"
)

# Create a cursor object
cur = conn.cursor()
#drop_table_query = "DROP TABLE IF EXISTS user_favorite;"
#cur.execute(drop_table_query)
# Define the SQL queries
create_table_query = '''
    CREATE TABLE IF NOT EXISTS user_favorite (
        session_id UUID,
        ip_address INET NOT NULL,
        celebrity_id INT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, timestamp)
    ) PARTITION BY RANGE (timestamp);
'''
cur.execute(create_table_query)
conn.commit()





# Calculate the start and end dates for the current week
end_date = datetime.now().date() + timedelta(days=1)
start_date = end_date - timedelta(days=7)

# Format the dates as strings in the 'YYYY-MM-DD' format
end_date_str = end_date.strftime('%Y-%m-%d')
start_date_str = start_date.strftime('%Y-%m-%d')

part_name = f'user_favorite_{start_date.year}_w{start_date.isocalendar()[1]:02d}'
# # Check if the partition for the current week already exists
# check_partition_query = f'''
#     SELECT 1 FROM information_schema.tables
#     WHERE table_name = {part_name}
# '''
print(part_name)
#cur.execute(check_partition_query)
#partition_exists = cur.fetchone()

# If the partition does not exist, create it
#if not partition_exists:
create_partition_query = f'''
    CREATE TABLE IF NOT EXISTS {part_name} PARTITION OF user_favorite
    FOR VALUES FROM ('{start_date_str}') TO ('{end_date_str}');
'''
cur.execute(create_partition_query)
conn.commit()






create_index_query = '''
    CREATE INDEX  IF NOT EXISTS idx_celebrity_id ON user_favorite (celebrity_id);
'''
# Execute the SQL queries


cur.execute(create_index_query)


#disable_partition = "ALTER TABLE user_favorite SET WITHOUT OIDS;"
#enable_partition = "ALTER TABLE user_favorite SET PARTITION BY RANGE (timestamp);"
#cur.execute(disable_partition)
#version = cur.fetchone()
#print("PostgreSQL version:", version)

your_session_id_value = str(uuid.uuid4())
your_ip_address_value = "1.2.3.4"
your_celebrity_id_value = "1"
insertion = f'''
    INSERT INTO user_favorite (session_id, ip_address, celebrity_id)
    VALUES ('{your_session_id_value}', '{your_ip_address_value}', {your_celebrity_id_value});
    '''
cur.execute(insertion)
conn.commit()


cur.execute("SELECT * FROM user_favorite")
rows = cur.fetchall()
for row in rows:
    print("__")
    print(row)


cur.close()
conn.close()
import psycopg2

# Database connection details
DB_HOST = "localhost"       # Change if using a remote DB
DB_NAME = "postgres"      # Update with your DB name
DB_USER = "postgres"          # Your PostgreSQL username
DB_PASS = "DBMS123"      # Your PostgreSQL password
DB_PORT = "5432"            # Default port

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT
)
cur = conn.cursor()

# Create the predictions table (if it doesn't exist)
cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        predicted_digit INTEGER,
        true_label INTEGER
    );
""")

conn.commit()
cur.close()
conn.close()

print("Table created successfully!")
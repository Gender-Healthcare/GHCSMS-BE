import os
from dotenv import load_dotenv
from pymongo import MongoClient
import logging

# Load environment variables from .env file
load_dotenv()

# Suppress MongoDB and other warnings (optional)
logging.getLogger('pymongo').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

# Get MongoDB URI from .env
MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file")

# Function to list everything in a MongoDB database, including vector search indexes
def list_database_contents():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        print(f"Connected to MongoDB at {MONGO_URI.split('@')[1].split('/')[0]}")

        # List all database names
        databases = client.list_database_names()
        print("\nDatabases found:")
        for db_name in databases:
            if db_name in ['admin', 'local', 'config']:  # Skip system databases
                continue
            print(f"- {db_name}")

            # Select the database
            db = client[db_name]

            # List all collections in the database
            collections = db.list_collection_names()
            print(f"\n  Collections in {db_name}:")
            for coll_name in collections:
                print(f"  - {coll_name}")

                # Select the collection
                collection = db[coll_name]

                # List all documents in the collection
                # documents = collection.find()
                # doc_count = collection.count_documents({})
                # print(f"    Documents in {coll_name} (Total: {doc_count}):")
                # for i, doc in enumerate(documents, 1):
                #     print(f"    {i}. {doc}")

                # List traditional indexes in the collection
                indexes = collection.list_indexes()
                print(f"    Traditional Indexes in {coll_name}:")
                for index in indexes:
                    print(f"    - {index}")

                # List vector search and Atlas Search indexes using $listSearchIndexes
                try:
                    search_indexes = collection.aggregate([{"$listSearchIndexes": {}}])
                    print(f"    Vector Search and Atlas Search Indexes in {coll_name}:")
                    for search_index in search_indexes:
                        print(f"    - {search_index}")
                except Exception as e:
                    print(f"    Failed to list Vector Search/Atlas Search indexes in {coll_name}: {e}")

        client.close()
        print("\nDisconnected from MongoDB.")
    except Exception as e:
        print(f"Error: {e}")

# Run the script
if __name__ == "__main__":
    print("Starting MongoDB database content listing...")
    list_database_contents()
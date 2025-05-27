import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def ping_mongo():
    try:
        client = MongoClient(os.getenv('MONGO_URI'))
        return True
    except ConnectionFailure as e:
        print(e)
        return False
    finally:
        client.close()

if ping_mongo() == True:
    print("Connected to MongoDB successfully!")
else:
    print("Failed to connect to MongoDB.")
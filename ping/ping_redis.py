import redis
import os
from dotenv import load_dotenv

load_dotenv()
r = redis.Redis(host=os.getenv('REDIS_HOST'), 
                port=os.getenv('REDIS_PORT'), 
                password=os.getenv('REDIS_PASSWORD'), 
                decode_responses=True)
try:
    r.ping()
    print("Connected to Redis successfully!")
except redis.exceptions.ConnectionError as e:
    print(f"Connection error: {e}")
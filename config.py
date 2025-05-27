import openai
import redis
import requests
from pymongo import MongoClient
from discord_webhook import DiscordWebhook
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import io
import json
from dotenv import load_dotenv
import os 
load_dotenv()

# === CONFIG ===

openai.api_key = os.getenv('OPENAI_API_KEY')

# === 1. Download File from Google Drive ===
def download_from_drive(file_id):
    creds = service_account.Credentials.from_service_account_file(
        os.getenv('GOOGLE_DRIVE_CREDENTIALS_FILE'),
        scopes=['https://www.googleapis.com/auth/drive'],
    )
    service = build('drive', 'v3', credentials=creds)
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        _, done = downloader.next_chunk()
    return fh.getvalue().decode('utf-8') #download into fh

# === 2. Split Text (simulate Recursive Character Text Splitter) ===
def split_text(text, chunk_size=1000, overlap=200):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

# === 3. Generate Embeddings ===
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002'
    )
    return response['data'][0]['embedding']

# === 4. Store Embeddings in MongoDB Atlas ===
def store_to_mongo(embeddings_chunks, collection_name='Vector', index_name='cvector'):
    client = MongoClient(os.getenv('MONGO_URI'))
    db = client.get_default_database()
    collection = db[os.getenv('MONGO_COLLECTION')]
    for chunk, vector in embeddings_chunks:
        collection.insert_one({
            "text": chunk,
            index_name: vector
        })

# === 5. Retrieve Similar Document ===
def search_mongo(query_embedding, collection_name='Vector', index_name='cvector', limit=1):
    client = MongoClient(os.getenv('MONGO_URI'))
    db = client.get_default_database()
    collection = db[os.getenv('MONGO_COLLECTION')]
    pipeline = [
        {
            '$vectorSearch': {
                'queryVector': query_embedding,
                'path': index_name,
                'numCandidates': 10,
                'limit': limit
            }
        }
    ]
    results = collection.aggregate(pipeline)
    return list(results)

# === 6. Calculate Cosine Similarity ===
def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

# === 7. Redis Chat Memory ===
def store_to_redis(key, value, ttl=300):
    r = redis.Redis(host=os.getenv('REDIS_HOST'), port=os.getenv('REDIS_PORT'), password=os.getenv('REDIS_PASSWORD'))
    r.setex(key, ttl, json.dumps(value))

def load_from_redis(key):
    r = redis.Redis(host=os.getenv('REDIS_HOST'), port=os.getenv('REDIS_PORT'), password=os.getenv('REDIS_PASSWORD'))
    value = r.get(key)
    return json.loads(value) if value else None

# === 8. Use OpenAI Chat (Langchain Agent Equivalent) ===
# def summarize_text(text):
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant. Keep the answer short and suitable for Discord."},
#         {"role": "user", "content": f"{text}"}
#     ]
#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=messages,
#         max_tokens=300
#     )
#     return response['choices'][0]['message']['content']

# === 9. Send Message to Discord ===
def send_to_discord(message):
    webhook = DiscordWebhook(url=os.getenv('DISCORD_WEBHOOK_URL'), content=message)
    webhook.execute()


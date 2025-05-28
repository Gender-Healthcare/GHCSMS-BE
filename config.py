import ollama
import redis
import json
import os
from pymongo import MongoClient
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import discord
from discord.ext import commands
import io
import asyncio
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configuration
GOOGLE_CREDS = os.getenv('GOOGLE_DRIVE_CREDENTIALS_FILE')
FILE_ID = os.getenv('GOOGLE_DRIVE_FILE_ID')
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB = os.getenv('MONGO_DB', 'angler')  # Updated to match your database
MONGO_COLL = os.getenv('MONGO_COLLECTION', 'Vector')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASS = os.getenv('REDIS_PASSWORD')
DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
DISCORD_CHANNEL = int(os.getenv('DISCORD_CHANNEL_ID', 0))
MODEL = os.getenv('LANGUAGE_MODEL', 'nomic-embed-text')

# Validate environment
if not all([GOOGLE_CREDS, FILE_ID, MONGO_URI, MONGO_DB, DISCORD_TOKEN, DISCORD_CHANNEL]):
    raise ValueError("Missing required environment variables: GOOGLE_DRIVE_CREDENTIALS_FILE, GOOGLE_DRIVE_FILE_ID, MONGO_URI, MONGO_DB, DISCORD_BOT_TOKEN, or DISCORD_CHANNEL_ID")

# 1. Download from Google Drive
def download_from_drive(file_id):
    try:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_CREDS, scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)
        service.files().get(fileId=file_id).execute()
        request = service.files().export_media(fileId=file_id, mimeType='text/plain')
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return fh.getvalue().decode('utf-8')
    except Exception as e:
        print(f"Error downloading file from Google Drive: {e}")
        return None

# 2. Split text
def split_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

# 3. Generate embeddings with Ollama
def get_embedding(text):
    try:
        response = ollama.embeddings(model=MODEL, prompt=text)
        return response.get('embedding') or response.get('embeddings')
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# 4. Store to MongoDB
def store_to_mongo(embeddings_chunks, collection_name=MONGO_COLL, index_name='embedding'):  # Updated field name
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        collection.delete_many({})  # Clear existing data
        for chunk, vector in embeddings_chunks:
            if vector:
                collection.insert_one({"text": chunk, index_name: vector})
        client.close()
    except Exception as e:
        print(f"Error storing to MongoDB: {e}")

# 5. Check if vector search index exists and is correct type
def check_vector_index(collection_name=MONGO_COLL, index_name='cvector'):
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        indexes = collection.list_indexes()
        for index in indexes:
            if index['name'] == index_name:
                # Check if it's a vectorSearch index
                if index.get('type') == 'vectorSearch':
                    client.close()
                    return True
                else:
                    print(f"Index '{index_name}' exists but is type '{index.get('type')}', not 'vectorSearch'. Please recreate as a vectorSearch index.")
                    client.close()
                    return False
        client.close()
        return False
    except Exception as e:
        print(f"Error checking index: {e}")
        return False

# 6. Search MongoDB
def search_mongo(query_embedding, collection_name=MONGO_COLL, index_name='cvector', field_name='embedding', limit=1):  # Updated field name
    if not query_embedding:
        return []
    if not check_vector_index(collection_name, index_name):
        print(f"Vector search index '{index_name}' not found in collection '{collection_name}'. Please create it in MongoDB Atlas with type 'vectorSearch'.")
        return []
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        pipeline = [
            {
                '$vectorSearch': {
                    'index': index_name,
                    'queryVector': query_embedding,
                    'path': field_name,  # Use the correct field name
                    'numCandidates': 10,
                    'limit': limit
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        client.close()
        return results
    except Exception as e:
        print(f"Error searching MongoDB: {e}")
        return []

# 7. Cosine similarity
def cosine_similarity(a, b):
    try:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0

# 8. Redis storage
def store_to_redis(key, value, ttl=300):
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=True)
        r.setex(key, ttl, json.dumps(value))
        r.close()
    except Exception as e:
        print(f"Error storing to Redis: {e}")

# 9. Process document
async def process_document(file_id, query, user_id):
    text = download_from_drive(file_id)
    if not text:
        return [{"text": "Failed to download document", "similarity": 0}]
    
    chunks = split_text(text)
    embeddings_chunks = [(chunk, get_embedding(chunk)) for chunk in chunks]
    store_to_mongo(embeddings_chunks)
    
    query_embedding = get_embedding(query)
    if not query_embedding:
        return [{"text": "Failed to generate query embedding", "similarity": 0}]
    
    results = search_mongo(query_embedding)
    similarities = [
        {"text": r['text'], "similarity": cosine_similarity(query_embedding, r['embedding'])}  # Updated field name
        for r in results
    ]
    
    store_to_redis(f"chat:{user_id}", similarities)
    return similarities

# 10. Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.command(name='search')
async def search_command(ctx, *, query):
    if ctx.channel.id != DISCORD_CHANNEL:
        return
    similarities = await process_document(FILE_ID, query, str(ctx.author.id))
    response = (
        f"**Top Match (Similarity: {similarities[0]['similarity']:.2f})**\n{similarities[0]['text'][:1000]}..."
        if similarities and similarities[0]['similarity'] > 0 else "No results found. Ensure the vector search index 'cvector' is configured correctly in MongoDB Atlas as type 'vectorSearch'."
    )
    await ctx.send(response)

# Main
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
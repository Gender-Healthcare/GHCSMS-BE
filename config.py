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
import fitz  # Using PyMuPDF as primary PDF library

# Load environment
load_dotenv()

# Configuration
GOOGLE_CREDS = os.getenv('GOOGLE_DRIVE_CREDENTIALS_FILE')
FILE_ID = os.getenv('GOOGLE_DRIVE_FILE_ID')
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB = os.getenv('MONGO_DB', 'angler')
MONGO_COLL = os.getenv('MONGO_COLLECTION', 'Vector')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')  # Added default
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))  # Default to 6379
REDIS_PASS = os.getenv('REDIS_PASSWORD', '')  # Default to empty string
DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
DISCORD_CHANNEL = int(os.getenv('DISCORD_CHANNEL_ID', 0))
# Updated to use a model that produces 768-dimensional embeddings
MODEL = os.getenv('LANGUAGE_MODEL')  # This model produces 768-dim embeddings

# Validate environment variables
if not all([GOOGLE_CREDS, FILE_ID, MONGO_URI, MONGO_DB, DISCORD_TOKEN, DISCORD_CHANNEL, REDIS_HOST]):
    raise ValueError("Missing required environment variables: GOOGLE_DRIVE_CREDENTIALS_FILE, GOOGLE_DRIVE_FILE_ID, MONGO_URI, MONGO_DB, DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID, or REDIS_HOST")

# 1. Enhanced Google Drive download with file type detection
def download_from_drive(file_id):
    try:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_CREDS, scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)
        
        file_metadata = service.files().get(fileId=file_id).execute()
        file_name = file_metadata.get('name', '')
        mime_type = file_metadata.get('mimeType', '')
        
        print(f"Downloading file: {file_name} (MIME: {mime_type})")
        
        if mime_type == 'application/pdf':
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return extract_text_from_pdf(fh.getvalue())
        elif 'document' in mime_type or 'text' in mime_type:
            if 'google-apps.document' in mime_type:
                request = service.files().export_media(fileId=file_id, mimeType='text/plain')
            else:
                request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return fh.getvalue().decode('utf-8', errors='ignore')
        else:
            print(f"Unsupported file type: {mime_type}")
            return None
    except Exception as e:
        print(f"Error downloading file from Google Drive: {e}")
        return None

# 2. PDF text extraction using PyMuPDF
def extract_text_from_pdf(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
            text += "\n\n"
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF with PyMuPDF: {e}")
        return None

# 3. Enhanced text splitting with better chunk handling
def split_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    
    text = text.strip()
    text = ' '.join(text.split())
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        break_point = text.rfind('.', start, end)
        if break_point == -1:
            break_point = text.rfind(' ', start, end)
        if break_point == -1:
            break_point = end
        chunks.append(text[start:break_point + 1].strip())
        start = break_point + 1 - overlap
        if start < 0:
            start = 0
    
    return [chunk for chunk in chunks if chunk]

# 4. Generate embeddings with better error handling - Updated for 768 dimensions
def get_embedding(text):
    try:
        text = text.strip()
        if not text:
            return None
        response = ollama.embeddings(model=MODEL, prompt=text)
        embedding = response.get('embedding') or response.get('embeddings')
        if not embedding:
            raise ValueError(f"Model '{MODEL}' returned no embedding data")
        # Updated to expect 768 dimensions for the 'default' index
        if len(embedding) != 768:
            raise ValueError(f"Embedding dimension mismatch: expected 768, got {len(embedding)}")
        return embedding
    except Exception as e:
        print(f"Error generating embedding with {MODEL}: {e}")
        return None

# 5. Store to MongoDB with better error handling
def store_to_mongo(embeddings_chunks, collection_name=MONGO_COLL):
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        collection.delete_many({})  # Clear existing data
        successful_inserts = 0
        for chunk, vector in embeddings_chunks:
            if vector and chunk.strip():
                collection.insert_one({"text": chunk, "embedding": vector})
                successful_inserts += 1
        client.close()
        print(f"Successfully stored {successful_inserts} documents to MongoDB")
    except Exception as e:
        print(f"Error storing to MongoDB: {e}")
    finally:
        if 'client' in locals():
            client.close()

# 6. Check if search index exists - Updated to use 'default' index name
def check_search_index(collection_name=MONGO_COLL, index_name='default'):
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        
        # Check vector search indexes using $listSearchIndexes
        try:
            search_indexes = list(collection.aggregate([{"$listSearchIndexes": {}}]))
            for index in search_indexes:
                if index['name'] == index_name and index['type'] == 'vectorSearch':
                    client.close()
                    return True
        except Exception as e:
            print(f"Error checking vector search indexes: {e}")
        
        client.close()
        print(f"Vector search index '{index_name}' not found in collection '{collection_name}'.")
        return False
    except Exception as e:
        print(f"Error checking index: {e}")
        return False

# 7. Search MongoDB - Updated to use $vectorSearch syntax
def search_mongo(query_embedding, collection_name=MONGO_COLL, index_name='default', field_name='embedding', limit=3):
    if not query_embedding:
        return []
    if not check_search_index(collection_name, index_name):
        print(f"Vector search index '{index_name}' not found or not properly configured. Please create it in MongoDB Atlas.")
        return []
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        
        # Updated pipeline for vector search
        pipeline = [
            {
                '$vectorSearch': {
                    'index': index_name,
                    'path': field_name,
                    'queryVector': query_embedding,
                    'numCandidates': 50,
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
    finally:
        if 'client' in locals():
            client.close()

# 8. Cosine similarity
def cosine_similarity(a, b):
    try:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0

# 9. Redis storage
def store_to_redis(key, value, ttl=300):
    r = None
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=True)
        r.setex(key, ttl, json.dumps(value))
    except Exception as e:
        print(f"Error storing to Redis: {e}")
    finally:
        if r:
            r.close()

# 10. Enhanced document processing
async def process_document(file_id, query, user_id):
    print(f"Processing document for query: {query}")
    
    # Download and extract text
    text = download_from_drive(file_id)
    if not text:
        return [{"text": "Failed to download or extract text from document", "similarity": 0}]
    
    print(f"Extracted {len(text)} characters from document")
    
    # Split into chunks
    chunks = split_text(text)
    print(f"Split into {len(chunks)} chunks")
    
    # Generate embeddings
    embeddings_chunks = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding:
            embeddings_chunks.append((chunk, embedding))
        if i + 1 == len(chunks) or (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks")
    
    print(f"Generated {len(embeddings_chunks)} embeddings")
    
    # Store to MongoDB
    store_to_mongo(embeddings_chunks)
    
    # Search for similar content
    query_embedding = get_embedding(query)
    if not query_embedding:
        return [{"text": "Failed to generate query embedding", "similarity": 0}]
    
    results = search_mongo(query_embedding, limit=3)
    similarities = [
        {"text": r['text'], "similarity": cosine_similarity(query_embedding, r['embedding'])}
        for r in results if 'embedding' in r
    ]
    
    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Store results in Redis
    store_to_redis(f"chat:{user_id}", similarities)
    return similarities

# 11. Discord bot
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
    
    await ctx.send(f"Searching for: '{query}' - This may take a moment...")
    
    try:
        similarities = await process_document(FILE_ID, query, str(ctx.author.id))
    
        if similarities and similarities[0]['similarity'] > 0:
            response = f"**Top Results for: '{query}'**\n\n"
            for i, result in enumerate(similarities[:3], 1):
                similarity_score = result['similarity']
                text_preview = result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
                response += f"**{i}. Match (Similarity: {similarity_score:.3f})**\n{text_preview}\n\n"
        else:
            response = "No results found. Make sure the vector search index is configured correctly."
            
        if len(response) > 2000:
            response = response[:1997] + "..."
            
        await ctx.send(response)
        
    except Exception as e:
        await ctx.send(f"Error processing request: {str(e)}")

@bot.command(name='info')
async def info_command(ctx):
    if ctx.channel.id != DISCORD_CHANNEL:
        return
    await ctx.send("**PDF Search Bot**\nSupports: PDF files, Google Docs, Text files\nCommands: `!search <your query>`, `!info`\nUsing 'default' vector search index with 768-dimensional embeddings")

# Main
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
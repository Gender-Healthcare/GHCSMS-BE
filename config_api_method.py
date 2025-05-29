import ollama
import redis
import json
import os
import re
from pymongo import MongoClient
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import io
import asyncio
from dotenv import load_dotenv
import fitz  # Using PyMuPDF as primary PDF library
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time

# Load environment
load_dotenv()

# Configuration
GOOGLE_CREDS = os.getenv('GOOGLE_DRIVE_CREDENTIALS_FILE')
FILE_ID = os.getenv('GOOGLE_DRIVE_FILE_ID')
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB = os.getenv('MONGO_DB', 'angler')
MONGO_COLL = os.getenv('MONGO_COLLECTION', 'Vector')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASS = os.getenv('REDIS_PASSWORD', '')
MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')  # Ensures 768D embeddings
LANGUAGE_MODEL = os.getenv('LANGUAGE_MODEL', 'mistral')  # Default for chat
API_PORT = int(os.getenv('API_PORT', 5000))
DOCUMENT_READY = False

# Set Ollama host
OLLAMA_HOST = os.getenv('OLLAMA_HOST', '127.0.0.1:11434')
os.environ['OLLAMA_HOST'] = OLLAMA_HOST
print(f"Using Ollama host: {OLLAMA_HOST}")

# Validate environment variables
if not all([GOOGLE_CREDS, FILE_ID, MONGO_URI, MONGO_DB, REDIS_HOST]):
    raise ValueError("Missing required environment variables")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for your local website

# 1. Google Drive download
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
        print(f"Error downloading file: {e}")
        return None

# 1.2. PDF text extraction
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
        print(f"Error extracting PDF text: {e}")
        return None

# 1.3. Text splitting
def recursive_split(text, chunk_size=1000, overlap=0, splitters=["\n\n", "\n", ".", " ", ""]):
    def split_with_delimiter(text, delimiter):
        return text.split(delimiter) if delimiter else list(text)

    def recurse(chunks, splitter_idx):
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk.strip())
            elif splitter_idx < len(splitters):
                pieces = split_with_delimiter(chunk, splitters[splitter_idx])
                temp = ""
                for piece in pieces:
                    if len(temp) + len(piece) + len(splitters[splitter_idx]) <= chunk_size:
                        temp += piece + (splitters[splitter_idx] if splitter_idx != len(splitters) - 1 else '')
                    else:
                        final_chunks.append(temp.strip())
                        temp = piece + (splitters[splitter_idx] if splitter_idx != len(splitters) - 1 else '')
                if temp:
                    final_chunks.append(temp.strip())
                final_chunks = recurse(final_chunks, splitter_idx + 1)
            else:
                final_chunks.append(chunk.strip())
        return final_chunks

    initial_chunks = [text.strip()]
    chunks = recurse(initial_chunks, 0)

    overlapped_chunks = []
    for i in range(0, len(chunks)):
        start = max(0, i - 1)
        context = " ".join(chunks[start:i + 1])
        overlapped_chunks.append(context)

    return overlapped_chunks

# 1.4. Generate embeddings
def get_embedding(text):
    try:
        text = text.strip()
        if not text:
            return None
        print(f"Generating embedding with model: {MODEL}")
        response = ollama.embeddings(model=MODEL, prompt=text)
        embedding = response.get('embedding') or response.get('embeddings')
        if not embedding:
            raise ValueError(f"Model '{MODEL}' returned no embedding data")
        if len(embedding) != 768:
            raise ValueError(f"Embedding dimension mismatch: expected 768, got {len(embedding)}")
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# 1.5. Store to MongoDB
def store_to_mongo(embeddings_chunks, collection_name=MONGO_COLL):
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        collection.delete_many({})
        successful_inserts = 0
        for chunk, vector in embeddings_chunks:
            if vector and chunk.strip():
                collection.insert_one({"text": chunk, "embedding": vector})
                successful_inserts += 1
        client.close()
        print(f"Stored {successful_inserts} documents to MongoDB")
    except Exception as e:
        print(f"Error storing to MongoDB: {e}")
    finally:
        if 'client' in locals():
            client.close()

# 1.6. Initialize document
def initialize_document():
    global DOCUMENT_READY
    if DOCUMENT_READY:
        return

    print("Initializing document...")

    text = download_from_drive(FILE_ID)
    if not text:
        print("Failed to download/extract document.")
        return

    print(f"Extracted {len(text)} characters")

    chunks = recursive_split(text)
    embeddings_chunks = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding:
            embeddings_chunks.append((chunk, embedding))
        if i + 1 == len(chunks) or (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks")

    print(f"Generated {len(embeddings_chunks)} embeddings")
    store_to_mongo(embeddings_chunks)
    DOCUMENT_READY = True

# 6. Check search index
def check_search_index(collection_name=MONGO_COLL, index_name='default'):
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        
        search_indexes = list(collection.aggregate([{"$listSearchIndexes": {}}]))
        for index in search_indexes:
            if index['name'] == index_name and index['type'] == 'vectorSearch':
                client.close()
                return True
        
        client.close()
        print(f"Vector search index '{index_name}' not found.")
        return False
    except Exception as e:
        print(f"Error checking index: {e}")
        return False

# 7. Search MongoDB
def search_mongo(query_embedding, collection_name=MONGO_COLL, index_name='default', field_name='embedding', limit=3):
    if not query_embedding:
        return []
    if not check_search_index(collection_name, index_name):
        print(f"Vector search index '{index_name}' not configured.")
        return []
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        
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

# 10. Document processing
def process_document(file_id, query, user_id):
    print(f"Processing query: {query}")
    
    if not DOCUMENT_READY:
        return [{"text": "Document not ready.", "similarity": 0}]
    
    query_embedding = get_embedding(query)
    if not query_embedding:
        return [{"text": "Failed to generate query embedding", "similarity": 0}]
    
    results = search_mongo(query_embedding, limit=3)
    similarities = [
        {"text": r['text'], "similarity": cosine_similarity(query_embedding, r['embedding'])}
        for r in results if 'embedding' in r
    ]
    
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    high_similarities = [s for s in similarities if s['similarity'] > 0.5]
    if high_similarities:
        store_to_redis(f"chat:{user_id}", high_similarities)
    
    return high_similarities

# 10.1. Ollama chat function
def get_chat_response(query, context_chunks=None):
    try:
        print(f"Using chat model: {LANGUAGE_MODEL} on host: {OLLAMA_HOST}")
        instruction_prompt = (
            "You are a helpful assistant. "
            "Use the following context from documents to provide accurate and helpful answers. "
            "If the answer is not in the context, state 'The document does not contain this specific information' "
            "Do not answer if the context is not relevant to the user query or No relevant document chunks found. "
            "Context:\n"
        )
        if context_chunks:
            for i, chunk in enumerate(context_chunks, 1):
                instruction_prompt += f"Chunk {i} (Similarity: {chunk['similarity']:.3f}):\n{chunk['text'][:500]}...\n\n"
        else:
            instruction_prompt += "No relevant document chunks found.\n\n"
        
        instruction_prompt += f"User Query: {query}"
        
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': query},
            ],
            stream=True,
        )
        response = ""
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                response += chunk['message']['content']
            else:
                print(f"Unexpected chunk structure: {chunk}")
        if not response:
            return "No response generated by the model."
        return response.strip()
    except Exception as e:
        print(f"Ollama chat error: {type(e).__name__}: {e}")
        return "Sorry, I couldn't generate a response. Please try again."

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "document_ready": DOCUMENT_READY,
        "service": "Gender Healthcare Chatbot API"
    })

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Main chat endpoint for the website"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "error": "Query cannot be empty"
            }), 400
        
        user_id = data.get('user_id', 'anonymous')
        
        if not DOCUMENT_READY:
            return jsonify({
                "response": "The healthcare knowledge base is still loading. Please try again in a moment.",
                "document_ready": False
            })
        
        # Get relevant document chunks
        similarities = process_document(FILE_ID, query, user_id)
        
        # Generate response using Ollama
        ollama_response = get_chat_response(query, similarities[:3])
        
        return jsonify({
            "response": ollama_response,
            "document_ready": True,
            "chunks_found": len(similarities)
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            "error": "An internal error occurred. Please try again.",
            "details": str(e)
        }), 500

@app.route('/search', methods=['POST'])
def search_endpoint():
    """Search endpoint that returns raw document chunks"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "error": "Query cannot be empty"
            }), 400
        
        user_id = data.get('user_id', 'anonymous')
        
        if not DOCUMENT_READY:
            return jsonify({
                "results": [],
                "document_ready": False,
                "message": "Document not ready"
            })
        
        similarities = process_document(FILE_ID, query, user_id)
        
        results = []
        for i, result in enumerate(similarities[:3], 1):
            results.append({
                "rank": i,
                "similarity": result['similarity'],
                "text": result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
            })
        
        return jsonify({
            "results": results,
            "document_ready": True,
            "total_found": len(similarities)
        })
        
    except Exception as e:
        print(f"Error in search endpoint: {e}")
        return jsonify({
            "error": "An internal error occurred. Please try again.",
            "details": str(e)
        }), 500

@app.route('/status', methods=['GET'])
def status_endpoint():
    """Get system status"""
    return jsonify({
        "document_ready": DOCUMENT_READY,
        "ollama_host": OLLAMA_HOST,
        "embedding_model": MODEL,
        "language_model": LANGUAGE_MODEL,
        "mongo_db": MONGO_DB,
        "mongo_collection": MONGO_COLL
    })

# Initialize document in background
def init_in_background():
    """Initialize document processing in background"""
    print("Starting document initialization in background...")
    time.sleep(2)  # Small delay to let Flask start
    initialize_document()

# Main
if __name__ == "__main__":
    # Start document initialization in background thread
    init_thread = threading.Thread(target=init_in_background)
    init_thread.daemon = True
    init_thread.start()
    
    print(f"Starting Gender Healthcare Chatbot API on port {API_PORT}")
    print(f"API will be available at: http://localhost:{API_PORT}")
    print("Endpoints:")
    print("  POST /chat - Main chatbot endpoint")
    print("  POST /search - Document search endpoint") 
    print("  GET /health - Health check")
    print("  GET /status - System status")
    
    app.run(host='0.0.0.0', port=API_PORT, debug=False)
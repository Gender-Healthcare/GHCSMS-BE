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

document_text = download_from_drive(os.getenv('GOOGLE_DRIVE_FILE_ID'))
chunks = split_text(document_text)
embeddings_chunks = [(chunk, get_embedding(chunk)) for chunk in chunks]

store_to_mongo(embeddings_chunks)

query = "DISCRETE RANDOM VARIABLES"
query_embedding = get_embedding(query)
documents = search_mongo(query_embedding)

if documents:
    top_doc = documents[0]['text']
    summary = summarize_text(top_doc)
    send_to_discord(summary)
    store_to_redis("last_summary", summary)
import os
from dotenv import load_dotenv
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

load_dotenv()

# Đặt biến môi trường
os.environ['GOOGLE_DRIVE_CREDENTIALS_FILE'] = os.getenv('GOOGLE_DRIVE_CREDENTIALS_FILE')
os.environ['GOOGLE_DRIVE_FILE_ID'] = os.getenv('GOOGLE_DRIVE_FILE_ID')

def download_from_drive(file_id):
    creds = service_account.Credentials.from_service_account_file(
        os.getenv('GOOGLE_DRIVE_CREDENTIALS_FILE'),
        scopes=['https://www.googleapis.com/auth/drive'],
    )
    service = build('drive', 'v3', credentials=creds)
    request = service.files().export_media(fileId=file_id, mimeType='text/plain')
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue().decode('utf-8')

# In ra nội dung file
print(download_from_drive(os.getenv('GOOGLE_DRIVE_FILE_ID')))

# In ra môi trường
print(os.getenv('GOOGLE_DRIVE_CREDENTIALS_FILE'))
print(os.getenv('GOOGLE_DRIVE_FILE_ID'))
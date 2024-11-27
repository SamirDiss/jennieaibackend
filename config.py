# config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Azure OpenAI
    ENDPOINT = os.getenv("ENDPOINT_URL")
    SUBSCRIPTION_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    API_VERSION = "2024-08-01-preview"

    # Chat APIs
    CHAT_COMPLETION_API_URL = os.getenv("jennie_api_key_4o_mini")
    REFERENCE_COMPLETION_API_URL = os.getenv("jennie_api_url_3.5_turbo_16k")
    API_KEY = os.getenv("jennie_api_key_3.5_turbo_16k")

    # Azure Search
    SEARCH_END_POINT = os.getenv("jennie_search_endpoint")
    SEARCH_KEY = os.getenv("SEARCH_KEY")

    # Azure Storage
    STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    BLOB_SAS_TOKEN = os.getenv("blob_storage_sas_token")

    # Azure Speech
    SPEECH_KEY = os.getenv("SPEECH_KEY")
    SPEECH_REGION = os.getenv("SPEECH_REGION")

    # Deployment Models
    DEPLOYMENT_4O = os.getenv("DEPLOYMENT_4o_mini")  # Add this if not already defined


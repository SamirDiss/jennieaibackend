# utils/clients.py

from openai import AsyncAzureOpenAI
 
import azure.cognitiveservices.speech as speechsdk
from .helpers import get_speech_config
from config import Config

# Initialize Azure OpenAI client
client = AsyncAzureOpenAI(
    azure_endpoint=Config.ENDPOINT,
    api_key=Config.SUBSCRIPTION_KEY,
    api_version=Config.API_VERSION,
)

# Initialize Speech SDK client
speech_config = get_speech_config()

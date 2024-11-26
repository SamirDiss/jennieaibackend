# models/schemas.py

from pydantic import BaseModel
from typing import Literal
class TextToSpeechRequest(BaseModel):
    text: str

class ChatCompletionRequest(BaseModel):
    messages: list
    currentModel: Literal["LottieAI", "JennieAI"]
    searchLibrary: str

class BlobRequest(BaseModel):
    container_name: str
    blob_path: str

# routes/text_to_speech.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import io
from azure.cognitiveservices.speech import SpeechSynthesizer, audio, ResultReason
from models.schemas import TextToSpeechRequest
from utils.clients import speech_config
from threading import Lock
import uuid
from fastapi.responses import JSONResponse

router = APIRouter()
synthesizer_sessions = {}
synthesizer_lock = Lock()

@router.post("/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    """Convert text to speech using Azure Speech Services"""
    try:
        session_id = str(uuid.uuid4())
        audio_config = audio.AudioOutputConfig(use_default_speaker=True)
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = synthesizer.speak_text_async(request.text).get()
        
        if result.reason == ResultReason.SynthesizingAudioCompleted:
            return StreamingResponse(
                io.BytesIO(result.audio_data),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=response.wav"}
            )
        raise Exception("Speech synthesis failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

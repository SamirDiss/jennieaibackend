# routes/reference_generation.py

from fastapi import APIRouter, HTTPException, Request
from utils.clients import client
from utils.helpers import _get_reference_system_prompt
from config import Config
import requests

router = APIRouter()

@router.post("/getReference")
async def get_reference(request: Request):
    """Generate formatted reference text"""
    body = await request.json()
    prompt = [
        {
            "role": "system",
            "content": _get_reference_system_prompt()
        },
        {
            "role": "user",
            "content": body.get('reference')
        }
    ]

    try:
        response = client.chat.completions.create(
            model="Jennei-gpt-35-turbo-16k",
            messages=prompt,
            max_tokens=500,
            temperature=0.2,
            top_p=0.8,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generateTitle")
async def generate_title(request: Request):
    """Generate a title for a conversation"""
    body = await request.json()
    prompt = [
        {
            "role": "system",
            "content": 'Generate a short title for the following conversation with maximum context.'
        },
        {
            "role": "user",
            "content": body.get('messages')
        }
    ]

    headers = {
        "Content-Type": "application/json",
        "api-key": Config.API_KEY,
    }

    try:
        response = requests.post(
            Config.REFERENCE_COMPLETION_API_URL,
            json={
                "messages": prompt,
                "max_tokens": 100,
                "temperature": 0.7,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "top_p": 0.8,
                "stop": None,
            },
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from dotenv import load_dotenv

from config import Config

# Load environment variables
load_dotenv()

router = APIRouter()



# API routes should be defined before static file handling
@router.get("/api/signed-url")
async def get_signed_url():
    agent_id = Config.AGENT_ID
    xi_api_key = Config.XI_API_KEY
    
    
    if not agent_id or not xi_api_key:
        raise HTTPException(status_code=500, detail="Missing environment variables")
    
    url = f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={agent_id}"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url,
                headers={"xi-api-key": xi_api_key}
            )
            response.raise_for_status()
            data = response.json()
            return {"signedUrl": data["signed_url"]}
            
        except httpx.HTTPError:
            raise HTTPException(status_code=500, detail="Failed to get signed URL")


#API route for getting Agent ID, used for public agents
@router.get("/api/getAgentId")
def get_unsigned_url():
    agent_id = Config.AGENT_ID
    return {"agentId": agent_id}

# Mount static files for specific assets (CSS, JS, etc.)
# app.mount("/static", StaticFiles(directory="dist"), name="static")

# Serve index.html for root path
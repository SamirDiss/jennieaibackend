# routes/blob_storage.py

from fastapi import APIRouter, HTTPException
from models.schemas import BlobRequest
from utils.helpers import generate_container_sas_token

router = APIRouter()

@router.post("/download-blob")
def download_blob(request: BlobRequest):
    """Generate SAS URL for blob download"""
    try:
        sas_token = generate_container_sas_token(request.container_name, expiration_mins=5)
        blob_url_with_sas = f"{request.blob_path}?{sas_token}"
        return {"sas_url": blob_url_with_sas}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating the SAS URL: {str(e)}"
        )

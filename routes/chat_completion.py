from tenacity import retry, stop_after_attempt, wait_fixed
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from models.schemas import ChatCompletionRequest
from utils.clients import client
from utils.helpers import _get_role_information
from config import Config
import json
import asyncio
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
async def _retry_request(func, *args, **kwargs):
    print("retrying...")
    return await func(*args, **kwargs)

@router.post("/getChatCompletion")
async def get_chat_completion(request: ChatCompletionRequest):
    """Handle chat completions asynchronously with reduced retries."""
    start_time = time.time()
    print("currentAiModel", request.aiModel)
    model = request.aiModel["deploymentName"]

    if not model:
        raise ValueError("Model deployment name is not configured properly.")

    try:
        logger.info(f"Initial setup took: {time.time() - start_time:.2f} seconds")
        if request.currentModel == "LottieAI":
            response = await _retry_request(_handle_lottie_ai_completion, model, request.messages)
            
        else:
            search_start = time.time()
            # Log search configuration time
            response = await _retry_request(_handle_search_based_completion, model, request)
            logger.info(f"Total search completion took: {time.time() - search_start:.2f} seconds")
           
        return response
    except ValueError as ve:
        print("value error")
        print(ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print("exception error")
        print(e)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Helper functions specific to chat completion
async def _handle_lottie_ai_completion(model: str, messages: list):
    """Handle LottieAI-specific chat completion."""
    system_message = {
        "role": "system",
        "content": (
            "You are an AI assistant named Lottie AI developed by 6 Sided Dice. "
            'Whenever asked about your name or developer, respond with "My name is Lottie AI." '
            'or "I have been expertly crafted by the skilled team of industry leading experts '
            'in AI at 6 Sided Dice" respectively.'
        )
    }

    return await client.chat.completions.create(
        model=model,
        messages=[system_message] + messages,
        max_tokens=4096,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

async def _handle_search_based_completion(model: str, request: ChatCompletionRequest, stream: bool = False):
    pre_search = time.time()
    """Handle search-based chat completion."""
    print("search library", request.searchLibrary)
    print("model", model)
    print("messages", request.messages)
    fields_mapping = {
        "content_fields_separator": "\n",
        "content_fields": ["content"],
        "filepath_field": "filepath" if request.searchLibrary == "jennie-v1" else "file_name",
        "title_field": "title" if request.searchLibrary == "jennie-v1" else "sub_title",
        "url_field": "url" if request.searchLibrary == "jennie-v1" else "file_url",
        "vector_fields": ["contentVector"] if request.searchLibrary == "jennie-v1" else ["vector"]
    }

    data_source = {
        "type": "azure_search",
        "parameters": {
            "endpoint": Config.SEARCH_END_POINT,
            "index_name": request.searchLibrary,
            "semantic_configuration": "default",
            "query_type": "vector_semantic_hybrid",
            "fields_mapping": fields_mapping,
            "include_contexts": ["citations", "intent", "all_retrieved_documents"],
            "in_scope": True,
            "role_information": _get_role_information(),
            "filter": None,
            "strictness": 2,
            "top_n_documents": 5 if request.searchLibrary == "jennie-v1" else 20,
            "authentication": {
                "type": "api_key",
                "key": Config.SEARCH_KEY
            },
            "embedding_dependency": {
                "type": "deployment_name",
                "deployment_name": "embeddings"
            }
        }
    }

    base_params = {
        "model": model,
        "messages": request.messages,
        "max_tokens": 1000,
        "temperature": 0.3,
        "top_p": 0.6,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.0,
        "stop": None,
        "stream": stream
    }
    logger.info(f"Search configuration took: {time.time() - pre_search:.2f} seconds")
    
    # Log actual API call time
    api_start = time.time()
    response = await client.chat.completions.create(**base_params, extra_body={"data_sources": [data_source]})
    logger.info(f"API call took: {time.time() - api_start:.2f} seconds")
    
    return response
# routes/chat_completion.py

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from models.schemas import ChatCompletionRequest
from utils.clients import client
from utils.helpers import _get_role_information
from config import Config
import json

router = APIRouter()

@router.post("/getChatCompletion")
async def get_chat_completion(request: ChatCompletionRequest):
    """Handle chat completions for both LottieAI and search-based responses"""
    model = Config.DEPLOYMENT_4O

    if not model:
        raise ValueError("Model deployment name is not configured properly.")

    try:
        if request.currentModel == "LottieAI":
            return await _handle_lottie_ai_completion(model, request.messages)
        else:
            return await _handle_search_based_completion(model, request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/getChatCompletionStream")
async def get_chat_completion_stream(request: ChatCompletionRequest):
    """Handle streaming chat completions"""
    model = Config.DEPLOYMENT_4O

    if not model:
        raise ValueError("Model deployment name is not configured properly.")

    try:
        if request.currentModel == "LottieAI":
            return await _handle_lottie_ai_completion(model, request.messages)
        else:
            return await _handle_search_based_completion(model, request, stream=True)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Helper functions specific to chat completion
async def _handle_lottie_ai_completion(model: str, messages: list):
    """Handle LottieAI-specific chat completion"""
    system_message = {
        "role": "system",
        "content": (
            "You are an AI assistant named Lottie AI developed by 6 Sided Dice. "
            'Whenever asked about your name or developer, respond with "My name is Lottie AI." '
            'or "I have been expertly crafted by the skilled team of industry leading experts '
            'in AI at 6 Sided Dice" respectively.'
        )
    }

    return client.chat.completions.create(
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


async def _handle_search_based_completion(model: str, request: ChatCompletionRequest,stream: bool = False):
    """Handle search-based chat completion"""
    
    
    print("search library",request.searchLibrary)
    print("search endpoint",Config.SEARCH_END_POINT)
    print("search key",Config.SEARCH_KEY)
    
    if request.searchLibrary in ["jennie-v1"]:
        
        fields_mapping = {
            "content_fields_separator": "\n",
            "content_fields": ["content"],
            "filepath_field": "filepath",
            "title_field": "title",
            "url_field": "url",
            "vector_fields": ["contentVector"]
        }
    else:
        fields_mapping = {
            "content_fields_separator": "\n",
            "content_fields": ["content"],
            "filepath_field": "file_name",
            "title_field": "sub_title",
            "url_field": "file_url",
            "vector_fields": ["vector"]
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
            "top_n_documents": 20 if request.searchLibrary in ["ebs-staff-index", "ebs-student-index", "oracle-redwood-index","office-of-vc-index","oracle-guided-learning-index"] else 5,
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
        "stream": stream,
        
        
    }
    
    response = client.chat.completions.create(**base_params, extra_body={"data_sources": [data_source]})
    
    if stream:
        def stream_generator():
            # response = client.chat.completions.create(**base_params, extra_body={"data_sources": [data_source]})
            for chunk in response:
                # Extract the content from the chunk
                if chunk.usage:
                    
                    print("usage",response.usage)
                
                delta = chunk.choices[0].delta 
                # print("delta",delta)
                content = delta.content if delta and delta.content else ""
                if content:
                    # Yield the content as a JSON string
                    yield json.dumps({'content': content}) + '\n\n'
        print("response",response)
        return StreamingResponse(stream_generator(), media_type='application/json')
    
    else:
        return response







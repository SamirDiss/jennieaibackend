from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import text_to_speech, chat_completion, blob_storage, reference_generation

app = FastAPI()

# CORS Configuration
origins = [
    "https://lottieai.azurewebsites.net",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = origins
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# Include Routers
app.include_router(text_to_speech.router)
app.include_router(chat_completion.router)
app.include_router(blob_storage.router)
app.include_router(reference_generation.router)
# Standard library imports
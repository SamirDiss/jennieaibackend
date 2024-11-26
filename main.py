from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import text_to_speech, chat_completion, blob_storage, reference_generation

app = FastAPI()

# CORS Configuration
origins = [
    "http://localhost:4200",
    "https://lottieai.azurewebsites.net"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(text_to_speech.router)
app.include_router(chat_completion.router)
app.include_router(blob_storage.router)
app.include_router(reference_generation.router)
# Standard library imports
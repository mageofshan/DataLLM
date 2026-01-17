from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, chat

app = FastAPI(
    title="AI-Powered Data Analysis Platform",
    description="Backend API for AP Statistics & Capstone Research Platform",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(upload.router, prefix="/api/v1", tags=["Data Processing"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Data Analysis Platform API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

import os
from dotenv import load_dotenv
load_dotenv() 
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ← ADDED
from backend.route import router as ai_router

# Application Instance

app = FastAPI(
    title="AI Document Analyzer",
    description="Privacy-first, contract-enforced document processing API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Middleware (ADDED — required for browser frontends)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",          # React / Next.js dev
        "http://127.0.0.1:3000",
        
        # Add production frontend domain below
        # "https://app.yourdomain.com",
    
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# v1 router

app.include_router(
    ai_router,
    prefix="/api/v1",
    tags=["v1"],
)

# Root endpoint (safe addition)

@app.get("/", tags=["system"])
def root():
    return {"message": "AI Document Analyzer API is running"}

# Health Check 

@app.get("/health", tags=["system"])
def health_check():
    return {
        "status": "ok",
        "service": "AI Document Analyzer",
        "version": "1.0.0"
    }

# Global Validation Handler

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """
    Ensures Pydantic validation errors follow
    your structured error contract.
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": "request_validation_error",
            "details": exc.errors()
        },
    )

# Global HTTPException Passthrough

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Preserve structured HTTP errors from lower layers
    (rate limiting, validation, AI client).
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=getattr(exc, "headers", None),
    )
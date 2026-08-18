import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings, logger
from app.database import init_db
from app.services.vector_db import init_vector_db
from app.routers import resume_router, candidate_router, assessment_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for startup and shutdown tasks.
    Initializes database tables asynchronously and Qdrant vector storage on startup.
    """
    logger.info("Starting up AI-Driven Resume Assessment System API...")
    await init_db()
    init_vector_db()
    logger.info("Application startup initialization completed.")
    yield
    logger.info("Shutting down AI-Driven Resume Assessment System API...")


app = FastAPI(
    title="AI-Driven Resume Assessment System",
    description="Unified API for PDF resume parsing, Qdrant semantic vector storage, 3-Phase dynamic Q&A interview engine, LLM-as-a-Judge scoring, and candidate ranking.",
    version="1.0.0",
    lifespan=lifespan
)

# ----------------------------------------------------
# CORS Middleware
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------
# Global Exception Handlers
# ----------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Global handler for standard HTTP exceptions.
    """
    logger.warning(f"HTTPException [{exc.status_code}] on {request.method} {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "path": request.url.path
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Global handler for Request Validation errors.
    """
    logger.warning(f"RequestValidationError on {request.method} {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "message": "Input validation error in request payload.",
            "details": exc.errors(),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Global catch-all handler for unhandled internal server exceptions.
    """
    logger.error(f"Unhandled Exception on {request.method} {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "Internal server error occurred.",
            "detail": str(exc),
            "path": request.url.path
        }
    )


# ----------------------------------------------------
# Router Registration
# ----------------------------------------------------
app.include_router(resume_router)
app.include_router(candidate_router)
app.include_router(assessment_router)


# ----------------------------------------------------
# Health Check Endpoint
# ----------------------------------------------------
@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Health check endpoint to verify API and service status.
    """
    return {
        "status": "online",
        "service": "AI-Driven Resume Assessment System",
        "version": "1.0.0"
    }

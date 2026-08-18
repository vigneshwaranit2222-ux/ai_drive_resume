import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads", "resumes")


def sanitize_filename_part(text: str) -> str:
    """
    Sanitizes string into a clean lowercase filename token.
    Replaces non-alphanumeric characters with underscores.
    """
    if not text:
        return ""
    # Convert to lowercase
    cleaned = text.strip().lower()
    # Replace non-alphanumeric characters (excluding alphanumeric) with single underscore
    cleaned = re.sub(r'[^a-z0-9]+', '_', cleaned)
    # Strip leading/trailing underscores
    cleaned = cleaned.strip('_')
    return cleaned


def generate_production_filename(original_filename: str, candidate_name: Optional[str] = None) -> str:
    """
    Generates a dynamic real-time production filename.
    Format: resume_<sanitized_candidate_name>_<YYYYMMDD_HHMMSS>_<short_uuid>.pdf
    """
    sanitized_name = sanitize_filename_part(candidate_name)
    
    if not sanitized_name or sanitized_name in ["not_specified", "null", "none"]:
        # Fallback to original file stem
        stem = os.path.splitext(original_filename)[0]
        sanitized_name = sanitize_filename_part(stem) or "candidate"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    
    return f"resume_{sanitized_name}_{timestamp}_{short_uuid}.pdf"


def save_uploaded_resume(file_bytes: bytes, original_filename: str, candidate_name: Optional[str] = None) -> dict:
    """
    Saves uploaded PDF bytes into the production upload directory with a dynamic filename.
    Returns metadata dict containing saved_filename, storage_path, and original_filename.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    saved_filename = generate_production_filename(original_filename, candidate_name)
    storage_path = os.path.join(UPLOAD_DIR, saved_filename)
    
    with open(storage_path, "wb") as f:
        f.write(file_bytes)
        
    return {
        "saved_filename": saved_filename,
        "storage_path": storage_path,
        "original_filename": original_filename,
        "file_size_bytes": len(file_bytes)
    }

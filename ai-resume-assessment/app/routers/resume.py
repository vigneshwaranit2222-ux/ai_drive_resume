from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.parser import (
    extract_text_from_pdf_bytes,
    extract_text_from_pdf,
    parse_resume_with_llm,
    detect_missing_gaps,
)
from app.services.file_storage import save_uploaded_resume

router = APIRouter(prefix="/api/v1/resume", tags=["Resume Parser"])


@router.post("/parse")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Read raw uploaded file bytes
        pdf_bytes = await file.read()

        # 1. Extract text from PDF
        raw_text = extract_text_from_pdf_bytes(pdf_bytes)

        # Print debug text in terminal to verify real extraction
        print(f"--- EXTRACTED TEXT FROM ({file.filename}) ---")
        print(raw_text[:300])  # Prints first 300 characters
        print("---------------------------------------------")

        if not raw_text or len(raw_text) < 20:
            raise HTTPException(
                status_code=400,
                detail="Unable to extract readable text from PDF. File might be scanned or empty.",
            )

        # 2. Parse text dynamically via Gemini API
        parsed_profile = parse_resume_with_llm(raw_text)

        # 3. Detect gaps
        missing_gaps = detect_missing_gaps(parsed_profile)

        # 4. Save uploaded file with dynamic production filename
        file_meta = save_uploaded_resume(
            file_bytes=pdf_bytes,
            original_filename=file.filename,
            candidate_name=parsed_profile.candidate_name
        )

        profile_dict = parsed_profile.model_dump()
        profile_dict["personal_info"] = parsed_profile.personal_info

        return {
            "status": "success",
            "filename": file.filename,
            "saved_filename": file_meta["saved_filename"],
            "storage_path": file_meta["storage_path"],
            "parsed_profile": profile_dict,
            "missing_gaps": missing_gaps,
        }


    except HTTPException:
        raise
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in err_msg or "API key" in err_msg:
            raise HTTPException(
                status_code=401,
                detail="Invalid GEMINI_API_KEY in .env file. Please provide a valid Gemini API key from https://aistudio.google.com/app/apikey (starts with 'AIzaSy...')."
            )
        raise HTTPException(status_code=500, detail=f"Error parsing resume: {err_msg}")
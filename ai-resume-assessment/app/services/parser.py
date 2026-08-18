import fitz  # PyMuPDF
import json
from typing import List, Optional
from pydantic import BaseModel, Field
import google.generativeai as genai  # type: ignore # pyright: ignore
from app.config import settings

# Configure Gemini with API key
genai.configure(api_key=settings.GEMINI_API_KEY)


# 1. Pydantic Models
class Education(BaseModel):
    degree: Optional[str] = Field(default=None)
    institution: Optional[str] = Field(default=None)
    year: Optional[str] = Field(default=None)


class WorkExperience(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None


class Project(BaseModel):
    name: Optional[str] = None
    tech_stack: Optional[List[str]] = None
    summary: Optional[str] = None


class ParsedResume(BaseModel):
    candidate_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    education: List[Education] = []
    work_experience: List[WorkExperience] = []
    core_skills: List[str] = []
    projects: List[Project] = []

    @property
    def personal_info(self) -> dict:
        return {
            "name": self.candidate_name,
            "email": self.email,
            "phone": self.phone,
        }


# 2. Extract Text from PDF
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()


# Alias for backward compatibility
extract_text_from_pdf = extract_text_from_pdf_bytes


# 3. Parse Resume with Gemini API
def parse_resume_with_llm(raw_text: str) -> ParsedResume:
    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)

    # Try available models with explicit prefix (prioritizing supported gemini-2.5-flash and gemini-flash-latest)
    model_setting = settings.LLM_MODEL or "gemini-2.5-flash"
    candidates = [
        model_setting,
        f"models/{model_setting}" if not model_setting.startswith("models/") else model_setting,
        "models/gemini-2.5-flash",
        "gemini-2.5-flash",
        "models/gemini-2.5-pro",
        "gemini-2.5-pro",
        "models/gemini-flash-latest",
        "gemini-flash-latest",
        "models/gemini-3.6-flash",
        "models/gemini-3.5-flash",
    ]
    available_models = []
    for m in candidates:
        if m and m not in available_models:
            available_models.append(m)

    prompt = (
        "You are an expert HR Resume Parser.\n"
        "Extract candidate information strictly from the provided raw text below.\n"
        "DO NOT use mock data, sample names, or placeholders like 'Elena Software Architect'.\n\n"
        "Return ONLY a valid JSON object matching this structure:\n"
        "{\n"
        '  "candidate_name": "extracted name or null",\n'
        '  "email": "extracted email or null",\n'
        '  "phone": "extracted phone or null",\n'
        '  "education": [{"degree": "...", "institution": "...", "year": "..."}],\n'
        '  "work_experience": [{"company": "...", "role": "...", "duration": "...", "description": "..."}],\n'
        '  "core_skills": ["skill1", "skill2"],\n'
        '  "projects": [{"name": "...", "tech_stack": ["..."], "summary": "..."}]\n'
        "}\n\n"
        "Raw Resume Text:\n"
        "-------------------\n"
        + raw_text + "\n"
        "-------------------\n"
    )

    last_exception = None
    for model_name in available_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            json_data = json.loads(response.text)
            return ParsedResume(**json_data)
        except Exception as e:
            last_exception = e
            continue

    raise Exception(f"All Gemini models failed. Last error: {str(last_exception)}")


# 4. Gap Detection Logic
def detect_missing_gaps(parsed_data: ParsedResume) -> List[dict]:
    gaps = []

    if not parsed_data.core_skills:
        gaps.append({
            "category": "Skills",
            "issue": "Core technical skills are missing.",
            "severity": "High"
        })

    if not parsed_data.education:
        gaps.append({
            "category": "Education",
            "issue": "Education details are missing.",
            "severity": "Medium"
        })
    else:
        for edu in parsed_data.education:
            if not edu.year or str(edu.year).lower() in ["null", "none", ""]:
                gaps.append({
                    "category": "Education",
                    "issue": f"Graduation year missing for degree: {edu.degree or 'Degree'}",
                    "severity": "Low"
                })

    if not parsed_data.work_experience and not parsed_data.projects:
        gaps.append({
            "category": "Experience",
            "issue": "Neither work experience nor project details are specified.",
            "severity": "High"
        })

    return gaps
import json
import asyncio
import logging
from typing import List
from openai import AsyncOpenAI
from fastapi import HTTPException, status

from app.config import settings
from app.schemas.resume import (
    ResumeProfile, PersonalInfo, EducationItem, WorkExperienceItem, ProjectItem, MissingDataGap
)

logger = logging.getLogger("ai_resume_assessment.services.llm_engine")


async def parse_resume_with_llm(resume_text: str) -> ResumeProfile:
    """
    Parses raw resume text into a dynamic structured ResumeProfile using LLM (Gemini / OpenAI).
    
    Strict Execution:
    - NO hardcoded fallbacks or mock data ("Elena").
    - Explicitly instructs prompt to extract ONLY from provided raw_text.
    """
    has_openai_key = bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here")
    has_gemini_key = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your_gemini_api_key_here")

    if not has_openai_key and not has_gemini_key:
        logger.error("Neither OPENAI_API_KEY nor GEMINI_API_KEY is configured in .env file.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM API key is not configured in .env file. Please set GEMINI_API_KEY or OPENAI_API_KEY in .env."
        )

    # 1. Try Gemini API if GEMINI_API_KEY is present or model is gemini-*
    if has_gemini_key or (settings.LLM_MODEL and settings.LLM_MODEL.startswith("gemini")):
        try:
            return await _parse_resume_with_gemini(resume_text)
        except Exception as e:
            logger.warning(f"Gemini API parsing failed or unavailable: {str(e)}. Fallback to OpenAI if configured.")
            if not has_openai_key:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Gemini API Error during resume parsing: {str(e)}"
                )

    # 2. Try OpenAI API if OPENAI_API_KEY is present
    if has_openai_key:
        try:
            return await _parse_resume_with_openai(resume_text)
        except Exception as e:
            logger.error(f"OpenAI API Error during resume parsing: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"OpenAI API Error during resume parsing: {str(e)}"
            )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to initialize LLM provider for resume parsing."
    )


async def _parse_resume_with_gemini(resume_text: str) -> ResumeProfile:
    """
    Executes dynamic resume profile extraction using Google Gemini API.
    """
    # pyrefly: ignore [missing-import]
    import google.generativeai as genai

    api_key = settings.GEMINI_API_KEY or settings.OPENAI_API_KEY
    genai.configure(api_key=api_key)

    model_name = settings.LLM_MODEL if settings.LLM_MODEL.startswith("gemini") else "gemini-1.5-flash"
    logger.info(f"Calling Gemini API ({model_name}) for strict dynamic resume parsing...")

    model = genai.GenerativeModel(model_name)

    prompt = (
        "You are an expert AI HR assistant specializing in parsing candidate resumes.\n"
        "Extract candidate information strictly from the provided raw text below. Do NOT hallucinate or use sample names like Elena. If a field is not present, mark it as null.\n\n"
        "Return ONLY a valid JSON object strictly adhering to the following JSON schema:\n"
        "{\n"
        '  "personal_info": {"name": string or null, "email": string or null, "phone": string or null},\n'
        '  "education": [{"degree": string or null, "institution": string or null, "year": string or null}],\n'
        '  "work_experience": [{"company": string or null, "role": string or null, "duration": string or null, "description": string or null}],\n'
        '  "core_skills": [string],\n'
        '  "projects": [{"name": string or null, "tech_stack": [string], "summary": string or null}]\n'
        "}\n\n"
        f"Raw Resume Text:\n{resume_text}"
    )

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json", "temperature": 0.0}
        )
    )

    if not response or not response.text:
        raise ValueError("Gemini API returned an empty response.")

    res_json = json.loads(response.text)
    parsed_profile = ResumeProfile.model_validate(res_json)
    logger.info(f"Successfully extracted dynamic profile via Gemini for candidate: '{parsed_profile.personal_info.name}'")
    return parsed_profile


async def _parse_resume_with_openai(resume_text: str) -> ResumeProfile:
    """
    Executes dynamic resume profile extraction using OpenAI Async API with Structured Outputs.
    """
    model_name = settings.LLM_MODEL if not settings.LLM_MODEL.startswith("gemini") else "gpt-4o-mini"
    logger.info(f"Calling OpenAI Structured Outputs API ({model_name}) for dynamic resume parsing...")

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    system_prompt = (
        "You are an expert AI HR assistant specializing in parsing candidate resumes.\n"
        "Extract candidate information strictly from the provided raw text below. Do NOT hallucinate or use sample names like Elena. If a field is not present, mark it as null."
    )

    completion = await client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Raw Resume Text:\n\n{resume_text}"}
        ],
        response_format=ResumeProfile,
        temperature=0.0
    )
    
    parsed_result = completion.choices[0].message.parsed
    if not parsed_result:
        raise ValueError("OpenAI LLM returned an empty parsed object.")

    logger.info(f"Successfully extracted dynamic profile via OpenAI for candidate: '{parsed_result.personal_info.name}'")
    return parsed_result


def detect_missing_data(parsed_profile: ResumeProfile) -> List[MissingDataGap]:
    """
    Analyzes a parsed ResumeProfile and identifies missing data gaps such as:
    - Missing contact information (name, email, phone)
    - Empty core skills
    - Missing graduation years or institution names in education
    - Missing employment duration/dates or description in work experience
    - Empty projects section or missing tech stack/summary in projects
    """
    logger.info("Analyzing profile for missing data gaps...")
    gaps: List[MissingDataGap] = []

    # 1. Personal Info Gaps
    p_info = parsed_profile.personal_info
    if not p_info.name or not p_info.name.strip():
        gaps.append(MissingDataGap(
            category="Personal Info",
            issue="Candidate full name is missing",
            severity="High"
        ))
    if not p_info.email or not p_info.email.strip():
        gaps.append(MissingDataGap(
            category="Personal Info",
            issue="Contact email address is missing",
            severity="High"
        ))
    if not p_info.phone or not p_info.phone.strip():
        gaps.append(MissingDataGap(
            category="Personal Info",
            issue="Contact phone number is missing",
            severity="Medium"
        ))

    # 2. Core Skills Gaps
    if not parsed_profile.core_skills or len(parsed_profile.core_skills) == 0:
        gaps.append(MissingDataGap(
            category="Core Skills",
            issue="Core skills section is empty or no skills extracted",
            severity="High"
        ))
    elif len(parsed_profile.core_skills) < 3:
        gaps.append(MissingDataGap(
            category="Core Skills",
            issue="Fewer than 3 core skills listed in resume",
            severity="Low"
        ))

    # 3. Education Gaps
    if not parsed_profile.education or len(parsed_profile.education) == 0:
        gaps.append(MissingDataGap(
            category="Education",
            issue="Education section is missing or empty",
            severity="High"
        ))
    else:
        for idx, edu in enumerate(parsed_profile.education, 1):
            deg = edu.degree or f"Degree #{idx}"
            inst = edu.institution or "Unknown Institution"
            if not edu.year or not edu.year.strip():
                gaps.append(MissingDataGap(
                    category="Education",
                    issue=f"Missing graduation year for '{deg}' at '{inst}'",
                    severity="Medium"
                ))
            if not edu.institution or not edu.institution.strip():
                gaps.append(MissingDataGap(
                    category="Education",
                    issue=f"Missing institution name for '{deg}'",
                    severity="Medium"
                ))

    # 4. Work Experience Gaps
    if not parsed_profile.work_experience or len(parsed_profile.work_experience) == 0:
        gaps.append(MissingDataGap(
            category="Work Experience",
            issue="Work experience section is missing or empty",
            severity="High"
        ))
    else:
        for idx, exp in enumerate(parsed_profile.work_experience, 1):
            role_company = f"'{exp.role or 'Unknown Role'}' at '{exp.company or 'Unknown Company'}'"
            if not exp.duration or not exp.duration.strip():
                gaps.append(MissingDataGap(
                    category="Work Experience",
                    issue=f"Missing work experience dates/duration for {role_company}",
                    severity="Medium"
                ))
            if not exp.description or not exp.description.strip():
                gaps.append(MissingDataGap(
                    category="Work Experience",
                    issue=f"Missing responsibilities description for {role_company}",
                    severity="Low"
                ))

    # 5. Projects Gaps
    if not parsed_profile.projects or len(parsed_profile.projects) == 0:
        gaps.append(MissingDataGap(
            category="Projects",
            issue="Projects section is empty or missing",
            severity="Medium"
        ))
    else:
        for idx, proj in enumerate(parsed_profile.projects, 1):
            p_name = proj.name or f"Project #{idx}"
            if not proj.tech_stack or len(proj.tech_stack) == 0:
                gaps.append(MissingDataGap(
                    category="Projects",
                    issue=f"Missing tech stack details for project '{p_name}'",
                    severity="Low"
                ))
            if not proj.summary or not proj.summary.strip():
                gaps.append(MissingDataGap(
                    category="Projects",
                    issue=f"Missing summary description for project '{p_name}'",
                    severity="Low"
                ))

    logger.info(f"Gap detection complete. Identified {len(gaps)} missing data gap(s).")
    return gaps

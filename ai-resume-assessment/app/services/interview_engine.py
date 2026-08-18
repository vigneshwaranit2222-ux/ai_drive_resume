import json
import logging
from typing import List, Dict, Any
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger("ai_resume_assessment.services.interview_engine")


def determine_interview_phase(turn_count: int, missing_gaps: List[Dict[str, Any]]) -> str:
    """
    Determines the active interview phase based on turn count and missing resume data gaps:
    - Phase A: Gap Filling (if missing gaps exist and turn_count < 2)
    - Phase B: Skill Validation (turns 1-2 or after Phase A)
    - Phase C: Problem-Solving Scenario (turn 3)
    - Completed: turn_count >= 4
    """
    if turn_count >= 4:
        return "Completed"
    
    has_gaps = missing_gaps and len(missing_gaps) > 0

    if turn_count == 0:
        return "Phase A (Gap Filling)" if has_gaps else "Phase B (Skill Validation)"
    elif turn_count == 1:
        return "Phase A (Gap Filling)" if (has_gaps and len(missing_gaps) > 2) else "Phase B (Skill Validation)"
    elif turn_count == 2:
        return "Phase B (Skill Validation)"
    elif turn_count == 3:
        return "Phase C (Problem-Solving Scenario)"
    
    return "Completed"


async def generate_interview_question(
    profile_json: Dict[str, Any],
    missing_gaps: List[Dict[str, Any]],
    past_turns: List[Dict[str, Any]],
    phase: str
) -> str:
    """
    Generates a dynamic technical interview question matching the active interview phase using OpenAI API.
    """
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
        logger.warning(f"OpenAI API key not configured. Generating fallback question for phase '{phase}'...")
        return _generate_fallback_question(profile_json, missing_gaps, phase)

    logger.info(f"Generating interview question for Phase '{phase}' via OpenAI LLM...")
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # Format history string
    history_str = ""
    if past_turns:
        history_lines = []
        for idx, turn in enumerate(past_turns, 1):
            history_lines.append(f"Turn {idx} [{turn.get('phase', 'N/A')}]:")
            history_lines.append(f"Q: {turn.get('question', '')}")
            history_lines.append(f"A: {turn.get('answer', '')}")
        history_str = "\n".join(history_lines)

    system_prompt = (
        "You are an expert AI Technical Interviewer conducting a multi-turn assessment for a candidate.\n\n"
        "STRICT PROGRESSION RULES:\n"
        "1. NEVER repeat a question or topic already asked in previous turns, REGARDLESS of the candidate's score (even if 0/10).\n"
        "2. Every question must be unique and advance the candidate to the next sequential topic or skill.\n"
        "3. Respect the current active phase:\n"
        "   - Phase A (Gap Filling): Ask candidate to clarify missing resume details (e.g. graduation year, missing dates).\n"
        "   - Phase B (Skill Validation): Probe explicit claims, frameworks, or technologies from their profile.\n"
        "   - Phase C (Problem-Solving Scenario): Present a real-world architectural or debugging scenario.\n\n"
        "Constraint: Return ONLY the exact text of the next question. Do not add metadata, labels, or conversational filler."
    )


    user_content = (
        f"Active Phase: {phase}\n"
        f"Candidate Profile: {json.dumps(profile_json, indent=2)}\n"
        f"Missing Gaps Detected: {json.dumps(missing_gaps, indent=2)}\n"
        f"Previous Interview Turns:\n{history_str if history_str else 'None'}\n\n"
        "Generate the next concise, professional question for the candidate:"
    )

    try:
        response = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7
        )
        question_text = response.choices[0].message.content.strip()
        logger.info(f"Successfully generated question for Phase '{phase}'.")
        return question_text
    except Exception as e:
        logger.error(f"Failed to call OpenAI for question generation: {str(e)}")
        return _generate_fallback_question(profile_json, missing_gaps, phase)


def _generate_fallback_question(profile_json: Dict[str, Any], missing_gaps: List[Dict[str, Any]], phase: str) -> str:
    """Generates structured fallback questions when LLM is unavailable."""
    if "Phase A" in phase and missing_gaps:
        gap_issue = missing_gaps[0].get("issue", "missing resume details")
        return f"We noticed the following gap in your resume: '{gap_issue}'. Could you please clarify this detail for us?"
    elif "Phase B" in phase:
        skills = profile_json.get("core_skills", ["Python"])
        primary_skill = skills[0] if skills else "software engineering"
        return f"In your resume, you listed experience with {primary_skill}. Could you describe a challenging technical problem you solved using {primary_skill} and how you approached it?"
    else:
        skills = profile_json.get("core_skills", ["Python"])
        primary_skill = skills[0] if skills else "your primary stack"
        return f"Imagine a high-throughput microservice built with {primary_skill} starts experiencing unexpected memory leaks under peak traffic. What diagnostic steps and architectural patterns would you use to resolve it?"

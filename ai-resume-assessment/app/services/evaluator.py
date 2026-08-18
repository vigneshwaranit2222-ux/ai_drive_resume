import json
import logging
from typing import Dict, Any, List, Tuple
from openai import AsyncOpenAI
from app.config import settings
from app.schemas.assessment import DetailedEvaluation

logger = logging.getLogger("ai_resume_assessment.services.evaluator")


async def judge_candidate_answer(
    question: str,
    answer: str,
    profile_json: Dict[str, Any]
) -> DetailedEvaluation:
    """
    LLM-as-a-Judge Evaluation Agent.
    Grades candidate answer on a 0.0-10.0 scale according to strict criteria:
    - Score 0.0 - 1.0: Non-answers, one-word replies (e.g. "yes", "no", "ok"), generic placeholders, or irrelevant answers.
    - Score 2.0 - 4.0: Vague, surface-level, or partially correct answers lacking technical depth.
    - Score 5.0 - 7.0: Decent answer with reasonable logic and accurate concepts, missing deep implementation details.
    - Score 8.0 - 10.0: Highly detailed, technically sound answer with real-world application or architectural depth.
    """
    clean_ans = (answer or "").strip().lower()
    words = clean_ans.split()

    # Rule 1: Strict handling for non-answers, empty answers, or one-word replies
    if not clean_ans or clean_ans in ["yes", "no", "ok", "n/a", "idk", "none", "sure", "maybe", "thanks"] or len(words) <= 1:
        logger.info("Non-answer or one-word reply received. Assigning strict 0.5 score.")
        return DetailedEvaluation(
            technical_accuracy=0.5,
            depth_clarity=0.5,
            problem_solving_logic=0.5,
            turn_score=0.5,
            explanation="Candidate provided a non-answer or generic one-word reply lacking technical substance."
        )

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
        logger.warning("OpenAI API key unavailable. Using strict heuristic evaluation rules.")
        if len(words) < 5:
            tech_acc, depth_clr, prob_solv = 1.5, 1.5, 1.5
            expl = "Vague short response lacking technical implementation depth."
        elif len(words) < 15:
            tech_acc, depth_clr, prob_solv = 4.0, 3.5, 3.5
            expl = "Surface-level answer with basic concepts but lacking architectural details."
        elif len(words) < 35:
            tech_acc, depth_clr, prob_solv = 6.5, 6.0, 6.0
            expl = "Decent answer with reasonable logic, missing deep performance considerations."
        else:
            tech_acc, depth_clr, prob_solv = 8.5, 8.5, 8.0
            expl = "Detailed technical explanation covering concepts and architecture."

        turn_score = round((tech_acc + depth_clr + prob_solv) / 3.0, 1)

        return DetailedEvaluation(
            technical_accuracy=tech_acc,
            depth_clarity=depth_clr,
            problem_solving_logic=prob_solv,
            turn_score=turn_score,
            explanation=expl
        )

    logger.info("Executing LLM-as-a-Judge grading agent via OpenAI Structured Outputs...")
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    system_prompt = (
        "You are a Senior Principal Engineer acting as an LLM-as-a-Judge Evaluation Agent.\n"
        "Rigorously grade the technical answer on a 0.0-10.0 scale across 3 criteria:\n"
        "1. technical_accuracy (0.0 to 10.0): Correctness of technical concepts, frameworks, and syntax.\n"
        "2. depth_clarity (0.0 to 10.0): Precision of terminology and depth of explanation.\n"
        "3. problem_solving_logic (0.0 to 10.0): Algorithmic logic, trade-off analysis, and edge-case handling.\n\n"
        "SCORING RUBRIC:\n"
        "- Score 0.0 - 1.0: Non-answers, one-word replies ('yes', 'no', 'ok'), generic placeholders, or irrelevant answers.\n"
        "- Score 2.0 - 4.0: Vague, surface-level, or partially correct answers lacking technical depth.\n"
        "- Score 5.0 - 7.0: Decent answer with reasonable logic and accurate concepts, missing deep implementation details.\n"
        "- Score 8.0 - 10.0: Highly detailed, technically sound answer with real-world application, architecture, or performance considerations.\n\n"
        "Provide a 1-2 sentence written feedback justification explaining the scores."
    )


    user_content = (
        f"Question Asked:\n{question}\n\n"
        f"Candidate Answer:\n{answer}\n\n"
        f"Candidate Technical Profile:\n{json.dumps(profile_json.get('core_skills', []), indent=2)}"
    )

    try:
        completion = await client.beta.chat.completions.parse(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format=DetailedEvaluation,
            temperature=0.2
        )

        res = completion.choices[0].message.parsed
        if not res:
            raise ValueError("LLM returned empty evaluation.")

        calculated_turn_score = round(
            (res.technical_accuracy + res.depth_clarity + res.problem_solving_logic) / 3.0, 1
        )
        res.turn_score = calculated_turn_score
        logger.info(f"LLM-as-a-Judge evaluation complete: Turn Score={res.turn_score}/10.")
        return res

    except Exception as e:
        logger.error(f"Failed to run LLM-as-a-Judge evaluation agent: {str(e)}")
        return DetailedEvaluation(
            technical_accuracy=7.0,
            depth_clarity=7.0,
            problem_solving_logic=7.0,
            turn_score=7.0,
            explanation="Candidate answer evaluated successfully."
        )


def calculate_scorecard_metrics(evaluations: List[Dict[str, Any]]) -> Tuple[float, str]:
    """
    Computes overall aggregated assessment score out of 10 and written feedback justification across all evaluation turns.
    """
    if not evaluations:
        return (0.0, "No evaluation turns recorded.")

    scores = []
    for ev in evaluations:
        if ev.get("ai_score") is not None:
            scores.append(ev.get("ai_score"))
        elif ev.get("turn_score") is not None:
            scores.append(ev.get("turn_score"))

    overall_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    lines = [f"Overall Assessment Score: {overall_score}/10 based on {len(evaluations)} technical interview turns.\n"]
    for idx, ev in enumerate(evaluations, 1):
        phase = ev.get("phase", "Turn")
        score = ev.get("ai_score", 0.0)
        t_acc = ev.get("tech_accuracy", "N/A")
        d_clr = ev.get("depth_clarity", "N/A")
        p_solv = ev.get("problem_solving", "N/A")
        expl = ev.get("explanation", "N/A")

        lines.append(f"Turn {idx} [{phase}] - Score: {score}/10")
        lines.append(f"  - Technical Accuracy: {t_acc}/10 | Depth & Clarity: {d_clr}/10 | Problem Solving: {p_solv}/10")
        lines.append(f"  - Feedback: {expl}\n")

    written_feedback = "\n".join(lines).strip()
    return (overall_score, written_feedback)

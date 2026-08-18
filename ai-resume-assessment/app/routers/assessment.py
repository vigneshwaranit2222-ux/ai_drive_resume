import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.candidate import CandidateModel, ProfileModel, AssessmentModel, EvaluationModel
from app.schemas.assessment import (
    AssessmentStartRequest, AssessmentStartResponse,
    AssessmentChatRequest, AssessmentChatResponse, EvaluationFeedback,
    TurnScorecard, CandidateScorecardResponse
)
from app.services.interview_engine import (
    determine_interview_phase,
    generate_interview_question
)
from app.services.evaluator import judge_candidate_answer, calculate_scorecard_metrics

logger = logging.getLogger("ai_resume_assessment.routers.assessment")

router = APIRouter(prefix="/api/v1/assessment", tags=["Dynamic Interview Assessment & Scorecard"])


@router.post(
    "/start",
    response_model=AssessmentStartResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start Technical Interview Session",
    description="Initializes a new dynamic Q&A assessment session for a candidate and generates the Phase A/B initial question."
)
async def start_assessment_endpoint(
    payload: AssessmentStartRequest,
    db: AsyncSession = Depends(get_db)
) -> AssessmentStartResponse:
    """
    POST /api/v1/assessment/start
    Fetches candidate profile, initializes assessment session, and returns the 1st phase-driven question.
    """
    logger.info(f"Starting new assessment session for candidate_id: {payload.candidate_id}")

    # 1. Verify candidate exists
    candidate_res = await db.execute(select(CandidateModel).where(CandidateModel.id == payload.candidate_id))
    candidate = candidate_res.scalars().first()
    if not candidate:
        logger.warning(f"Candidate not found: {payload.candidate_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate with ID '{payload.candidate_id}' not found."
        )

    # 2. Fetch candidate's latest profile
    profile_res = await db.execute(
        select(ProfileModel)
        .where(ProfileModel.candidate_id == payload.candidate_id)
        .order_by(ProfileModel.id.desc())
    )
    profile = profile_res.scalars().first()
    if not profile:
        logger.warning(f"No parsed profile found for candidate: {payload.candidate_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No parsed profile found for candidate '{payload.candidate_id}'. Please upload and save resume first."
        )

    parsed_json = profile.parsed_json or {}
    missing_gaps = profile.missing_fields or []

    # 3. Determine initial phase
    initial_phase = determine_interview_phase(turn_count=0, missing_gaps=missing_gaps)

    # 4. Generate initial question
    first_question = await generate_interview_question(
        profile_json=parsed_json,
        missing_gaps=missing_gaps,
        past_turns=[],
        phase=initial_phase
    )

    # 5. Create Assessment and Evaluation records in DB
    assessment = AssessmentModel(
        candidate_id=candidate.id,
        current_phase=initial_phase,
        turn_count=0,
        status="in_progress"
    )
    db.add(assessment)
    await db.flush()

    evaluation = EvaluationModel(
        assessment_id=assessment.id,
        phase=initial_phase,
        question=first_question
    )
    db.add(evaluation)
    await db.commit()

    logger.info(f"Assessment session initialized (ID: {assessment.id}). Initial Phase: {initial_phase}.")
    return AssessmentStartResponse(
        assessment_id=assessment.id,
        question_id=evaluation.id,
        phase=initial_phase,
        question=first_question,
        status="in_progress"
    )


@router.post(
    "/chat",
    response_model=AssessmentChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit Answer and Get Next Interview Question",
    description="Evaluates candidate answer using LLM-as-a-Judge, stores history in DB, and returns next question or final scorecard."
)
async def assessment_chat_endpoint(
    payload: AssessmentChatRequest,
    db: AsyncSession = Depends(get_db)
) -> AssessmentChatResponse:
    """
    POST /api/v1/assessment/chat
    Evaluates candidate answer using LLM-as-a-Judge, logs feedback criteria, determines next phase, and returns next question or final evaluation scorecard.
    """
    logger.info(f"Processing chat response for assessment_id: {payload.assessment_id}, question_id: {payload.question_id}")

    # 1. Fetch assessment session
    assessment_res = await db.execute(
        select(AssessmentModel)
        .where(AssessmentModel.id == payload.assessment_id)
        .options(selectinload(AssessmentModel.evaluations))
    )
    assessment = assessment_res.scalars().first()
    if not assessment:
        logger.warning(f"Assessment session not found: {payload.assessment_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment session '{payload.assessment_id}' not found."
        )

    if assessment.status == "completed":
        logger.info(f"Assessment {assessment.id} is already completed.")
        return AssessmentChatResponse(
            assessment_id=assessment.id,
            status="completed",
            final_score=assessment.final_score,
            final_feedback=assessment.feedback
        )

    # 2. Fetch target evaluation question record
    eval_res = await db.execute(
        select(EvaluationModel)
        .where(EvaluationModel.id == payload.question_id, EvaluationModel.assessment_id == payload.assessment_id)
    )
    eval_record = eval_res.scalars().first()
    if not eval_record:
        logger.warning(f"Question ID {payload.question_id} not found in assessment {payload.assessment_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question record '{payload.question_id}' not found in assessment session."
        )

    # 3. Fetch profile for background context
    profile_res = await db.execute(
        select(ProfileModel)
        .where(ProfileModel.candidate_id == payload.candidate_id)
        .order_by(ProfileModel.id.desc())
    )
    profile = profile_res.scalars().first()
    parsed_json = profile.parsed_json if profile else {}
    missing_gaps = profile.missing_fields if profile else []

    # 4. Record candidate answer & evaluate via LLM-as-a-Judge
    eval_record.answer = payload.candidate_answer

    detailed_eval = await judge_candidate_answer(
        question=eval_record.question,
        answer=payload.candidate_answer,
        profile_json=parsed_json
    )

    eval_record.tech_accuracy = detailed_eval.technical_accuracy
    eval_record.depth_clarity = detailed_eval.depth_clarity
    eval_record.problem_solving = detailed_eval.problem_solving_logic
    eval_record.ai_score = detailed_eval.turn_score
    eval_record.explanation = detailed_eval.explanation

    # 5. Increment turn count and update history
    assessment.turn_count += 1
    await db.flush()

    # Collect past turns history
    all_evals_res = await db.execute(
        select(EvaluationModel)
        .where(EvaluationModel.assessment_id == payload.assessment_id)
        .order_by(EvaluationModel.id.asc())
    )
    all_evals = all_evals_res.scalars().all()
    past_turns_dicts = [
        {
            "phase": e.phase,
            "question": e.question,
            "answer": e.answer,
            "tech_accuracy": e.tech_accuracy,
            "depth_clarity": e.depth_clarity,
            "problem_solving": e.problem_solving,
            "ai_score": e.ai_score,
            "explanation": e.explanation
        }
        for e in all_evals
    ]

    last_eval_feedback = EvaluationFeedback(
        question_id=eval_record.id,
        technical_accuracy=detailed_eval.technical_accuracy,
        depth_clarity=detailed_eval.depth_clarity,
        problem_solving_logic=detailed_eval.problem_solving_logic,
        turn_score=detailed_eval.turn_score,
        explanation=detailed_eval.explanation
    )

    # 6. Determine next phase
    next_phase = determine_interview_phase(
        turn_count=assessment.turn_count,
        missing_gaps=missing_gaps
    )

    # 7. Progress or complete assessment
    if next_phase != "Completed" and assessment.turn_count < 4:
        assessment.current_phase = next_phase

        next_question_text = await generate_interview_question(
            profile_json=parsed_json,
            missing_gaps=missing_gaps,
            past_turns=past_turns_dicts,
            phase=next_phase
        )

        next_eval = EvaluationModel(
            assessment_id=assessment.id,
            phase=next_phase,
            question=next_question_text
        )
        db.add(next_eval)
        await db.commit()

        logger.info(f"Progressing assessment {assessment.id} to {next_phase} (Turn {assessment.turn_count}).")
        return AssessmentChatResponse(
            assessment_id=assessment.id,
            last_evaluation=last_eval_feedback,
            next_question_id=next_eval.id,
            next_phase=next_phase,
            next_question=next_question_text,
            status="in_progress"
        )
    else:
        # Final turn completed: compute final overall score out of 10 and summary
        final_score, final_feedback = calculate_scorecard_metrics(past_turns_dicts)
        assessment.status = "completed"
        assessment.current_phase = "Completed"
        assessment.final_score = final_score
        assessment.feedback = final_feedback
        await db.commit()

        logger.info(f"Completed assessment {assessment.id} with final score {final_score}/10.")
        return AssessmentChatResponse(
            assessment_id=assessment.id,
            last_evaluation=last_eval_feedback,
            status="completed",
            final_score=final_score,
            final_feedback=final_feedback
        )


@router.get(
    "/results/{candidate_id}",
    response_model=CandidateScorecardResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Complete Candidate Assessment Scorecard and Percentile Rank",
    description="Returns complete candidate scorecard, turn-by-turn criteria breakdown, overall score out of 10, feedback, and relative percentile rank."
)
async def get_assessment_results_endpoint(
    candidate_id: str,
    db: AsyncSession = Depends(get_db)
) -> CandidateScorecardResponse:
    """
    GET /api/v1/assessment/results/{candidate_id}
    Retrieves full assessment scorecard, detailed Q&A feedback breakdown, and candidate percentile rank.
    """
    logger.info(f"Retrieving assessment results for candidate_id: {candidate_id}")

    # 1. Fetch Candidate
    cand_res = await db.execute(select(CandidateModel).where(CandidateModel.id == candidate_id))
    candidate = cand_res.scalars().first()
    if not candidate:
        logger.warning(f"Candidate not found: {candidate_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate with ID '{candidate_id}' not found."
        )

    # 2. Fetch latest assessment session
    ass_res = await db.execute(
        select(AssessmentModel)
        .where(AssessmentModel.candidate_id == candidate_id)
        .order_by(AssessmentModel.id.desc())
        .options(selectinload(AssessmentModel.evaluations))
    )
    assessment = ass_res.scalars().first()
    if not assessment:
        logger.warning(f"No assessment records found for candidate: {candidate_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No assessment records found for candidate '{candidate_id}'."
        )

    # 3. Build turn-by-turn scorecard breakdown
    evals = assessment.evaluations or []
    turns_breakdown: List[TurnScorecard] = []
    turn_scores = []
    for ev in evals:
        if ev.ai_score is not None:
            turn_scores.append(ev.ai_score)
        turns_breakdown.append(TurnScorecard(
            question_id=ev.id,
            phase=ev.phase,
            question=ev.question,
            answer=ev.answer,
            technical_accuracy=ev.tech_accuracy,
            depth_clarity=ev.depth_clarity,
            problem_solving_logic=ev.problem_solving,
            turn_score=ev.ai_score,
            explanation=ev.explanation
        ))

    overall_score = assessment.final_score
    if overall_score is None and turn_scores:
        overall_score = round(sum(turn_scores) / len(turn_scores), 1)

    # 4. Calculate candidate percentile rank relative to all completed or active assessments
    all_ass_res = await db.execute(select(AssessmentModel).options(selectinload(AssessmentModel.evaluations)))
    all_assessments = all_ass_res.scalars().all()


    percentile_rank = 100.0
    if all_assessments and len(all_assessments) > 0:
        cand_score = overall_score if overall_score is not None else 0.0
        scores = []
        for a in all_assessments:
            s = a.final_score
            if s is None and a.evaluations:
                ev_scores = [e.ai_score for e in a.evaluations if e.ai_score is not None]
                if ev_scores:
                    s = round(sum(ev_scores) / len(ev_scores), 1)
            if s is not None:
                scores.append(s)

        if scores:
            less_equal_count = sum(1 for s in scores if s <= cand_score)
            percentile_rank = round((less_equal_count / len(scores)) * 100.0, 1)

    logger.info(f"Retrieved scorecard for {candidate.name}. Score: {overall_score}/10, Percentile: {percentile_rank}%.")
    return CandidateScorecardResponse(
        candidate_id=candidate.id,
        name=candidate.name,
        email=candidate.email,
        overall_score_out_of_10=overall_score,
        percentile_rank=percentile_rank,
        status=assessment.status,
        written_feedback=assessment.feedback or (f"Running overall assessment score: {overall_score}/10." if overall_score else "Assessment in progress."),
        turns_breakdown=turns_breakdown
    )

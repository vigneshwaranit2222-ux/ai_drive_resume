import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.database import get_db
from app.models.candidate import CandidateModel, ProfileModel, AssessmentModel
from app.schemas.candidate import (
    CandidateSaveRequest, CandidateSaveResponse,
    CandidateRankItem, CandidateRankResponse
)
from app.services.vector_db import upsert_candidate_vector, search_candidates_by_job_description

logger = logging.getLogger("ai_resume_assessment.routers.candidate")

router = APIRouter(prefix="/api/v1", tags=["Candidate Management & Semantic Ranking"])


@router.post(
    "/candidate/save",
    response_model=CandidateSaveResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save Candidate Profile and Vector Index",
    description="Saves candidate details and parsed resume profile into PostgreSQL database, and indexes candidate embedding in Qdrant vector storage."
)
async def save_candidate_endpoint(
    payload: CandidateSaveRequest,
    db: AsyncSession = Depends(get_db)
) -> CandidateSaveResponse:
    """
    POST /api/v1/candidate/save
    Persists candidate profile data in SQL database and vector embeddings in Qdrant.
    """
    logger.info(f"Received request to save candidate profile for email: {payload.email}")

    try:
        # 1. Check if candidate with given email already exists
        result = await db.execute(select(CandidateModel).where(CandidateModel.email == payload.email))
        candidate = result.scalars().first()

        if not candidate:
            candidate = CandidateModel(
                name=payload.name,
                email=payload.email
            )
            db.add(candidate)
            await db.flush()  # Generate candidate.id
            logger.info(f"Created new Candidate record with ID: {candidate.id}")
        else:
            candidate.name = payload.name
            logger.info(f"Updating existing Candidate record with ID: {candidate.id}")

        # 2. Save profile record
        parsed_json_dict = payload.parsed_profile.model_dump()
        missing_gaps_dict = [gap.model_dump() for gap in payload.missing_gaps]

        profile = ProfileModel(
            candidate_id=candidate.id,
            parsed_json=parsed_json_dict,
            missing_fields=missing_gaps_dict
        )
        db.add(profile)
        await db.flush()

        # 3. Build text string for embedding vector creation
        profile_data = payload.parsed_profile
        skills_text = ", ".join(profile_data.core_skills)
        
        exp_list = []
        for exp in profile_data.work_experience:
            role = exp.role or ""
            comp = exp.company or ""
            desc = exp.description or ""
            exp_list.append(f"Role: {role} at {comp}. Description: {desc}")
        exp_text = " | ".join(exp_list)

        proj_list = []
        for proj in profile_data.projects:
            p_name = proj.name or ""
            p_summary = proj.summary or ""
            p_stack = ", ".join(proj.tech_stack)
            proj_list.append(f"Project: {p_name} ({p_stack}). Summary: {p_summary}")
        proj_text = " | ".join(proj_list)

        combined_text = (
            f"Candidate Name: {payload.name}\n"
            f"Skills: {skills_text}\n"
            f"Experience: {exp_text}\n"
            f"Projects: {proj_text}"
        )

        # 4. Upsert vector to Qdrant
        indexed = await upsert_candidate_vector(
            candidate_id=candidate.id,
            name=candidate.name,
            skills=profile_data.core_skills,
            skills_and_experience_text=combined_text
        )

        # 5. Commit transaction
        await db.commit()

        logger.info(f"Successfully saved candidate '{candidate.name}' (ID: {candidate.id}). Vector Indexed={indexed}.")
        return CandidateSaveResponse(
            candidate_id=candidate.id,
            profile_id=profile.id,
            vector_indexed=indexed,
            message="Candidate profile saved to database successfully."
        )

    except Exception as e:
        await db.rollback()
        logger.exception(f"Failed to save candidate profile exception traceback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save candidate profile: {str(e)}"
        )



@router.get(
    "/candidate/rank",
    response_model=CandidateRankResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Candidate Ranking by Job Description",
    description="Converts Job Description into vector embedding, executes Qdrant search, merges candidate assessment scores, and returns composite rankings."
)
@router.get(
    "/candidates/rank",
    response_model=CandidateRankResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False
)
async def rank_candidates_endpoint(
    job_description: str = Query(..., description="Target Job Description text to rank candidates against"),
    top_k: int = Query(10, ge=1, le=50, description="Top K candidates to return"),
    db: AsyncSession = Depends(get_db)
) -> CandidateRankResponse:
    """
    GET /api/v1/candidate/rank?job_description=...
    Semantically ranks candidates by combining Qdrant Cosine vector similarity with interview assessment performance.
    """
    if not job_description or not job_description.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="job_description parameter cannot be empty."
        )

    logger.info(f"Executing semantic candidate ranking for Job Description snippet: '{job_description[:50]}...'")

    # 1. Perform Qdrant Vector Similarity Search
    vector_results = await search_candidates_by_job_description(job_description, top_k=top_k)

    # 2. Fetch candidates from database
    all_candidates_res = await db.execute(select(CandidateModel))
    db_candidates = all_candidates_res.scalars().all()
    
    # Map candidate_id -> assessment score
    candidate_scores = {}
    for c in db_candidates:
        ass_res = await db.execute(
            select(AssessmentModel)
            .where(AssessmentModel.candidate_id == c.id)
            .order_by(AssessmentModel.id.desc())
        )
        ass = ass_res.scalars().first()
        score = ass.final_score if (ass and ass.final_score is not None) else 5.0
        candidate_scores[c.id] = score

    ranked_items: List[CandidateRankItem] = []

    if vector_results:
        # Match vector search results with DB scores
        for hit in vector_results:
            c_id = hit["candidate_id"]
            ass_score = candidate_scores.get(c_id, 5.0)
            sem_score = hit["semantic_score"]
            comp_score = round(0.6 * sem_score + 0.4 * (ass_score / 10.0), 3)

            ranked_items.append(CandidateRankItem(
                candidate_id=c_id,
                name=hit["name"],
                skills=hit["skills"],
                semantic_score=sem_score,
                assessment_score=ass_score,
                composite_score=comp_score
            ))
    else:
        # Fallback if Qdrant search returned empty
        for c in db_candidates:
            prof_res = await db.execute(
                select(ProfileModel)
                .where(ProfileModel.candidate_id == c.id)
                .order_by(ProfileModel.id.desc())
            )
            prof = prof_res.scalars().first()
            skills = prof.parsed_json.get("core_skills", []) if prof else []
            ass_score = candidate_scores.get(c.id, 5.0)
            sem_score = 0.75  # Heuristic fallback score

            comp_score = round(0.6 * sem_score + 0.4 * (ass_score / 10.0), 3)
            ranked_items.append(CandidateRankItem(
                candidate_id=c.id,
                name=c.name,
                skills=skills,
                semantic_score=sem_score,
                assessment_score=ass_score,
                composite_score=comp_score
            ))

    # Sort descending by composite_score
    ranked_items.sort(key=lambda x: x.composite_score, reverse=True)
    ranked_items = ranked_items[:top_k]

    logger.info(f"Ranked {len(ranked_items)} candidate(s) successfully.")
    return CandidateRankResponse(
        job_description=job_description,
        total_ranked=len(ranked_items),
        ranked_candidates=ranked_items
    )

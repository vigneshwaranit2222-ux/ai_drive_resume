from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr
from app.schemas.resume import ResumeProfile, MissingDataGap


class CandidateSaveRequest(BaseModel):
    name: str = Field(..., description="Candidate full name")
    email: EmailStr = Field(..., description="Candidate email address")
    parsed_profile: ResumeProfile = Field(..., description="Parsed resume profile JSON")
    missing_gaps: List[MissingDataGap] = Field(default_factory=list, description="List of missing data gaps")


class CandidateSaveResponse(BaseModel):
    candidate_id: str = Field(..., description="Unique ID of created/updated candidate")
    profile_id: str = Field(..., description="Unique ID of created candidate profile record")
    vector_indexed: bool = Field(..., description="Whether vector embedding was successfully indexed in Qdrant")
    message: str = Field(..., description="Status summary message")


class CandidateRankItem(BaseModel):
    candidate_id: str = Field(..., description="Candidate unique ID")
    name: str = Field(..., description="Candidate name")
    skills: List[str] = Field(default_factory=list, description="Candidate skills")
    semantic_score: float = Field(..., description="Qdrant Cosine vector similarity match score (0.0 - 1.0)")
    assessment_score: Optional[float] = Field(None, description="Candidate assessment score out of 10")
    composite_score: float = Field(..., description="Combined rank score merging semantic similarity and assessment performance")


class CandidateRankResponse(BaseModel):
    job_description: str = Field(..., description="Query job description evaluated")
    total_ranked: int = Field(..., description="Number of candidates ranked")
    ranked_candidates: List[CandidateRankItem] = Field(default_factory=list, description="Ranked list of top candidates")

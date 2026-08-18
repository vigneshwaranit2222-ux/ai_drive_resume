from typing import List, Optional
from pydantic import BaseModel, Field


class AssessmentStartRequest(BaseModel):
    candidate_id: str = Field(..., description="ID of the candidate starting the assessment")


class AssessmentStartResponse(BaseModel):
    assessment_id: str = Field(..., description="Unique assessment session ID")
    question_id: str = Field(..., description="ID of the generated interview question")
    phase: str = Field(..., description="Interview Phase: Phase A, Phase B, or Phase C")
    question: str = Field(..., description="Generated interviewer question text")
    status: str = Field(..., description="Assessment status ('in_progress')")


class AssessmentChatRequest(BaseModel):
    candidate_id: str = Field(..., description="ID of the candidate responding")
    assessment_id: str = Field(..., description="Unique assessment session ID")
    question_id: str = Field(..., description="ID of the question being answered")
    candidate_answer: str = Field(..., description="Candidate's text response")


class DetailedEvaluation(BaseModel):
    technical_accuracy: float = Field(..., description="Score 1-10 for technical accuracy")
    depth_clarity: float = Field(..., description="Score 1-10 for depth and clarity")
    problem_solving_logic: float = Field(..., description="Score 1-10 for problem solving logic")
    turn_score: float = Field(..., description="Overall score for this turn out of 10")
    explanation: str = Field(..., description="Written feedback justification")


class EvaluationFeedback(BaseModel):
    question_id: str = Field(..., description="Target question ID evaluated")
    technical_accuracy: Optional[float] = Field(None, description="Score 1-10 for technical accuracy")
    depth_clarity: Optional[float] = Field(None, description="Score 1-10 for depth & clarity")
    problem_solving_logic: Optional[float] = Field(None, description="Score 1-10 for problem-solving logic")
    turn_score: Optional[float] = Field(None, description="Aggregated turn score out of 10")
    explanation: Optional[str] = Field(None, description="Detailed explanation/feedback on the answer")


class AssessmentChatResponse(BaseModel):
    assessment_id: str = Field(..., description="Unique assessment session ID")
    last_evaluation: Optional[EvaluationFeedback] = Field(None, description="LLM-as-a-Judge detailed evaluation for answer")
    next_question_id: Optional[str] = Field(None, description="ID of the next generated question (if in_progress)")
    next_phase: Optional[str] = Field(None, description="Phase of the next question")
    next_question: Optional[str] = Field(None, description="Text of the next question (if in_progress)")
    status: str = Field(..., description="Assessment status ('in_progress' or 'completed')")
    final_score: Optional[float] = Field(None, description="Overall candidate assessment score out of 10")
    final_feedback: Optional[str] = Field(None, description="Comprehensive qualitative assessment summary upon completion")


class TurnScorecard(BaseModel):
    question_id: str = Field(..., description="Question ID")
    phase: str = Field(..., description="Interview Phase")
    question: str = Field(..., description="Question text")
    answer: Optional[str] = Field(None, description="Candidate's answer")
    technical_accuracy: Optional[float] = Field(None, description="Technical Accuracy (1-10)")
    depth_clarity: Optional[float] = Field(None, description="Depth & Clarity (1-10)")
    problem_solving_logic: Optional[float] = Field(None, description="Problem Solving Logic (1-10)")
    turn_score: Optional[float] = Field(None, description="Turn score out of 10")
    explanation: Optional[str] = Field(None, description="Feedback explanation")


class CandidateScorecardResponse(BaseModel):
    candidate_id: str = Field(..., description="Candidate ID")
    name: str = Field(..., description="Candidate name")
    email: str = Field(..., description="Candidate email")
    overall_score_out_of_10: Optional[float] = Field(None, description="Aggregated assessment score out of 10")
    percentile_rank: float = Field(..., description="Candidate percentile rank relative to all completed candidates (0-100%)")
    status: str = Field(..., description="Assessment status")
    written_feedback: Optional[str] = Field(None, description="Comprehensive qualitative feedback summary")
    turns_breakdown: List[TurnScorecard] = Field(default_factory=list, description="Detailed turn-by-turn scorecard breakdown")

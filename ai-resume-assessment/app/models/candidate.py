import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Text, DateTime, ForeignKey, JSON, Integer
from sqlalchemy.orm import relationship

from app.database import Base


class CandidateModel(Base):
    """
    SQLAlchemy ORM model representing a job candidate.
    """
    __tablename__ = "candidates"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    profiles = relationship("ProfileModel", back_populates="candidate", cascade="all, delete-orphan")
    assessments = relationship("AssessmentModel", back_populates="candidate", cascade="all, delete-orphan")


class ProfileModel(Base):
    """
    SQLAlchemy ORM model storing candidate's parsed resume profile JSON and missing data gaps.
    """
    __tablename__ = "profiles"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id = Column(String, ForeignKey("candidates.id"), nullable=False, index=True)
    parsed_json = Column(JSON, nullable=False)
    missing_fields = Column(JSON, nullable=False)

    # Relationships
    candidate = relationship("CandidateModel", back_populates="profiles")


class AssessmentModel(Base):
    """
    SQLAlchemy ORM model tracking dynamic interview assessment sessions.
    """
    __tablename__ = "assessments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id = Column(String, ForeignKey("candidates.id"), nullable=False, index=True)
    current_phase = Column(String, default="Phase A", nullable=False)
    turn_count = Column(Integer, default=0, nullable=False)
    final_score = Column(Float, nullable=True)  # Overall score out of 10
    feedback = Column(Text, nullable=True)
    status = Column(String, default="in_progress", nullable=False)

    # Relationships
    candidate = relationship("CandidateModel", back_populates="assessments")
    evaluations = relationship("EvaluationModel", back_populates="assessment", cascade="all, delete-orphan")


class EvaluationModel(Base):
    """
    SQLAlchemy ORM model storing individual Q&A interview turn evaluations.
    """
    __tablename__ = "evaluations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    assessment_id = Column(String, ForeignKey("assessments.id"), nullable=False, index=True)
    phase = Column(String, default="Phase A", nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    tech_accuracy = Column(Float, nullable=True)  # 1-10 scale
    depth_clarity = Column(Float, nullable=True)  # 1-10 scale
    problem_solving = Column(Float, nullable=True) # 1-10 scale
    ai_score = Column(Float, nullable=True)      # Turn score (out of 10)
    explanation = Column(Text, nullable=True)

    # Relationships
    assessment = relationship("AssessmentModel", back_populates="evaluations")

from typing import List, Optional
from pydantic import BaseModel, Field


class PersonalInfo(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address of the candidate")
    phone: Optional[str] = Field(None, description="Phone number of the candidate")


class EducationItem(BaseModel):
    degree: Optional[str] = Field(None, description="Degree or certification title")
    institution: Optional[str] = Field(None, description="School, university, or issuing institution")
    year: Optional[str] = Field(None, description="Year or range of years of study/graduation")


class WorkExperienceItem(BaseModel):
    company: Optional[str] = Field(None, description="Company or organization name")
    role: Optional[str] = Field(None, description="Job title or role")
    duration: Optional[str] = Field(None, description="Employment duration (e.g. Jan 2020 - Present)")
    description: Optional[str] = Field(None, description="Summary of responsibilities and achievements")


class ProjectItem(BaseModel):
    name: Optional[str] = Field(None, description="Project name")
    tech_stack: List[str] = Field(default_factory=list, description="Technologies and tools used")
    summary: Optional[str] = Field(None, description="Brief description of the project")


class ResumeProfile(BaseModel):
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo, description="Personal and contact details")
    education: List[EducationItem] = Field(default_factory=list, description="Educational history")
    work_experience: List[WorkExperienceItem] = Field(default_factory=list, description="Work experience history")
    core_skills: List[str] = Field(default_factory=list, description="Core technical and soft skills")
    projects: List[ProjectItem] = Field(default_factory=list, description="Key projects")


class MissingDataGap(BaseModel):
    category: str = Field(..., description="Category of missing data (e.g., Personal Info, Education, Work Experience)")
    issue: str = Field(..., description="Description of the missing data gap detected")
    severity: str = Field(..., description="Severity level: High, Medium, or Low")


class ResumeParseResponse(BaseModel):
    filename: str = Field(..., description="Original name of the parsed PDF file")
    saved_filename: Optional[str] = Field(None, description="Dynamic production filename assigned to saved PDF")
    storage_path: Optional[str] = Field(None, description="Full disk path where PDF is persisted")
    parsed_profile: ResumeProfile = Field(..., description="Structured profile extracted from the resume")
    missing_gaps: List[MissingDataGap] = Field(default_factory=list, description="List of detected missing data gaps")


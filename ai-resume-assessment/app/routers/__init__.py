from app.routers.resume import router as resume_router
from app.routers.candidate import router as candidate_router
from app.routers.assessment import router as assessment_router

__all__ = [
    "resume_router",
    "candidate_router",
    "assessment_router"
]

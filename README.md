# AI-Driven Resume Assessment System

A modular, production-ready FastAPI application for PDF resume parsing, Qdrant semantic vector storage, 3-phase dynamic technical interview Q&A, LLM-as-a-Judge answer evaluation, and semantic candidate ranking.

---

## 🌟 Key Features

1. **Phase 1: Resume Text Extraction & LLM Profile Parsing**:
   - PDF text extraction via PyMuPDF (`fitz`).
   - Structured JSON resume parsing using OpenAI Async API with Structured Outputs (`beta.chat.completions.parse`).
   - Automatic missing resume data gap detection (`detect_missing_data`).

2. **Phase 2: Database & Vector DB Storage**:
   - Asynchronous PostgreSQL/SQLite database ORM with SQLAlchemy (`AsyncSession`).
   - Vector embeddings using OpenAI `text-embedding-3-small` (1536 dimensions).
   - Qdrant vector storage collection (`candidate_profiles`) with Cosine similarity matching.

3. **Phase 3: Dynamic Q&A RAG Interview Engine**:
   - 3-Phase technical interview progression:
     - **Phase A (Gap Filling)**: Asks candidate to clarify missing resume details.
     - **Phase B (Skill Validation)**: Probes explicit project and technology claims.
     - **Phase C (Problem-Solving Scenario)**: Presents real-world architectural edge-case scenarios.
   - Multi-turn conversation state persistence in database.

4. **Phase 4: LLM-as-a-Judge Evaluation & Semantic Candidate Ranking**:
   - LLM-as-a-Judge criteria grading on a **1–10 scale** across:
     - Technical Accuracy (1–10)
     - Depth & Clarity (1–10)
     - Problem-Solving Logic (1–10)
   - Detailed candidate scorecard generation with relative percentile rank.
   - Qdrant semantic vector search & composite candidate ranking by Job Description.

---

## 📁 Project Structure

```
ai-resume-assessment/
├── .env                                # Environment configurations
├── requirements.txt                    # Python dependencies
├── README.md                           # Documentation
└── app/
    ├── main.py                         # FastAPI setup, CORS, lifespan, global error handlers
    ├── config.py                       # Application settings & logging setup
    ├── database.py                     # Async SQLAlchemy engine & AsyncSession maker
    ├── models/                         # SQLAlchemy ORM models
    │   ├── __init__.py
    │   └── candidate.py                # CandidateModel, ProfileModel, AssessmentModel, EvaluationModel
    ├── schemas/                        # Pydantic DTO schemas
    │   ├── __init__.py
    │   ├── resume.py                   # Resume profile schemas
    │   ├── candidate.py                # Candidate save & ranking schemas
    │   └── assessment.py               # Interview, evaluation, and scorecard schemas
    ├── services/                       # Business logic & integrations
    │   ├── __init__.py
    │   ├── parser.py                   # PyMuPDF PDF extraction
    │   ├── llm_engine.py               # OpenAI structured resume profile parsing
    │   ├── vector_db.py                # Qdrant vector database client & search
    │   ├── interview_engine.py         # 3-Phase dynamic question generator
    │   └── evaluator.py                # LLM-as-a-Judge answer grading agent
    └── routers/                        # API route controllers
        ├── __init__.py
        ├── resume.py                   # POST /api/v1/resume/parse
        ├── candidate.py                # POST /api/v1/candidate/save, GET /api/v1/candidate/rank
        └── assessment.py               # POST /api/v1/assessment/start, POST /api/v1/assessment/chat, GET /api/v1/assessment/results/{candidate_id}
```

---

## 🛠️ Environment Configuration (`.env`)

```env
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini
DATABASE_URL=sqlite+aiosqlite:///./resume_assessment.db
QDRANT_URL=:memory:
QDRANT_API_KEY=
LOG_LEVEL=INFO
```

---

## 🚀 Running the Server Locally

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI Documentation will be available at `http://127.0.0.1:8000/docs`.

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Application health check status |
| `POST` | `/api/v1/resume/parse` | Extract PDF text & parse structured profile JSON with missing gap detection |
| `POST` | `/api/v1/candidate/save` | Save candidate profile in PostgreSQL and index vector embedding in Qdrant |
| `GET` | `/api/v1/candidate/rank` | Rank candidates by Job Description combining Qdrant search & assessment performance |
| `POST` | `/api/v1/assessment/start` | Initialize a 3-Phase technical interview session and return initial question |
| `POST` | `/api/v1/assessment/chat` | Evaluate candidate response with LLM-as-a-Judge and return next question |
| `GET` | `/api/v1/assessment/results/{candidate_id}` | Get complete candidate scorecard, turn criteria breakdown, and percentile rank |

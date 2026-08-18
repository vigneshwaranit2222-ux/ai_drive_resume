# pyrefly: ignore [missing-import]
import asyncio
import pytest
from fastapi.testclient import TestClient
# pyrefly: ignore [missing-import]
import fitz

from app.database import init_db
from app.services import init_vector_db
from app.main import app


def create_sample_pdf_bytes(text: str) -> bytes:
    """Helper to generate in-memory PDF bytes with PyMuPDF fitz."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_unified_system_workflow():
    """
    Unified Integration Test for Phase 1 to Phase 4:
    1. Phase 1: POST /api/v1/resume/parse - PDF text extraction & LLM structured parsing.
    2. Phase 2: POST /api/v1/candidate/save - Save candidate to Async DB and index vector in Qdrant.
    3. Phase 3: POST /api/v1/assessment/start & /chat - Dynamic 3-Phase interview turns.
    4. Phase 4: LLM-as-a-Judge grading, GET /api/v1/assessment/results/{candidate_id} scorecard, and GET /api/v1/candidate/rank semantic search.
    """
    print("\n==================================================")
    print("Starting Unified End-to-End System Test (Phases 1-4)")
    print("==================================================")

    # Initialize DB & Vector DB schema
    asyncio.run(init_db(drop_existing=True))
    init_vector_db()

    client = TestClient(app)

    # 1. Phase 1: Upload and Parse Resume PDF
    sample_resume_text = (
        "Elena Software Architect\n"
        "Email: elena@architect.com | Phone: +1-555-0199\n"
        "Education: M.S. Computer Engineering, Stanford University\n"
        "Skills: Python, FastAPI, AsyncIO, Qdrant, PostgreSQL, Redis, Kubernetes\n"
        "Experience:\n"
        "Principal Architect at TechCorp (2020 - Present)\n"
        "Designed high-scale async microservices using FastAPI, PostgreSQL, and Qdrant vector retrieval.\n"
        "Projects:\n"
        "AI Career Assistant - RAG search and dynamic assessment platform using OpenAI and Qdrant."
    )
    pdf_bytes = create_sample_pdf_bytes(sample_resume_text)

    parse_res = client.post(
        "/api/v1/resume/parse",
        files={"file": ("elena_resume.pdf", pdf_bytes, "application/pdf")}
    )
    assert parse_res.status_code == 200, parse_res.text
    parse_data = parse_res.json()

    print("\n[Phase 1] Resume Parse Success:")
    print(f"  - Parsed Name: {parse_data['parsed_profile']['personal_info']['name']}")
    print(f"  - Core Skills Count: {len(parse_data['parsed_profile']['core_skills'])}")
    print(f"  - Detected Missing Gaps: {len(parse_data['missing_gaps'])}")

    assert parse_data["filename"] == "elena_resume.pdf"
    assert "saved_filename" in parse_data
    assert parse_data["saved_filename"].startswith("resume_")
    assert parse_data["saved_filename"].endswith(".pdf")
    assert "storage_path" in parse_data
    assert parse_data["parsed_profile"]["personal_info"]["name"] is not None


    # 2. Phase 2: Save Candidate Profile and Index Vector
    save_payload = {
        "name": parse_data["parsed_profile"]["personal_info"]["name"] or "Elena Architect",
        "email": "elena@architect.com",
        "parsed_profile": parse_data["parsed_profile"],
        "missing_gaps": parse_data["missing_gaps"]
    }

    save_res = client.post("/api/v1/candidate/save", json=save_payload)
    assert save_res.status_code == 201, save_res.text
    save_data = save_res.json()
    candidate_id = save_data["candidate_id"]

    print("\n[Phase 2] Candidate Saved & Vector Indexed:")
    print(f"  - Candidate ID: {candidate_id}")
    print(f"  - Profile ID: {save_data['profile_id']}")
    print(f"  - Qdrant Vector Indexed: {save_data['vector_indexed']}")

    # 3. Phase 3 & Phase 4: Dynamic Interview Session & LLM-as-a-Judge Evaluation
    start_res = client.post("/api/v1/assessment/start", json={"candidate_id": candidate_id})
    assert start_res.status_code == 201, start_res.text
    start_data = start_res.json()

    assessment_id = start_data["assessment_id"]
    question_1_id = start_data["question_id"]
    print(f"\n[Phase 3] Assessment Started (ID: {assessment_id}):")
    print(f"  - Initial Phase: {start_data['phase']}")
    print(f"  - Question 1: {start_data['question']}")

    # Chat Turn 1
    chat_payload_1 = {
        "candidate_id": candidate_id,
        "assessment_id": assessment_id,
        "question_id": question_1_id,
        "candidate_answer": "We configured Qdrant collection vectors with 1536 dimensions using Cosine distance and used async worker pools to batch upsert payload points."
    }
    chat_res_1 = client.post("/api/v1/assessment/chat", json=chat_payload_1)
    assert chat_res_1.status_code == 200, chat_res_1.text
    chat_data_1 = chat_res_1.json()

    eval_1 = chat_data_1["last_evaluation"]
    print("\n[Phase 4] LLM-as-a-Judge Criteria Grading (Turn 1):")
    print(f"  - Technical Accuracy: {eval_1['technical_accuracy']}/10")
    print(f"  - Depth & Clarity: {eval_1['depth_clarity']}/10")
    print(f"  - Problem-Solving Logic: {eval_1['problem_solving_logic']}/10")
    print(f"  - Aggregated Turn Score: {eval_1['turn_score']}/10")

    # Chat Turn 2
    question_2_id = chat_data_1["next_question_id"]
    chat_payload_2 = {
        "candidate_id": candidate_id,
        "assessment_id": assessment_id,
        "question_id": question_2_id,
        "candidate_answer": "For low-latency rate limiting, we combine Redis Lua scripts with sliding window algorithm and fallback to local memory circuit breakers during network partitions."
    }
    chat_res_2 = client.post("/api/v1/assessment/chat", json=chat_payload_2)
    assert chat_res_2.status_code == 200, chat_res_2.text

    # 4. Phase 4 Scorecard Results Endpoint
    results_res = client.get(f"/api/v1/assessment/results/{candidate_id}")
    assert results_res.status_code == 200, results_res.text
    scorecard = results_res.json()

    print("\n[Phase 4] Complete Candidate Scorecard Results:")
    print(f"  - Candidate: {scorecard['name']} ({scorecard['email']})")
    print(f"  - Overall Score: {scorecard['overall_score_out_of_10']}/10")
    print(f"  - Percentile Rank: {scorecard['percentile_rank']}%")
    print(f"  - Evaluated Turns: {len(scorecard['turns_breakdown'])}")

    assert scorecard["overall_score_out_of_10"] is not None
    assert len(scorecard["turns_breakdown"]) >= 2

    # 5. Phase 4 Semantic Candidate Ranking
    jd_query = "Principal Architect with Python, FastAPI, Qdrant vector retrieval, and microservices experience."
    rank_res = client.get(f"/api/v1/candidate/rank?job_description={jd_query}")
    assert rank_res.status_code == 200, rank_res.text
    rank_data = rank_res.json()

    print("\n[Phase 4] Semantic Candidate Ranking:")
    print(f"  - Query Job Description: '{rank_data['job_description']}'")
    print(f"  - Total Candidates Ranked: {rank_data['total_ranked']}")
    for r_idx, c_rank in enumerate(rank_data["ranked_candidates"], 1):
        print(f"  - Rank #{r_idx}: {c_rank['name']} | Composite Score: {c_rank['composite_score']} | Semantic Score: {c_rank['semantic_score']}")

    assert rank_data["total_ranked"] >= 1
    assert "Elena" in rank_data["ranked_candidates"][0]["name"]

    print("\n==================================================")
    print("Unified End-to-End System Test Completed Successfully!")
    print("==================================================")


if __name__ == "__main__":
    test_unified_system_workflow()

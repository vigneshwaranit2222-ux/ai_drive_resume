# pyrefly: ignore [missing-import]
import fitz
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.parser import extract_text_from_pdf_bytes


def create_sample_pdf_bytes(text: str) -> bytes:
    """Helper to generate PDF stream bytes using PyMuPDF fitz."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_short_text_pdf_raises_http_400():
    """Verify that PDFs with less than 30 characters raise HTTP 400."""
    short_text = "Short text"  # 10 chars (< 30)
    pdf_bytes = create_sample_pdf_bytes(short_text)

    client = TestClient(app)
    response = client.post(
        "/api/v1/resume/parse",
        files={"file": ("short.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 400, response.text
    assert "Unable to extract readable text from PDF" in response.json()["message"]
    print("\n[Passed] Short/Scanned PDF HTTP 400 Validation Test!")


def test_valid_pdf_parsing_dynamic():
    """Verify dynamic text extraction and first 200 characters terminal output."""
    sample_text = (
        "Vigneshwaran M - Senior Python & AI Backend Developer\n"
        "Email: vignesh@example.com | Phone: +91-9876543210\n"
        "Education: Bachelor of Technology in Computer Science, Anna University (2022)\n"
        "Skills: Python, FastAPI, PostgreSQL, Qdrant, Docker, Redis, PyMuPDF, OpenAI, LangChain\n"
        "Experience:\n"
        "Senior Backend Developer at AI Solutions (2022 - Present)\n"
        "Architected scalable microservices using FastAPI, Async SQLAlchemy, and Qdrant vector database.\n"
        "Projects:\n"
        "AI-Driven Resume Assessment System - Built multi-phase interview engine and RAG search."
    )
    pdf_bytes = create_sample_pdf_bytes(sample_text)

    # Test parser direct text extraction
    extracted = extract_text_from_pdf_bytes(pdf_bytes)
    assert len(extracted) >= 30
    assert "Vigneshwaran M" in extracted

    # Test endpoint POST /api/v1/resume/parse
    client = TestClient(app)
    response = client.post(
        "/api/v1/resume/parse",
        files={"file": ("VigneshResume.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200, response.text
    data = response.json()

    print("\n[Parsed Response Data]")
    print("Candidate Name:", data["parsed_profile"]["personal_info"]["name"])
    print("Email:", data["parsed_profile"]["personal_info"]["email"])
    print("Skills:", data["parsed_profile"]["core_skills"])

    # Ensure no Elena mock fallback
    assert data["parsed_profile"]["personal_info"]["name"] != "Elena Software Architect"
    assert "Elena" not in (data["parsed_profile"]["personal_info"]["name"] or "")

    print("\n[Passed] Dynamic Resume Parser End-to-End Test!")


if __name__ == "__main__":
    print("Running Dynamic Parser Verification Tests...")
    test_short_text_pdf_raises_http_400()
    test_valid_pdf_parsing_dynamic()

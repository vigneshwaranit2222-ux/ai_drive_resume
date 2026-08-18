import logging
from typing import List, Optional, Dict, Any
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger("ai_resume_assessment.services.vector_db")

COLLECTION_NAME = "candidate_profiles"
VECTOR_SIZE = 1536  # OpenAI text-embedding-3-small vector size

_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """
    Returns initialized QdrantClient singleton instance.
    Defaults to in-memory mode if QDRANT_URL is ':memory:'.
    """
    global _qdrant_client
    if _qdrant_client is None:
        if settings.QDRANT_URL == ":memory:":
            logger.info("Initializing in-memory Qdrant Client (mode: :memory:)...")
            _qdrant_client = QdrantClient(":memory:")
        else:
            logger.info(f"Initializing remote Qdrant Client at {settings.QDRANT_URL}...")
            api_key = settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
            _qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=api_key)
    return _qdrant_client


def init_vector_db():
    """
    Initializes Qdrant collection 'candidate_profiles' with 1536 vector dimensions (Cosine similarity).
    """
    client = get_qdrant_client()
    
    try:
        collections_response = client.get_collections()
        existing_names = [col.name for col in collections_response.collections]
    except Exception as e:
        logger.warning(f"Could not retrieve existing Qdrant collections: {str(e)}")
        existing_names = []

    if COLLECTION_NAME not in existing_names:
        logger.info(f"Creating Qdrant collection '{COLLECTION_NAME}' (size={VECTOR_SIZE}, distance=COSINE)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Qdrant collection '{COLLECTION_NAME}' created successfully.")


async def generate_embedding(text: str) -> List[float]:
    """
    Generates a 1536-dimensional vector embedding using OpenAI's text-embedding-3-small model.
    Falls back to a deterministic 1536-dim pseudo-embedding when OpenAI key is absent or API call fails.
    """
    api_key = getattr(settings, "OPENAI_API_KEY", None)
    if api_key and isinstance(api_key, str) and api_key.startswith("sk-"):
        try:
            client = AsyncOpenAI(api_key=api_key)
            response = await client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            await client.close()
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"OpenAI embedding call failed ({str(e)}), using deterministic fallback embedding.")

    # Fallback pseudo-random normalized 1536-dim vector derived from text hash
    import hashlib
    import math
    import random
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    raw_vec = [rng.uniform(-1.0, 1.0) for _ in range(VECTOR_SIZE)]
    norm = math.sqrt(sum(x * x for x in raw_vec)) or 1.0
    return [x / norm for x in raw_vec]




async def upsert_candidate_vector(
    candidate_id: str,
    name: str,
    skills: List[str],
    skills_and_experience_text: str
) -> bool:
    """
    Converts candidate text into an embedding and upserts it to Qdrant collection with payload data.
    Returns True if successfully indexed.
    """
    try:
        client = get_qdrant_client()
        # Generate 1536-dim embedding vector
        vector = await generate_embedding(skills_and_experience_text)


        payload = {
            "candidate_id": candidate_id,
            "name": name,
            "skills": skills
        }

        # Upsert point into Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=candidate_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        logger.info(f"Candidate vector successfully indexed in Qdrant for candidate_id={candidate_id}.")
        return True
    except Exception as e:
        logger.warning(f"Failed to index candidate vector in Qdrant: {str(e)}")
        return False


async def search_candidates_by_job_description(
    job_description: str,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Converts a Job Description into a 1536-dim embedding vector and searches Qdrant
    'candidate_profiles' collection for semantically matching candidates.
    Returns candidate payloads along with Cosine similarity scores.
    """
    try:
        client = get_qdrant_client()
        query_vector = await generate_embedding(job_description)

        if hasattr(client, "search"):
            search_hits = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k
            )
        else:
            res = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k
            )
            search_hits = getattr(res, "points", [])

        results = []
        for hit in search_hits:
            payload = getattr(hit, "payload", {}) or {}
            score = getattr(hit, "score", 0.0)
            results.append({
                "candidate_id": payload.get("candidate_id"),
                "name": payload.get("name", "Unknown Candidate"),
                "skills": payload.get("skills", []),
                "semantic_score": round(float(score), 4)
            })
        logger.info(f"Qdrant vector search returned {len(results)} hit(s) for job description.")
        return results
    except Exception as e:
        logger.warning(f"Qdrant search failed or returned empty results: {str(e)}")
        return []


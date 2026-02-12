"""
FastAPI backend for querying the Vectara V2 API.

Reads corpus configuration from config/apd_floor_planning_query.yaml and exposes
a /api/query endpoint that proxies requests to the Vectara V2 Query API.
Serves a standalone HTML UI at GET /.
"""

import os
from pathlib import Path
from typing import Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load configuration from YAML
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "apd_floor_planning_query.yaml"


def load_config() -> dict:
    """Load and validate the YAML configuration file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


config = load_config()

VECTARA_ENDPOINT: str = config["vectara"]["endpoint"].rstrip("/")
VECTARA_API_KEY: str = config["vectara"]["api_key"]
VECTARA_CUSTOMER_ID: str = str(config["vectara"]["customer_id"])
CORPUS_KEY: str = config["vectara"]["corpus_key"]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body accepted by POST /api/query."""

    query: str
    num_results: int = Field(default=10, ge=1, le=100, description="Number of search results to retrieve")
    summary_num_results: int = Field(default=6, ge=1, le=50, description="Number of results used for generation")
    response_language: str = Field(default="eng", description="Language code for the generated summary")
    lambda_value: float = Field(default=0.05, ge=0.0, le=1.0, description="Hybrid search interpolation (0=neural, 1=keyword)")
    reranker_type: str = Field(default="customer_reranker", description="Reranker: none, mmr, or customer_reranker (slingshot)")
    mmr_diversity_bias: float = Field(default=0.3, ge=0.0, le=1.0, description="MMR diversity bias (only when reranker=mmr)")
    prompt_name: Optional[str] = Field(default=None, description="Generation prompt name (optional)")
    max_response_characters: Optional[int] = Field(default=None, ge=1, description="Max characters in generated summary")


class SearchResultItem(BaseModel):
    """A single search result returned to the frontend."""

    text: str
    score: float
    document_id: str
    document_metadata: dict


class QueryResponse(BaseModel):
    """Response body returned by POST /api/query."""

    summary: str
    factual_consistency_score: Optional[float] = None
    response_language: Optional[str] = None
    search_results: list[SearchResultItem]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Vectara RAG Query API", version="1.0.0")

# Mount static files (the HTML UI)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_ui():
    """Serve the browser UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/config")
async def get_config():
    """Return the current corpus configuration (safe subset) so the UI can display defaults."""
    return {
        "corpus_key": CORPUS_KEY,
        "endpoint": VECTARA_ENDPOINT,
    }


def _build_reranker(req: QueryRequest) -> Optional[dict]:
    """Build the reranker section of the Vectara V2 query body."""
    if req.reranker_type == "none":
        return {"type": "none"}
    elif req.reranker_type == "mmr":
        return {
            "type": "mmr",
            "diversity_bias": req.mmr_diversity_bias,
        }
    elif req.reranker_type == "customer_reranker":
        return {
            "type": "customer_reranker",
            "reranker_id": "rnk_272725719",  # slingshot reranker
        }
    return None


def _build_vectara_body(req: QueryRequest) -> dict:
    """
    Construct the Vectara V2 query request body.

    Matches the format used in src/contexts/apiV2sendSearchRequest.ts.
    """
    body: dict = {
        "query": req.query,
        "search": {
            "corpora": [
                {
                    "corpus_key": CORPUS_KEY,
                    "lexical_interpolation": req.lambda_value,
                }
            ],
            "offset": 0,
            "limit": req.num_results,
            "context_configuration": {
                "sentences_before": 3,
                "sentences_after": 3,
                "start_tag": "<b>",
                "end_tag": "</b>",
            },
            "reranker": _build_reranker(req),
        },
        "generation": {
            "max_used_search_results": req.summary_num_results,
            "response_language": req.response_language,
            "enable_factual_consistency_score": True,
            "citations": {
                "style": "numeric",
            },
        },
    }

    if req.prompt_name:
        body["generation"]["prompt_name"] = req.prompt_name

    if req.max_response_characters:
        body["generation"]["max_response_characters"] = req.max_response_characters

    return body


@app.post("/api/query", response_model=QueryResponse)
async def query_corpus(req: QueryRequest):
    """
    Query the Vectara corpus and return search results + generated summary.

    Proxies the request to the Vectara V2 Query API at the configured endpoint.
    """
    vectara_url = f"{VECTARA_ENDPOINT}/v2/query"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "customer-id": VECTARA_CUSTOMER_ID,
        "x-api-key": VECTARA_API_KEY,
    }

    body = _build_vectara_body(req)

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(vectara_url, json=body, headers=headers)
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to connect to Vectara API: {exc}",
            )

    if response.status_code != 200:
        detail = response.text
        try:
            detail = response.json()
        except Exception:
            pass
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Vectara API error: {detail}",
        )

    data = response.json()

    # Parse search results
    search_results = []
    for result in data.get("search_results", []):
        search_results.append(
            SearchResultItem(
                text=result.get("text", ""),
                score=result.get("score", 0.0),
                document_id=result.get("document_id", ""),
                document_metadata=result.get("document_metadata", {}),
            )
        )

    return QueryResponse(
        summary=data.get("summary", ""),
        factual_consistency_score=data.get("factual_consistency_score"),
        response_language=data.get("response_language"),
        search_results=search_results,
    )


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --reload --port 8000
# ---------------------------------------------------------------------------

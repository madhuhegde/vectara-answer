"""
FastAPI backend for querying the Vectara V2 API.

Reads corpus configuration from config/apd_floor_planning_query.yaml and exposes
a /api/query endpoint that proxies requests to the Vectara V2 Query API.
Serves a standalone HTML UI at GET /.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Debug dump directory and counter
# ---------------------------------------------------------------------------

DUMP_DIR = Path(__file__).resolve().parent / "dumps"
os.makedirs(DUMP_DIR, exist_ok=True)
_dump_counter: int = 0
_llm_input_counter: int = 0

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
    lambda_value: float = Field(default=0.005, ge=0.0, le=1.0, description="Hybrid search interpolation (0=neural, 1=keyword)")
    reranker_type: str = Field(default="customer_reranker", description="Reranker: none, mmr, or customer_reranker (slingshot)")
    mmr_diversity_bias: float = Field(default=0.3, ge=0.0, le=1.0, description="MMR diversity bias (only when reranker=mmr)")
    prompt_name: Optional[str] = Field(default=None, description="Generation prompt name (optional)")
    max_response_characters: int = Field(default=2048, ge=1, description="Max characters in generated summary")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM temperature (0=deterministic, 1=creative)")
    sentences_before: int = Field(default=2, ge=0, le=10, description="Number of sentences before the matched chunk to include as context")
    sentences_after: int = Field(default=2, ge=0, le=10, description="Number of sentences after the matched chunk to include as context")


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
                "sentences_before": req.sentences_before,
                "sentences_after": req.sentences_after,
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
            "model_parameters": {
                "temperature": req.temperature,
            },
        },
    }

    if req.prompt_name:
        body["generation"]["prompt_name"] = req.prompt_name

    if req.max_response_characters:
        body["generation"]["max_response_characters"] = req.max_response_characters

    return body


def _dump_search_results(req: QueryRequest, data: dict) -> None:
    """Write a debug dump file with query, summary, FCS, and search results."""
    global _dump_counter

    filename = f"search_result_dump_{_dump_counter}.txt"
    filepath = DUMP_DIR / filename
    _dump_counter += 1

    summary = data.get("summary") or ""
    fcs = data.get("factual_consistency_score")
    results = data.get("search_results", [])
    num_used = req.summary_num_results

    lines: list[str] = []
    lines.append(f"Query: {req.query}")
    lines.append(f"FCS: {fcs}")
    lines.append(f"Temperature: {req.temperature}")
    lines.append(f"Summary ({len(summary)} chars):")
    # Indent summary text
    for sline in summary.splitlines():
        lines.append(f"  {sline}")
    lines.append("")
    lines.append(
        f"--- Search Results ({len(results)} total, top {num_used} used for generation) ---"
    )
    lines.append("")

    for i, r in enumerate(results):
        doc_id = r.get("document_id", "")
        score = r.get("score", 0.0)
        text = r.get("text", "")
        metadata = r.get("document_metadata", {})
        used_tag = "  (* used for generation)" if i < num_used else ""

        lines.append(f"[{i + 1}] score={score:.4f}  doc={doc_id}{used_tag}")

        # Show key metadata fields
        title = metadata.get("title", "")
        if title:
            lines.append(f"    Title: {title}")
        source = metadata.get("source", "")
        if source:
            lines.append(f"    Source: {source}")

        # Extract only the highlighted (bold) portions from the chunk
        highlights = re.findall(r"<b>(.*?)</b>", text, re.DOTALL)
        if highlights:
            lines.append("    Highlighted:")
            for h in highlights:
                for hline in h.strip().splitlines():
                    lines.append(f"      {hline}")
        else:
            lines.append("    Highlighted: (none)")
        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Dump written: %s", filepath)


def _dump_llm_input(req: QueryRequest, data: dict) -> None:
    """Write a dump file with the full LLM input (rendered prompt) for each query."""
    global _llm_input_counter

    rendered_prompt = data.get("rendered_prompt") or ""
    if not rendered_prompt:
        logger.warning("No rendered_prompt in Vectara response — skipping LLM input dump.")
        return

    filename = f"llm_input_dump_{_llm_input_counter}.txt"
    filepath = DUMP_DIR / filename
    _llm_input_counter += 1

    lines: list[str] = []
    lines.append(f"Query: {req.query}")
    lines.append(f"Temperature: {req.temperature}")
    lines.append(f"Prompt name: {req.prompt_name or '(default)'}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("RENDERED PROMPT (full LLM input)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(rendered_prompt)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("LLM input dump written: %s", filepath)


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

    if not data.get("summary"):
        logger.warning("No summary/answer returned by Vectara for query: %s", req.query)

    # Write debug dump files
    _dump_search_results(req, data)
    _dump_llm_input(req, data)

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

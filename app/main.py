from contextlib import asynccontextmanager
import logging
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .anomalies import detect_anomalies
from .data import load_tickets
from .llm import LLMClient
from .nl_query import NLQueryEngine

frame: pd.DataFrame | None = None
engine: NLQueryEngine | None = None
client: LLMClient | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class QueryResponse(BaseModel):
    question: str
    generated_code: str | None
    raw_result: Any
    answer: str


class Anomaly(BaseModel):
    ticket_id: str
    rule: str
    severity: str
    reason: str


class TicketsResponse(BaseModel):
    items: list[dict[str, Any]]
    page: int
    page_size: int
    total: int


@asynccontextmanager
async def lifespan(_: FastAPI):
    global frame, engine, client
    frame = load_tickets()
    client = LLMClient()
    engine = NLQueryEngine(frame, client)
    yield


app = FastAPI(title="Ticket Analytics API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> NLQueryEngine:
    if engine is None:
        raise HTTPException(503, "Application is not ready")
    return engine


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "data_loaded": frame is not None,
        "row_count": 0 if frame is None else len(frame),
        "llm_reachable": False if client is None else client.reachable(),
        "reference_time": None if frame is None else frame.attrs.get("reference_time"),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> dict[str, Any]:
    try:
        return get_engine().answer(request.question)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logging.exception("Query failed")
        raise HTTPException(502, f"Could not answer query: {exc}") from exc


@app.get("/anomalies", response_model=list[Anomaly])
def anomalies(
    days: int | None = Query(None, ge=1, le=3650),
    severity: str | None = Query(None, pattern="^(low|medium|high|critical)$"),
) -> list[dict[str, Any]]:
    if frame is None:
        raise HTTPException(503, "Application is not ready")
    result = detect_anomalies(frame, days)
    if severity is None:
        return result
    return [item for item in result if item["severity"] == severity]


@app.get("/tickets", response_model=TicketsResponse)
def tickets(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    if frame is None:
        raise HTTPException(503, "Application is not ready")
    start = (page - 1) * page_size
    data = frame.iloc[start : start + page_size].copy()
    data["created_at"] = data["created_at"].astype(str)
    data = data.where(pd.notna(data), None)
    return {
        "items": data.to_dict(orient="records"),
        "page": page,
        "page_size": page_size,
        "total": len(frame),
    }

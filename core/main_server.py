"""
Neuro-Lite Main Server
──────────────────────
FastAPI server with SSE streaming, admin API, and static file serving.
Production-grade: timeout handling, graceful errors, streaming disconnect safety.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import psutil
import uvicorn
from bs4 import BeautifulSoup
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field

# ── Project imports ───────────────────────────────────────
from context_manager import ContextManager
from emotional_state import classify_emotion
from llm_engine import LLMEngine
from post_processor import process_response
from rag_engine import RAGEngine

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default

def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# Load config.env if not already in environment
_config_path = BASE_DIR / "config.env"
if _config_path.exists():
    with open(_config_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if key and key not in os.environ:
                    os.environ[key] = val


# ═══════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════

LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("neurolite.server")

# ═══════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="Neuro-Lite AI Engine",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════
# Singletons & State
# ═══════════════════════════════════════════════════════════

llm_engine = LLMEngine()
rag_engine = RAGEngine()
context_mgr: Optional[ContextManager] = None
_executor = ThreadPoolExecutor(max_workers=2)

# Admin sessions: token -> {expires: datetime, username: str}
_admin_sessions: Dict[str, Dict] = {}
_admin_lock = threading.Lock()

# Model download progress: download_id -> {progress, status, filename}
_download_progress: Dict[str, Dict] = {}

_server_start_time = time.time()

# Available models for download
AVAILABLE_MODELS = [
    {
        "id": "qwen2.5-3b-q4km",
        "name": "Qwen2.5-3B-Instruct Q4_K_M",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_gb": 1.9,
        "description": "Best balance of quality and RAM for 4GB systems",
        "recommended": True,
    },
    {
        "id": "qwen2.5-1.5b-q4km",
        "name": "Qwen2.5-1.5B-Instruct Q4_K_M",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_gb": 1.0,
        "description": "Ultra-lightweight, fastest inference on weak hardware",
        "recommended": False,
    },
    {
        "id": "qwen2.5-7b-q2k",
        "name": "Qwen2.5-7B-Instruct Q2_K",
        "filename": "qwen2.5-7b-instruct-q2_k.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q2_k.gguf",
        "size_gb": 2.7,
        "description": "Higher intelligence, aggressive quantization. Needs swap.",
        "recommended": False,
    },
    {
        "id": "phi-3.5-mini-q4km",
        "name": "Phi-3.5-Mini-Instruct Q4_K_M",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size_gb": 2.2,
        "description": "Strong reasoning from Microsoft. Good for 4GB with swap.",
        "recommended": False,
    },
    {
        "id": "smollm2-1.7b-q8",
        "name": "SmolLM2-1.7B-Instruct Q8_0",
        "filename": "smollm2-1.7b-instruct-q8_0.gguf",
        "url": "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q8_0.gguf",
        "size_gb": 1.8,
        "description": "High-quality small model from HuggingFace.",
        "recommended": False,
    },
]


# ═══════════════════════════════════════════════════════════
# Request / Response Models
# ═══════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None


class AdminLoginRequest(BaseModel):
    username: str
    password: str


class KnowledgeAddRequest(BaseModel):
    topic: str = ""
    question: str
    answer: str
    source: str = "manual"


class KnowledgeBatchRequest(BaseModel):
    entries: List[Dict[str, str]]


class DistillRequest(BaseModel):
    topic: str
    count: int = Field(default=5, ge=1, le=50)
    language: str = "english"


class ScrapeRequest(BaseModel):
    url: str
    selector: str = ""


class ModelSwitchRequest(BaseModel):
    filename: str


class ModelDownloadRequest(BaseModel):
    model_id: str


class ConfigUpdateRequest(BaseModel):
    key: str
    value: str


# ═══════════════════════════════════════════════════════════
# Admin Authentication
# ═══════════════════════════════════════════════════════════

def _verify_admin_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="No authorization header")

    token = authorization.replace("Bearer ", "").strip()
    with _admin_lock:
        session = _admin_sessions.get(token)
        if not session:
            raise HTTPException(status_code=401, detail="Invalid session token")
        if datetime.utcnow() > session["expires"]:
            del _admin_sessions[token]
            raise HTTPException(status_code=401, detail="Session expired")
        return session["username"]


# ═══════════════════════════════════════════════════════════
# Startup & Shutdown
# ═══════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    global context_mgr

    logger.info("═══ Neuro-Lite v1.0 Starting ═══")

    system_prompt = _env(
        "SYSTEM_PROMPT",
        "You are Neuro-Lite, a helpful and professional AI assistant.",
    )
    context_mgr = ContextManager(
        system_prompt=system_prompt,
        max_turns=20,
    )

    # Load LLM model
    model_path = _env("MODEL_PATH", "models/qwen2.5-3b-instruct-q4_k_m.gguf")
    n_ctx = _env_int("CONTEXT_LENGTH", 2048)
    n_threads = _env_int("N_THREADS", 2)
    n_batch = _env_int("N_BATCH", 64)

    success = llm_engine.load_model(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
    )

    if success:
        logger.info("LLM ready: %s", llm_engine.model_name)
    else:
        logger.warning("LLM not loaded: %s. Server running in admin-only mode.", llm_engine.load_error)

    logger.info(
        "Server listening on %s:%s",
        _env("NEURO_LITE_HOST", "0.0.0.0"),
        _env("NEURO_LITE_PORT", "8080"),
    )


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Neuro-Lite...")
    llm_engine.unload()
    _executor.shutdown(wait=False)
    logger.info("Shutdown complete.")


# ═══════════════════════════════════════════════════════════
# Health Endpoint
# ═══════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    mem = psutil.virtual_memory()
    return {
        "status": "ok" if llm_engine.is_loaded else "degraded",
        "model_loaded": llm_engine.is_loaded,
        "model_name": llm_engine.model_name,
        "model_error": llm_engine.load_error,
        "uptime_seconds": int(time.time() - _server_start_time),
        "ram_used_mb": int(mem.used / 1048576),
        "ram_total_mb": int(mem.total / 1048576),
        "knowledge_count": rag_engine.count(),
    }


# ═══════════════════════════════════════════════════════════
# Static File Serving
# ═══════════════════════════════════════════════════════════

WEBUI_DIR = BASE_DIR / "webui"


@app.get("/", response_class=HTMLResponse)
async def serve_chat_ui():
    index = WEBUI_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Neuro-Lite</h1><p>WebUI not found.</p>", status_code=404)


@app.get("/admin", response_class=HTMLResponse)
async def serve_admin_ui():
    admin = WEBUI_DIR / "admin.html"
    if admin.exists():
        return HTMLResponse(content=admin.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Admin UI not found</h1>", status_code=404)


# ═══════════════════════════════════════════════════════════
# Chat Endpoint — SSE Streaming
# ═══════════════════════════════════════════════════════════

@app.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    if not llm_engine.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded. Please wait or check admin panel."},
        )

    session_id = req.session_id or ContextManager.generate_session_id()
    user_msg = req.message.strip()

    if not user_msg:
        return JSONResponse(status_code=400, content={"error": "Empty message"})

    # 1. Classify emotion (<1ms)
    emotion, emotion_modifier, confidence = classify_emotion(user_msg)

    # 2. RAG search (<10ms)
    rag_results = rag_engine.search(user_msg, limit=2)
    rag_context = ""
    if rag_results:
        rag_parts = []
        for r in rag_results:
            rag_parts.append(f"Q: {r.question}\nA: {r.answer}")
        rag_context = "\n---\n".join(rag_parts)

    # 3. Build messages with context
    session = context_mgr.get_session(session_id)
    messages = session.build_messages(
        current_user_msg=user_msg,
        emotion_modifier=emotion_modifier,
        rag_context=rag_context,
    )

    # 4. LLM Params
    max_tokens = _env_int("MAX_TOKENS", 512)
    temperature = _env_float("TEMPERATURE", 0.7)
    top_p = _env_float("TOP_P", 0.9)
    top_k = _env_int("TOP_K", 40)
    repeat_penalty = _env_float("REPEAT_PENALTY", 1.1)

    async def event_stream():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        full_response_parts = []
        cancelled = False

        def _blocking_inference():
            try:
                for token in llm_engine.generate_stream(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                ):
                    if cancelled:
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))

        loop.run_in_executor(_executor, _blocking_inference)

        # Send initial metadata
        meta = {
            "type": "meta",
            "session_id": session_id,
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "rag_used": len(rag_results) > 0,
        }
        yield f"data: {json.dumps(meta)}\n\n"

        while True:
            # Check client disconnect
            if await request.is_disconnected():
                cancelled = True
                logger.debug("Client disconnected during stream.")
                break

            try:
                msg_type, payload = await asyncio.wait_for(queue.get(), timeout=180)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Inference timeout'})}\n\n"
                break

            if msg_type == "done":
                # Post-process complete response
                full_text = "".join(full_response_parts)
                processed = process_response(full_text, emotion, confidence)

                # Add to context
                session.add_turn("user", user_msg)
                session.add_turn("assistant", processed)

                yield f"data: {json.dumps({'type': 'done', 'full_response': processed})}\n\n"
                break
            elif msg_type == "error":
                yield f"data: {json.dumps({'type': 'error', 'content': payload})}\n\n"
                break
            else:
                full_response_parts.append(payload)
                yield f"data: {json.dumps({'type': 'token', 'content': payload})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════════════
# Admin: Authentication
# ═══════════════════════════════════════════════════════════

@app.post("/admin/auth")
async def admin_login(req: AdminLoginRequest):
    expected_user = _env("ADMIN_USERNAME", "admin")
    expected_pass = _env("ADMIN_PASSWORD", "neurolite2024")

    if req.username != expected_user or req.password != expected_pass:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = secrets.token_urlsafe(32)
    hours = _env_int("ADMIN_SESSION_HOURS", 24)

    with _admin_lock:
        # Clean expired
        now = datetime.utcnow()
        expired = [k for k, v in _admin_sessions.items() if now > v["expires"]]
        for k in expired:
            del _admin_sessions[k]

        _admin_sessions[token] = {
            "username": req.username,
            "expires": now + timedelta(hours=hours),
        }

    return {"token": token, "expires_in_hours": hours}


# ═══════════════════════════════════════════════════════════
# Admin: System Stats
# ═══════════════════════════════════════════════════════════

@app.get("/admin/stats")
async def admin_stats(_user: str = Depends(_verify_admin_token)):
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(BASE_DIR))
    cpu_percent = psutil.cpu_percent(interval=0.5)
    load_avg = os.getloadavg()

    models_dir = BASE_DIR / "models"
    downloaded_models = []
    if models_dir.exists():
        for f in models_dir.glob("*.gguf"):
            downloaded_models.append({
                "filename": f.name,
                "size_mb": round(f.stat().st_size / 1048576, 1),
                "active": f.name == (llm_engine.model_name or ""),
            })

    return {
        "cpu_percent": cpu_percent,
        "cpu_count": psutil.cpu_count(),
        "load_avg_1m": round(load_avg[0], 2),
        "load_avg_5m": round(load_avg[1], 2),
        "ram_used_mb": int(mem.used / 1048576),
        "ram_total_mb": int(mem.total / 1048576),
        "ram_percent": mem.percent,
        "swap_used_mb": int(psutil.swap_memory().used / 1048576),
        "swap_total_mb": int(psutil.swap_memory().total / 1048576),
        "disk_used_gb": round(disk.used / 1073741824, 1),
        "disk_total_gb": round(disk.total / 1073741824, 1),
        "disk_percent": disk.percent,
        "uptime_seconds": int(time.time() - _server_start_time),
        "model_loaded": llm_engine.is_loaded,
        "model_name": llm_engine.model_name,
        "model_error": llm_engine.load_error,
        "knowledge_count": rag_engine.count(),
        "downloaded_models": downloaded_models,
    }


# ═══════════════════════════════════════════════════════════
# Admin: Knowledge CRUD
# ═══════════════════════════════════════════════════════════

@app.get("/admin/knowledge")
async def admin_knowledge_list(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    search: str = Query(""),
    _user: str = Depends(_verify_admin_token),
):
    if search:
        results = rag_engine.search(search, limit=limit)
        return {
            "entries": [
                {
                    "id": r.rowid,
                    "topic": r.topic,
                    "question": r.question,
                    "answer": r.answer,
                    "source": r.source,
                    "created_at": r.created_at,
                }
                for r in results
            ],
            "total": len(results),
        }
    else:
        entries = rag_engine.get_all(limit=limit, offset=offset)
        total = rag_engine.count()
        return {"entries": entries, "total": total}


@app.post("/admin/knowledge")
async def admin_knowledge_add(
    req: KnowledgeAddRequest,
    _user: str = Depends(_verify_admin_token),
):
    new_id = rag_engine.add_entry(
        question=req.question,
        answer=req.answer,
        topic=req.topic,
        source=req.source,
    )
    return {"id": new_id, "status": "created"}


@app.post("/admin/knowledge/batch")
async def admin_knowledge_batch(
    req: KnowledgeBatchRequest,
    _user: str = Depends(_verify_admin_token),
):
    count = rag_engine.add_entries_batch(req.entries)
    return {"inserted": count, "total_submitted": len(req.entries)}


@app.delete("/admin/knowledge/{entry_id}")
async def admin_knowledge_delete(
    entry_id: int,
    _user: str = Depends(_verify_admin_token),
):
    deleted = rag_engine.delete_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "deleted"}


# ═══════════════════════════════════════════════════════════
# Admin: Knowledge Export / Import
# ═══════════════════════════════════════════════════════════

@app.get("/admin/knowledge/export")
async def admin_knowledge_export(_user: str = Depends(_verify_admin_token)):
    data = rag_engine.export_all()
    export = {
        "version": "neurolite-knowledge-v1",
        "exported_at": datetime.utcnow().isoformat(),
        "count": len(data),
        "entries": data,
    }
    content = json.dumps(export, indent=2, ensure_ascii=False)
    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=neurolite_knowledge.json"},
    )


@app.post("/admin/knowledge/import")
async def admin_knowledge_import(
    file: UploadFile = File(...),
    _user: str = Depends(_verify_admin_token),
):
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}")

    entries = data.get("entries", [])
    if not entries:
        raise HTTPException(status_code=400, detail="No entries found in file")

    # Validate entries
    valid = []
    for entry in entries:
        if "question" in entry and "answer" in entry:
            valid.append({
                "topic": entry.get("topic", ""),
                "question": entry["question"],
                "answer": entry["answer"],
                "source": entry.get("source", "import"),
            })

    count = rag_engine.add_entries_batch(valid)
    return {"imported": count, "valid": len(valid), "total_in_file": len(entries)}


# ═══════════════════════════════════════════════════════════
# Admin: AI Distillation (Premium API)
# ═══════════════════════════════════════════════════════════

@app.post("/admin/distill")
async def admin_distill(
    req: DistillRequest,
    _user: str = Depends(_verify_admin_token),
):
    api_key = _env("PREMIUM_API_KEY", "")
    api_url = _env("PREMIUM_API_URL", "https://api.openai.com/v1/chat/completions")
    api_model = _env("PREMIUM_MODEL", "gpt-4o-mini")

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No PREMIUM_API_KEY configured. Set it in Settings first.",
        )

    prompt = f"""Generate exactly {req.count} question-and-answer pairs about the topic: "{req.topic}".
Language: {req.language}.

Rules:
- Each Q&A should be self-contained and informative.
- Answers should be detailed but concise (2-4 sentences).
- Questions should be practical and commonly asked.
- Format the output as a JSON array of objects with "question" and "answer" fields.

Output ONLY the JSON array, nothing else."""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": api_model,
                    "messages": [
                        {"role": "system", "content": "You are a knowledge generation assistant. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Premium API returned {response.status_code}: {response.text[:200]}",
                )

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON from response (handle markdown code blocks)
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            qa_pairs = json.loads(content)

            if not isinstance(qa_pairs, list):
                raise ValueError("Expected a JSON array")

            # Add topic to each entry
            for pair in qa_pairs:
                pair["topic"] = req.topic
                pair["source"] = "ai_distill"

            return {
                "pairs": qa_pairs,
                "count": len(qa_pairs),
                "topic": req.topic,
                "model_used": api_model,
            }

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Premium API request timed out")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Failed to parse AI response as JSON: {e}")
    except KeyError as e:
        raise HTTPException(status_code=502, detail=f"Unexpected API response format: {e}")


@app.post("/admin/distill/save")
async def admin_distill_save(
    req: KnowledgeBatchRequest,
    _user: str = Depends(_verify_admin_token),
):
    count = rag_engine.add_entries_batch(req.entries)
    return {"saved": count}


# ═══════════════════════════════════════════════════════════
# Admin: Web Scraper
# ═══════════════════════════════════════════════════════════

@app.post("/admin/scrape")
async def admin_scrape(
    req: ScrapeRequest,
    _user: str = Depends(_verify_admin_token),
):
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; NeuroLite/1.0; Knowledge Bot)",
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Website request timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Website returned {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {e}")

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {content_type}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    # Extract text from specific selector or full body
    if req.selector:
        target = soup.select(req.selector)
        text_parts = [el.get_text(separator="\n", strip=True) for el in target]
        text = "\n\n".join(text_parts)
    else:
        # Get main content area
        main = soup.find("main") or soup.find("article") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else ""

    # Clean up excessive whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    # Truncate very long content
    if len(text) > 50000:
        text = text[:50000] + "\n\n[Content truncated at 50000 characters]"

    return {
        "title": title,
        "url": url,
        "text": text,
        "length": len(text),
    }


# ═══════════════════════════════════════════════════════════
# Admin: Model Management
# ═══════════════════════════════════════════════════════════

@app.get("/admin/models")
async def admin_models_list(_user: str = Depends(_verify_admin_token)):
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for f in models_dir.glob("*.gguf"):
        downloaded[f.name] = {
            "filename": f.name,
            "size_mb": round(f.stat().st_size / 1048576, 1),
            "active": f.name == (llm_engine.model_name or ""),
        }

    catalog = []
    for m in AVAILABLE_MODELS:
        entry = dict(m)
        entry["downloaded"] = m["filename"] in downloaded
        entry["active"] = downloaded.get(m["filename"], {}).get("active", False)
        if entry["downloaded"]:
            entry["local_size_mb"] = downloaded[m["filename"]]["size_mb"]
        catalog.append(entry)

    # Add any locally downloaded models not in catalog
    for fname, info in downloaded.items():
        if not any(m["filename"] == fname for m in AVAILABLE_MODELS):
            catalog.append({
                "id": fname,
                "name": fname,
                "filename": fname,
                "url": "",
                "size_gb": round(info["size_mb"] / 1024, 1),
                "description": "Manually added model",
                "recommended": False,
                "downloaded": True,
                "active": info["active"],
                "local_size_mb": info["size_mb"],
            })

    return {
        "models": catalog,
        "active_model": llm_engine.model_name,
        "downloads_in_progress": {
            k: {"progress": v["progress"], "filename": v["filename"]}
            for k, v in _download_progress.items()
            if v.get("status") == "downloading"
        },
    }


@app.post("/admin/models/download")
async def admin_model_download(
    req: ModelDownloadRequest,
    _user: str = Depends(_verify_admin_token),
):
    model_info = None
    for m in AVAILABLE_MODELS:
        if m["id"] == req.model_id:
            model_info = m
            break

    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found in catalog")

    dest = BASE_DIR / "models" / model_info["filename"]
    if dest.exists():
        return {"status": "already_downloaded", "filename": model_info["filename"]}

    download_id = str(uuid.uuid4())[:8]
    _download_progress[download_id] = {
        "progress": 0,
        "status": "downloading",
        "filename": model_info["filename"],
        "error": None,
    }

    async def _do_download():
        try:
            async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
                async with client.stream("GET", model_info["url"]) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))
                    downloaded = 0
                    tmp_path = dest.with_suffix(".tmp")

                    with open(tmp_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=1048576):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                _download_progress[download_id]["progress"] = round(
                                    (downloaded / total) * 100, 1
                                )

                    tmp_path.rename(dest)
                    _download_progress[download_id]["status"] = "complete"
                    _download_progress[download_id]["progress"] = 100
                    logger.info("Model downloaded: %s", model_info["filename"])

        except Exception as e:
            _download_progress[download_id]["status"] = "error"
            _download_progress[download_id]["error"] = str(e)
            logger.error("Model download failed: %s", e)
            tmp_path = dest.with_suffix(".tmp")
            if tmp_path.exists():
                tmp_path.unlink()

    asyncio.create_task(_do_download())
    return {"download_id": download_id, "filename": model_info["filename"]}


@app.get("/admin/models/download-progress/{download_id}")
async def admin_download_progress(
    download_id: str,
    _user: str = Depends(_verify_admin_token),
):
    progress = _download_progress.get(download_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Download not found")
    return progress


@app.post("/admin/models/switch")
async def admin_model_switch(
    req: ModelSwitchRequest,
    _user: str = Depends(_verify_admin_token),
):
    model_file = BASE_DIR / "models" / req.filename
    if not model_file.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    n_ctx = _env_int("CONTEXT_LENGTH", 2048)
    n_threads = _env_int("N_THREADS", 2)
    n_batch = _env_int("N_BATCH", 64)

    loop = asyncio.get_event_loop()

    def _switch():
        return llm_engine.load_model(
            model_path=str(model_file),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
        )

    success = await loop.run_in_executor(_executor, _switch)

    if success:
        # Update config.env
        _update_config_file("MODEL_PATH", f"models/{req.filename}")
        return {"status": "switched", "model": req.filename}
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {llm_engine.load_error}",
        )


# ═══════════════════════════════════════════════════════════
# Admin: Settings / Config
# ═══════════════════════════════════════════════════════════

SAFE_CONFIG_KEYS = {
    "PREMIUM_API_KEY", "PREMIUM_API_URL", "PREMIUM_MODEL",
    "TEMPERATURE", "TOP_P", "TOP_K", "REPEAT_PENALTY",
    "MAX_TOKENS", "CONTEXT_LENGTH", "N_THREADS", "N_BATCH",
    "SYSTEM_PROMPT", "ADMIN_PASSWORD", "LOG_LEVEL",
}


@app.get("/admin/config")
async def admin_config_get(_user: str = Depends(_verify_admin_token)):
    config = {}
    for key in SAFE_CONFIG_KEYS:
        val = _env(key, "")
        # Mask sensitive values
        if key == "PREMIUM_API_KEY" and val:
            config[key] = val[:8] + "..." + val[-4:] if len(val) > 12 else "***"
        elif key == "ADMIN_PASSWORD":
            config[key] = "********"
        else:
            config[key] = val
    return {"config": config}


@app.post("/admin/config")
async def admin_config_update(
    req: ConfigUpdateRequest,
    _user: str = Depends(_verify_admin_token),
):
    if req.key not in SAFE_CONFIG_KEYS:
        raise HTTPException(status_code=400, detail=f"Key '{req.key}' is not configurable")

    os.environ[req.key] = req.value
    _update_config_file(req.key, req.value)
    return {"status": "updated", "key": req.key}


def _update_config_file(key: str, value: str):
    """Persist config change to config.env."""
    config_path = BASE_DIR / "config.env"
    if not config_path.exists():
        return

    lines = config_path.read_text().splitlines()
    found = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            lines[i] = f"{key}={value}"
            found = True
            break

    if not found:
        lines.append(f"{key}={value}")

    config_path.write_text("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════
# Admin: Rebuild FTS Index
# ═══════════════════════════════════════════════════════════

@app.post("/admin/knowledge/rebuild-index")
async def admin_rebuild_fts(_user: str = Depends(_verify_admin_token)):
    try:
        rag_engine.rebuild_fts()
        return {"status": "rebuilt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# Admin: Test Chat (from admin panel)
# ═══════════════════════════════════════════════════════════

@app.post("/admin/test-chat")
async def admin_test_chat(
    req: ChatRequest,
    _user: str = Depends(_verify_admin_token),
):
    """Non-streaming chat for admin testing."""
    if not llm_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    emotion, modifier, confidence = classify_emotion(req.message)
    rag_results = rag_engine.search(req.message, limit=2)
    rag_ctx = ""
    if rag_results:
        parts = [f"Q: {r.question}\nA: {r.answer}" for r in rag_results]
        rag_ctx = "\n---\n".join(parts)

    system_prompt = _env("SYSTEM_PROMPT", "You are a helpful AI assistant.")
    messages = [{"role": "system", "content": system_prompt}]
    if modifier:
        messages[0]["content"] += f"\n\n{modifier}"
    if rag_ctx:
        messages[0]["content"] += f"\n\n[Reference]\n{rag_ctx}"
    messages.append({"role": "user", "content": req.message})

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        _executor,
        lambda: llm_engine.generate(
            messages=messages,
            max_tokens=_env_int("MAX_TOKENS", 512),
            temperature=_env_float("TEMPERATURE", 0.7),
        ),
    )

    processed = process_response(response, emotion, confidence)

    return {
        "response": processed,
        "emotion": emotion,
        "confidence": round(confidence, 2),
        "rag_results": [
            {"question": r.question, "answer": r.answer[:200]} for r in rag_results
        ],
    }


# ═══════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════

def _handle_signal(sig, frame):
    logger.info("Received signal %s, shutting down...", sig)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    host = _env("NEURO_LITE_HOST", "0.0.0.0")
    port = _env_int("NEURO_LITE_PORT", 8080)

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        log_level=_env("LOG_LEVEL", "info").lower(),
        access_log=False,
        timeout_keep_alive=65,
    )

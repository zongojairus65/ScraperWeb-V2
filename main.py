import os
import asyncio
import csv
import functools
import io
import json
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List

import aiosqlite
import requests as http_requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from mistralai import Mistral
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

DB_PATH = "scraper.db"
executor = ThreadPoolExecutor(max_workers=5)
scheduler = AsyncIOScheduler()

# Fix #2 — Mistral client instantiated once at startup, not per request.
_mistral_client: Mistral | None = None

def get_mistral_client() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_key:
            raise ValueError("Clé API Mistral manquante. Configurez MISTRAL_API_KEY.")
        _mistral_client = Mistral(api_key=mistral_key)
    return _mistral_client

# ─── Base de données SQLite (async) ───────────────────────────────────────────

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS scrape_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                prompt TEXT NOT NULL,
                result TEXT,
                status TEXT DEFAULT 'success',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                prompt TEXT NOT NULL,
                cron TEXT NOT NULL,
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await db.commit()

async def save_to_history(url: str, prompt: str, result: str, status: str = "success") -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO scrape_history (url, prompt, result, status) VALUES (?, ?, ?, ?)",
            (url, prompt, json.dumps(result), status)
        )
        row_id = cursor.lastrowid
        await db.commit()
    return row_id

# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    get_mistral_client()  # Valide la clé API dès le démarrage
    scheduler.start()
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, name, url, prompt, cron FROM scheduled_jobs WHERE active=1"
        ) as cursor:
            jobs = await cursor.fetchall()
    for job in jobs:
        job_id, name, url, prompt, cron = job
        scheduler.add_job(
            scheduled_scrape,
            CronTrigger.from_crontab(cron),
            args=[url, prompt],
            id=f"job_{job_id}",
            name=name,
            replace_existing=True
        )
    yield
    # Shutdown
    scheduler.shutdown(wait=False)

# ─── App ──────────────────────────────────────────────────────────────────────

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
API_KEY = os.getenv("API_KEY")  # Optionnel : protège l'API si défini dans .env

app = FastAPI(
    title="ScraperWeb V2 API",
    description="API de web scraping IA — Mistral + SQLite + Export + Planification",
    version="2.0.0",
    lifespan=lifespan,
)

# Fix #8 — CORS enregistré en premier pour s'exécuter en couche externe.
# Les requêtes OPTIONS (preflight) reçoivent ainsi les headers CORS
# avant que le middleware d'auth ne les intercepte.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Auth middleware ───────────────────────────────────────────────────────────

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if API_KEY:
        public_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
        # Fix #8 — OPTIONS exempté pour laisser passer les preflight CORS
        if request.method != "OPTIONS" and request.url.path not in public_paths:
            key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
            if key != API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Clé API invalide ou manquante. Fournissez X-API-Key dans les headers."}
                )
    return await call_next(request)

# ─── Scraping ─────────────────────────────────────────────────────────────────

def fetch_page_content(url: str) -> str:
    """Récupère le contenu HTML brut via requests."""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ScraperWebBot/2.0)"}
    response = http_requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)[:12000]  # Limiter tokens

def run_mistral(content: str, prompt: str) -> str:
    """Envoie le contenu à Mistral pour analyse."""
    # Fix #2 — réutilise le client partagé
    client = get_mistral_client()
    messages = [
        {
            "role": "user",
            "content": f"""Voici le contenu d'une page web :

{content}

---
Question : {prompt}

Réponds de manière précise et structurée en te basant uniquement sur le contenu ci-dessus."""
        }
    ]
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages
    )
    return response.choices[0].message.content

def scrape_single(url: str, prompt: str) -> dict:
    """Scrape une URL et retourne le résultat."""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    content = fetch_page_content(url)
    answer = run_mistral(content, prompt)
    return {"url": url, "prompt": prompt, "answer": answer}

# ─── Modèles Pydantic ──────────────────────────────────────────────────────────

class BulkScrapeRequest(BaseModel):
    urls: List[str]
    prompt: str

class ScheduleRequest(BaseModel):
    name: str
    url: str
    prompt: str
    cron: str  # ex: "0 8 * * *" = tous les jours à 8h

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "ScraperWeb V2 — Powered by Mistral AI",
        "endpoints": {
            "scrape": "GET /scrape?url=...&prompt=...",
            "bulk": "POST /scrape/bulk",
            "history": "GET /history",
            "export": "GET /export/{id}?format=json|csv|pdf",
            "schedule": "POST /schedule",
            "jobs": "GET /schedule",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ScraperWeb V2", "model": "mistral-large-latest"}

# ── Scraping simple ──

@app.get("/scrape")
async def scrape(
    url: str = Query(..., description="URL à scraper"),
    prompt: str = Query(..., description="Question sur la page")
):
    try:
        # Fix #1 — get_running_loop() au lieu de get_event_loop()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, scrape_single, url, prompt)
        history_id = await save_to_history(url, prompt, result["answer"])
        return {"status": "success", "id": history_id, **result}
    except Exception as e:
        await save_to_history(url, prompt, str(e), status="error")
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

# ── Scraping en masse ──

@app.post("/scrape/bulk")
async def scrape_bulk(request: BulkScrapeRequest):
    results = []
    errors = []
    # Fix #1 — get_running_loop() au lieu de get_event_loop()
    loop = asyncio.get_running_loop()

    async def scrape_one(url):
        try:
            result = await loop.run_in_executor(executor, scrape_single, url, request.prompt)
            hid = await save_to_history(url, request.prompt, result["answer"])
            results.append({"id": hid, **result})
        except Exception as e:
            errors.append({"url": url, "error": str(e)})
            await save_to_history(url, request.prompt, str(e), status="error")

    await asyncio.gather(*[scrape_one(url) for url in request.urls])
    return {
        "status": "success",
        "total": len(request.urls),
        "succeeded": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }

# ── Historique ──

@app.get("/history")
async def get_history(limit: int = 50, offset: int = 0):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, url, prompt, status, created_at FROM scrape_history ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ) as cursor:
            rows = await cursor.fetchall()
        async with db.execute("SELECT COUNT(*) FROM scrape_history") as cursor:
            total = (await cursor.fetchone())[0]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [
            {"id": r[0], "url": r[1], "prompt": r[2], "status": r[3], "created_at": r[4]}
            for r in rows
        ]
    }

@app.get("/history/{item_id}")
async def get_history_item(item_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT * FROM scrape_history WHERE id=?", (item_id,)) as cursor:
            row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Entrée introuvable")
    return {
        "id": row[0], "url": row[1], "prompt": row[2],
        "result": json.loads(row[3]) if row[3] else None,
        "status": row[4], "created_at": row[5]
    }

# ── Export ──

@app.get("/export/{item_id}")
# Fix #4 — `fmt` en interne, alias="format" pour préserver l'API publique
async def export_result(item_id: int, fmt: str = Query("json", alias="format", description="json | csv | pdf")):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT * FROM scrape_history WHERE id=?", (item_id,)) as cursor:
            row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Entrée introuvable")

    data = {
        "id": row[0], "url": row[1], "prompt": row[2],
        "result": json.loads(row[3]) if row[3] else "",
        "status": row[4], "created_at": row[5]
    }

    if fmt == "json":
        content = json.dumps(data, ensure_ascii=False, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=scrape_{item_id}.json"}
        )

    elif fmt == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=scrape_{item_id}.csv"}
        )

    elif fmt == "pdf":
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"ScraperWeb V2 — Export #{item_id}", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"<b>URL :</b> {data['url']}", styles["Normal"]),
            Spacer(1, 6),
            Paragraph(f"<b>Prompt :</b> {data['prompt']}", styles["Normal"]),
            Spacer(1, 6),
            Paragraph(f"<b>Date :</b> {data['created_at']}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("<b>Résultat :</b>", styles["Heading2"]),
            Spacer(1, 6),
            Paragraph(str(data["result"]).replace("\n", "<br/>"), styles["Normal"]),
        ]
        # Fix #6 — doc.build() est synchrone/CPU-bound, on l'offload à l'executor
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, functools.partial(doc.build, story))
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=scrape_{item_id}.pdf"}
        )

    raise HTTPException(status_code=400, detail="Format invalide. Utilise : json, csv ou pdf")

# ── Planification ──

async def scheduled_scrape(url: str, prompt: str):
    try:
        # Fix #1 — get_running_loop() au lieu de get_event_loop()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, scrape_single, url, prompt)
        await save_to_history(url, prompt, result["answer"])
    except Exception as e:
        await save_to_history(url, prompt, str(e), status="error")

@app.post("/schedule")
async def create_schedule(request: ScheduleRequest):
    # Fix #7 — validation du cron avant l'insertion en DB ; 400 propre si invalide
    try:
        trigger = CronTrigger.from_crontab(request.cron)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Expression cron invalide : {e}")

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO scheduled_jobs (name, url, prompt, cron) VALUES (?, ?, ?, ?)",
            (request.name, request.url, request.prompt, request.cron)
        )
        job_id = cursor.lastrowid
        await db.commit()

    scheduler.add_job(
        scheduled_scrape,
        trigger,  # réutilise l'objet déjà validé
        args=[request.url, request.prompt],
        id=f"job_{job_id}",
        name=request.name,
        replace_existing=True
    )

    return {
        "status": "scheduled",
        "id": job_id,
        "name": request.name,
        "url": request.url,
        "prompt": request.prompt,
        "cron": request.cron
    }

@app.get("/schedule")
async def list_schedules():
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, name, url, prompt, cron, active, created_at FROM scheduled_jobs"
        ) as cursor:
            rows = await cursor.fetchall()
    return {
        "jobs": [
            {"id": r[0], "name": r[1], "url": r[2], "prompt": r[3],
             "cron": r[4], "active": bool(r[5]), "created_at": r[6]}
            for r in rows
        ]
    }

@app.delete("/schedule/{job_id}")
async def delete_schedule(job_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE scheduled_jobs SET active=0 WHERE id=?", (job_id,))
        await db.commit()
    try:
        scheduler.remove_job(f"job_{job_id}")
    except Exception:
        pass
    return {"status": "deleted", "id": job_id}

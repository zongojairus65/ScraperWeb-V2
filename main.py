import os
import asyncio
import csv
import io
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional

import requests as http_requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mistralai import Mistral
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv

load_dotenv()

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ScraperWeb V2 API",
    description="API de web scraping IA — Mistral + SQLite + Export + Planification",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=5)
scheduler = AsyncIOScheduler()

# ─── Base de données SQLite ────────────────────────────────────────────────────

DB_PATH = "scraper.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS scrape_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            prompt TEXT NOT NULL,
            result TEXT,
            status TEXT DEFAULT 'success',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
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
    conn.commit()
    conn.close()

def save_to_history(url: str, prompt: str, result: str, status: str = "success") -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO scrape_history (url, prompt, result, status) VALUES (?, ?, ?, ?)",
        (url, prompt, json.dumps(result), status)
    )
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id

# ─── Scraping ─────────────────────────────────────────────────────────────────

def fetch_page_content(url: str) -> str:
    """Récupère le contenu HTML brut via requests."""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ScraperWebBot/2.0)"}
    response = http_requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    # Supprimer scripts/styles
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)[:12000]  # Limiter tokens

def run_mistral(content: str, prompt: str) -> str:
    """Envoie le contenu à Mistral pour analyse."""
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise ValueError("Clé API Mistral manquante. Configurez MISTRAL_API_KEY.")

    client = Mistral(api_key=mistral_key)
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

@app.on_event("startup")
async def startup_event():
    init_db()
    scheduler.start()
    # Recharger les jobs planifiés depuis la DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, url, prompt, cron FROM scheduled_jobs WHERE active=1")
    jobs = c.fetchall()
    conn.close()
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
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, scrape_single, url, prompt)
        history_id = save_to_history(url, prompt, result["answer"])
        return {"status": "success", "id": history_id, **result}
    except Exception as e:
        save_to_history(url, prompt, str(e), status="error")
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

# ── Scraping en masse ──

@app.post("/scrape/bulk")
async def scrape_bulk(request: BulkScrapeRequest):
    results = []
    errors = []
    loop = asyncio.get_event_loop()

    async def scrape_one(url):
        try:
            result = await loop.run_in_executor(executor, scrape_single, url, request.prompt)
            hid = save_to_history(url, request.prompt, result["answer"])
            results.append({"id": hid, **result})
        except Exception as e:
            errors.append({"url": url, "error": str(e)})
            save_to_history(url, request.prompt, str(e), status="error")

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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, url, prompt, status, created_at FROM scrape_history ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    )
    rows = c.fetchall()
    c.execute("SELECT COUNT(*) FROM scrape_history")
    total = c.fetchone()[0]
    conn.close()
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM scrape_history WHERE id=?", (item_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Entrée introuvable")
    return {
        "id": row[0], "url": row[1], "prompt": row[2],
        "result": json.loads(row[3]) if row[3] else None,
        "status": row[4], "created_at": row[5]
    }

# ── Export ──

@app.get("/export/{item_id}")
async def export_result(item_id: int, format: str = Query("json", description="json | csv | pdf")):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM scrape_history WHERE id=?", (item_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Entrée introuvable")

    data = {
        "id": row[0], "url": row[1], "prompt": row[2],
        "result": json.loads(row[3]) if row[3] else "",
        "status": row[4], "created_at": row[5]
    }

    if format == "json":
        content = json.dumps(data, ensure_ascii=False, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=scrape_{item_id}.json"}
        )

    elif format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=scrape_{item_id}.csv"}
        )

    elif format == "pdf":
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
        doc.build(story)
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
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, scrape_single, url, prompt)
        save_to_history(url, prompt, result["answer"])
    except Exception as e:
        save_to_history(url, prompt, str(e), status="error")

@app.post("/schedule")
async def create_schedule(request: ScheduleRequest):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO scheduled_jobs (name, url, prompt, cron) VALUES (?, ?, ?, ?)",
        (request.name, request.url, request.prompt, request.cron)
    )
    job_id = c.lastrowid
    conn.commit()
    conn.close()

    scheduler.add_job(
        scheduled_scrape,
        CronTrigger.from_crontab(request.cron),
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, url, prompt, cron, active, created_at FROM scheduled_jobs")
    rows = c.fetchall()
    conn.close()
    return {
        "jobs": [
            {"id": r[0], "name": r[1], "url": r[2], "prompt": r[3],
             "cron": r[4], "active": bool(r[5]), "created_at": r[6]}
            for r in rows
        ]
    }

@app.delete("/schedule/{job_id}")
async def delete_schedule(job_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE scheduled_jobs SET active=0 WHERE id=?", (job_id,))
    conn.commit()
    conn.close()
    try:
        scheduler.remove_job(f"job_{job_id}")
    except Exception:
        pass
    return {"status": "deleted", "id": job_id}

import os
import asyncio
import base64
import csv
import functools
import io
import json
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal

import aiosqlite
import requests as http_requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from mistralai import Mistral
from playwright.async_api import async_playwright
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
                mode TEXT DEFAULT 'simple',
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
                mode TEXT DEFAULT 'simple',
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # Migration retrocompatible : ajout colonne mode si absente
        for table in ("scrape_history", "scheduled_jobs"):
            try:
                await db.execute(f"ALTER TABLE {table} ADD COLUMN mode TEXT DEFAULT 'simple'")
            except Exception:
                pass
        await db.commit()

async def save_to_history(
    url: str, prompt: str, result: str,
    status: str = "success", mode: str = "simple"
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO scrape_history (url, prompt, result, status, mode) VALUES (?, ?, ?, ?, ?)",
            (url, prompt, json.dumps(result), status, mode)
        )
        row_id = cursor.lastrowid
        await db.commit()
    return row_id

# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    get_mistral_client()
    scheduler.start()
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, name, url, prompt, cron, mode FROM scheduled_jobs WHERE active=1"
        ) as cursor:
            jobs = await cursor.fetchall()
    for job in jobs:
        job_id, name, url, prompt, cron, mode = job
        scheduler.add_job(
            scheduled_scrape,
            CronTrigger.from_crontab(cron),
            args=[url, prompt, mode or "simple"],
            id=f"job_{job_id}",
            name=name,
            replace_existing=True
        )
    yield
    scheduler.shutdown(wait=False)

# ─── App ──────────────────────────────────────────────────────────────────────

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
API_KEY = os.getenv("API_KEY")

app = FastAPI(
    title="ScraperWeb V2 API",
    description="API de web scraping IA — Mistral + SQLite + Export + Planification + Agent navigateur",
    version="2.2.0",
    lifespan=lifespan,
)

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
        if request.method != "OPTIONS" and request.url.path not in public_paths:
            key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
            if key != API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Clé API invalide ou manquante. Fournissez X-API-Key dans les headers."}
                )
    return await call_next(request)

# ─── Scraping HTTP — mode simple (inchange) ───────────────────────────────────

def fetch_page_content(url: str) -> str:
    """Recupere le contenu HTML brut via requests (synchrone, executor)."""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ScraperWebBot/2.0)"}
    response = http_requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)[:12000]

# ─── Scraping navigateur — mode browser (URL connue) ─────────────────────────

async def fetch_page_browser(url: str) -> str:
    """Recupere le contenu d'une URL precise via Playwright headless."""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        await page.goto(url, wait_until="networkidle", timeout=30000)
        for sel in [
            "button:has-text('Accept all')", "button:has-text('Tout accepter')",
            "button:has-text('Accept')", "button:has-text('Accepter')",
        ]:
            try:
                await page.click(sel, timeout=1500)
                break
            except Exception:
                pass
        await page.wait_for_load_state("networkidle")
        content = await page.content()
        await browser.close()
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)[:12000]

# ─── Mistral analyse texte (inchange) ─────────────────────────────────────────

def run_mistral(content: str, prompt: str) -> str:
    """Envoie le contenu a Mistral pour analyse (synchrone, executor)."""
    client = get_mistral_client()
    messages = [{
        "role": "user",
        "content": f"""Voici le contenu d'une page web :

{content}

---
Question : {prompt}

Reponds de maniere precise et structuree en te basant uniquement sur le contenu ci-dessus."""
    }]
    response = client.chat.complete(model="mistral-large-latest", messages=messages)
    return response.choices[0].message.content

# ─── Scraping unifie simple + browser ────────────────────────────────────────

async def scrape_single(url: str, prompt: str, mode: str = "simple") -> dict:
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    loop = asyncio.get_running_loop()
    if mode == "browser":
        content = await fetch_page_browser(url)
    else:
        content = await loop.run_in_executor(executor, fetch_page_content, url)
    answer = await loop.run_in_executor(executor, run_mistral, content, prompt)
    return {"url": url, "prompt": prompt, "answer": answer, "mode": mode}

# ─── Agent navigateur autonome (nouveau) ──────────────────────────────────────
#
# Boucle :  screenshot -> Pixtral decide -> Playwright execute -> recommence
# Arret :   action "extract" ou max_steps atteint
# Modele :  pixtral-large-latest (vision) pour les decisions
#           mistral-large-latest pour l'analyse finale du contenu
# ─────────────────────────────────────────────────────────────────────────────

AGENT_DECISION_PROMPT = """Tu es un agent navigateur web expert.

Objectif : {intent}
URL actuelle : {url}
Etape {step} sur {max_steps} maximum
Historique des actions : {history}

Regarde le screenshot et decide de la prochaine action pour atteindre l'objectif.
Reponds UNIQUEMENT avec un objet JSON valide, sans markdown ni explication.

Actions disponibles :
{{"type": "click", "text": "texte visible de l'element a cliquer"}}
{{"type": "type", "text": "texte a saisir dans le champ actif"}}
{{"type": "scroll"}}
{{"type": "extract"}}

Regles :
- Clique sur un champ de saisie AVANT d'utiliser "type"
- Utilise "extract" des que les donnees demandees sont clairement visibles
- Si une popup bloque la vue, clique sur Accepter ou Fermer
- Ne repete pas la meme action sans raison"""


def decide_action_with_mistral(
    screenshot_b64: str,
    intent: str,
    url: str,
    step: int,
    max_steps: int,
    history: list
) -> dict:
    """
    Appelle Pixtral avec le screenshot courant pour obtenir la prochaine action.
    Synchrone — exécuté dans le thread executor.
    """
    client = get_mistral_client()
    prompt = AGENT_DECISION_PROMPT.format(
        intent=intent,
        url=url,
        step=step,
        max_steps=max_steps,
        history=json.dumps(history, ensure_ascii=False) if history else "aucune action encore"
    )
    response = client.chat.complete(
        model="pixtral-large-latest",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}
                },
                {"type": "text", "text": prompt}
            ]
        }]
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


async def execute_action(page, action: dict) -> str:
    """
    Execute une action Playwright.
    Tente plusieurs strategies pour le clic afin d'etre robuste
    face a des structures HTML variees selon les sites.
    """
    action_type = action.get("type")

    if action_type == "click":
        text = action.get("text", "")
        try:
            await page.get_by_text(text, exact=False).first.click(timeout=4000)
        except Exception:
            try:
                await page.get_by_placeholder(text, exact=False).first.click(timeout=3000)
            except Exception:
                await page.locator(f"text={text}").first.click(timeout=3000)
        try:
            await page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        return f"Clique sur '{text}'"

    elif action_type == "type":
        text = action.get("text", "")
        await page.keyboard.type(text, delay=50)  # delai humain anti-detection
        await page.keyboard.press("Enter")
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        return f"Saisi '{text}' + Entree"

    elif action_type == "scroll":
        await page.evaluate("window.scrollBy(0, 600)")
        await asyncio.sleep(1.5)
        return "Scroll de 600px vers le bas"

    return f"Action non reconnue : {action_type}"


async def run_browser_agent(intent: str, source: str, max_steps: int = 12) -> dict:
    """
    Agent navigateur autonome piloté par Mistral Vision.

    Prend en entree une intention en langage naturel et un site de depart.
    Navigue de lui-meme, etape par etape, jusqu'a trouver et extraire le contenu.
    Aucune URL cible ni ID numerique requis.
    """
    base_url = source if source.startswith("http") else f"https://{source}"
    loop = asyncio.get_running_loop()
    history = []    # contexte JSON pour Pixtral
    steps_log = []  # journal lisible pour la reponse API

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800}
        )
        page = await context.new_page()
        await page.goto(base_url, wait_until="networkidle", timeout=30000)

        # Gestion popup cookies au demarrage
        for sel in [
            "button:has-text('Accept all')", "button:has-text('Tout accepter')",
            "button:has-text('Accept')", "button:has-text('Accepter')",
        ]:
            try:
                await page.click(sel, timeout=1500)
                break
            except Exception:
                pass

        for step in range(1, max_steps + 1):

            # 1. Screenshot de l'etat actuel
            screenshot_bytes = await page.screenshot(type="jpeg", quality=75, full_page=False)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
            current_url = page.url

            # 2. Pixtral decide la prochaine action
            try:
                action = await loop.run_in_executor(
                    executor,
                    decide_action_with_mistral,
                    screenshot_b64, intent, current_url, step, max_steps, history
                )
            except Exception as e:
                steps_log.append({"step": step, "url": current_url, "error": f"Decision Mistral echouee : {e}"})
                break

            log_entry = {"step": step, "action": action, "url": current_url}

            # 3. Extract = contenu pret, on arrete
            if action.get("type") == "extract":
                log_entry["result"] = "Contenu extrait"
                steps_log.append(log_entry)
                content = await page.content()
                await browser.close()
                soup = BeautifulSoup(content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                return {
                    "success": True,
                    "final_url": current_url,
                    "steps": steps_log,
                    "content": soup.get_text(separator="\n", strip=True)[:12000]
                }

            # 4. Executer l'action
            try:
                description = await execute_action(page, action)
                log_entry["result"] = description
            except Exception as e:
                log_entry["error"] = str(e)
                # On continue : Pixtral verra l'etat reel au prochain screenshot

            steps_log.append(log_entry)
            history.append({"step": step, **action})

        # Max steps atteint — extraire ce qu'on a quand meme
        final_url = page.url
        content = await page.content()
        await browser.close()

    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return {
        "success": False,
        "reason": f"Nombre maximum d'etapes ({max_steps}) atteint",
        "final_url": final_url,
        "steps": steps_log,
        "content": soup.get_text(separator="\n", strip=True)[:12000]
    }

# ─── Modeles Pydantic ──────────────────────────────────────────────────────────

class BulkScrapeRequest(BaseModel):
    urls: List[str]
    prompt: str
    mode: Literal["simple", "browser"] = "simple"

class ScheduleRequest(BaseModel):
    name: str
    url: str
    prompt: str
    cron: str
    mode: Literal["simple", "browser"] = "simple"

class AgentSearchRequest(BaseModel):
    intent: str      # intention en langage naturel ("Stats PSG vs Barca 7 mai 2025")
    source: str      # site de depart ("sofascore.com")
    prompt: str      # question Mistral sur le contenu extrait
    max_steps: int = 12

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "ScraperWeb V2 — Powered by Mistral AI",
        "endpoints": {
            "scrape":  "GET  /scrape?url=...&prompt=...&mode=simple|browser",
            "bulk":    "POST /scrape/bulk",
            "agent":   "POST /agent/search",
            "history": "GET  /history",
            "export":  "GET  /export/{id}?format=json|csv|pdf",
            "schedule":"POST /schedule",
            "jobs":    "GET  /schedule",
            "health":  "GET  /health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ScraperWeb V2", "model": "mistral-large-latest"}

# ── Scraping simple ──

@app.get("/scrape")
async def scrape(
    url: str = Query(..., description="URL a scraper"),
    prompt: str = Query(..., description="Question sur la page"),
    mode: Literal["simple", "browser"] = Query(
        "simple", description="simple = HTTP | browser = Playwright URL connue"
    )
):
    try:
        result = await scrape_single(url, prompt, mode)
        history_id = await save_to_history(url, prompt, result["answer"], mode=mode)
        return {"status": "success", "id": history_id, **result}
    except Exception as e:
        await save_to_history(url, prompt, str(e), status="error", mode=mode)
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

# ── Scraping en masse ──

@app.post("/scrape/bulk")
async def scrape_bulk(request: BulkScrapeRequest):
    results = []
    errors = []

    async def scrape_one(url):
        try:
            result = await scrape_single(url, request.prompt, request.mode)
            hid = await save_to_history(url, request.prompt, result["answer"], mode=request.mode)
            results.append({"id": hid, **result})
        except Exception as e:
            errors.append({"url": url, "error": str(e)})
            await save_to_history(url, request.prompt, str(e), status="error", mode=request.mode)

    await asyncio.gather(*[scrape_one(url) for url in request.urls])
    return {
        "status": "success",
        "total": len(request.urls),
        "succeeded": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }

# ── Agent navigateur autonome ──

@app.post("/agent/search")
async def agent_search(request: AgentSearchRequest):
    """
    Lance un agent navigateur autonome pilote par Mistral Vision (Pixtral).

    L'agent navigue de lui-meme sur le site source pour trouver le contenu
    correspondant a l'intention, sans URL ni ID a fournir.
    Mistral analyse ensuite le contenu extrait et repond a la question.

    Exemple :
    {
      "intent": "Stats du match PSG vs Barcelone du 7 mai 2025",
      "source": "sofascore.com",
      "prompt": "Extrais possession, xG et tirs cadres pour chaque equipe",
      "max_steps": 12
    }
    """
    try:
        agent_result = await run_browser_agent(request.intent, request.source, request.max_steps)

        if not agent_result.get("content"):
            raise HTTPException(status_code=500, detail="L'agent n'a pas pu extraire de contenu")

        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(
            executor, run_mistral, agent_result["content"], request.prompt
        )

        history_id = await save_to_history(
            agent_result["final_url"], request.prompt, answer, mode="agent"
        )

        return {
            "status": "success",
            "id": history_id,
            "intent": request.intent,
            "prompt": request.prompt,
            "answer": answer,
            "mode": "agent",
            "agent_success": agent_result["success"],
            "final_url": agent_result["final_url"],
            "steps": agent_result["steps"]
        }

    except HTTPException:
        raise
    except Exception as e:
        await save_to_history(request.source, request.prompt, str(e), status="error", mode="agent")
        raise HTTPException(status_code=500, detail=f"Erreur agent : {str(e)}")

# ── Historique ──

@app.get("/history")
async def get_history(limit: int = 50, offset: int = 0):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, url, prompt, status, mode, created_at FROM scrape_history ORDER BY id DESC LIMIT ? OFFSET ?",
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
            {"id": r[0], "url": r[1], "prompt": r[2], "status": r[3], "mode": r[4], "created_at": r[5]}
            for r in rows
        ]
    }

@app.get("/history/{item_id}")
async def get_history_item(item_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT * FROM scrape_history WHERE id=?", (item_id,)) as cursor:
            row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Entree introuvable")
    return {
        "id": row[0], "url": row[1], "prompt": row[2],
        "result": json.loads(row[3]) if row[3] else None,
        "status": row[4], "created_at": row[5],
        "mode": row[6] if len(row) > 6 else "simple"
    }

# ── Export ──

@app.get("/export/{item_id}")
async def export_result(item_id: int, fmt: str = Query("json", alias="format", description="json | csv | pdf")):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT * FROM scrape_history WHERE id=?", (item_id,)) as cursor:
            row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Entree introuvable")

    data = {
        "id": row[0], "url": row[1], "prompt": row[2],
        "result": json.loads(row[3]) if row[3] else "",
        "status": row[4], "created_at": row[5],
        "mode": row[6] if len(row) > 6 else "simple"
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
            Paragraph(f"<b>Mode :</b> {data['mode']}", styles["Normal"]),
            Spacer(1, 6),
            Paragraph(f"<b>Date :</b> {data['created_at']}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("<b>Resultat :</b>", styles["Heading2"]),
            Spacer(1, 6),
            Paragraph(str(data["result"]).replace("\n", "<br/>"), styles["Normal"]),
        ]
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

async def scheduled_scrape(url: str, prompt: str, mode: str = "simple"):
    try:
        result = await scrape_single(url, prompt, mode)
        await save_to_history(url, prompt, result["answer"], mode=mode)
    except Exception as e:
        await save_to_history(url, prompt, str(e), status="error", mode=mode)

@app.post("/schedule")
async def create_schedule(request: ScheduleRequest):
    try:
        trigger = CronTrigger.from_crontab(request.cron)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Expression cron invalide : {e}")

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO scheduled_jobs (name, url, prompt, cron, mode) VALUES (?, ?, ?, ?, ?)",
            (request.name, request.url, request.prompt, request.cron, request.mode)
        )
        job_id = cursor.lastrowid
        await db.commit()

    scheduler.add_job(
        scheduled_scrape,
        trigger,
        args=[request.url, request.prompt, request.mode],
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
        "cron": request.cron,
        "mode": request.mode
    }

@app.get("/schedule")
async def list_schedules():
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, name, url, prompt, cron, mode, active, created_at FROM scheduled_jobs"
        ) as cursor:
            rows = await cursor.fetchall()
    return {
        "jobs": [
            {"id": r[0], "name": r[1], "url": r[2], "prompt": r[3],
             "cron": r[4], "mode": r[5], "active": bool(r[6]), "created_at": r[7]}
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

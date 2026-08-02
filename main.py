import os
import asyncio
import base64
import csv
import functools
import io
import ipaddress
import json
import socket
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal
from urllib.parse import urlparse
from xml.sax.saxutils import escape as xml_escape

import aiosqlite
import requests as http_requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from google import genai
from google.genai import types as genai_types
from mistralai import Mistral
from playwright.async_api import async_playwright, Route
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

DB_PATH = "scraper.db"
executor = ThreadPoolExecutor(max_workers=8)
scheduler = AsyncIOScheduler()

GEMINI_LITE_MODEL = "models/gemini-flash-lite-latest"
GEMINI_MODEL = "models/gemini-flash-latest"
GEMMA_MODEL = "gemma-4-26b-a4b-it"
MAGISTRAL_MODEL = "magistral-medium-latest"

# ── Réglages perf agent navigateur (OPTIM) ──
# Avant : wait_until="networkidle" partout -> beaucoup de sites (trackers,
# polling live-score, ads) ne deviennent JAMAIS "idle", donc Playwright
# attendait systématiquement le timeout complet à chaque étape.
# Après : "domcontentloaded" + attente courte explicite, nettement plus fiable
# et rapide dans ce contexte (agent multi-étapes).
NAV_TIMEOUT_MS = 15000
POST_NAV_SETTLE_MS = 600
CLICK_STRATEGY_TIMEOUT_MS = 2500
DEFAULT_MAX_STEPS = 8  # avant : 12

_mistral_client: Mistral | None = None
_gemini_client: "genai.Client | None" = None

def get_mistral_client() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_key:
            raise ValueError("Clé API Mistral manquante. Configurez MISTRAL_API_KEY.")
        _mistral_client = Mistral(api_key=mistral_key)
    return _mistral_client

def get_gemini_client() -> "genai.Client":
    global _gemini_client
    if _gemini_client is None:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("Clé API Gemini manquante. Configurez GEMINI_API_KEY.")
        _gemini_client = genai.Client(api_key=gemini_key)
    return _gemini_client

# ─── SSRF protection ───────────────────────────────────────────────────────────

def assert_public_url(url: str) -> None:
    """
    Empeche le scraping d'adresses privees / locales (SSRF).
    Resout le hostname et bloque les plages d'IP privees, loopback, link-local, etc.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Schema d'URL invalide. Utilise http:// ou https://")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL invalide : hostname manquant")
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise ValueError(f"Impossible de resoudre l'hote : {hostname}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified
        ):
            raise ValueError(f"URL refusee : adresse non autorisee ({ip})")

# ─── Base de données SQLite (async, connexion persistante) ────────────────────
# OPTIM : une seule connexion aiosqlite ouverte au demarrage de l'app et
# reutilisee partout, au lieu d'ouvrir/fermer une connexion a chaque requete.
# aiosqlite serialise en interne les requetes sur une connexion via son thread
# dedie, donc c'est sans risque en usage concurrent (bulk scraping, etc.).

async def init_db(db: aiosqlite.Connection):
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("""
        CREATE TABLE IF NOT EXISTS scrape_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            prompt TEXT NOT NULL,
            result TEXT,
            status TEXT DEFAULT 'success',
            mode TEXT DEFAULT 'simple',
            model TEXT,
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

    async def add_column_if_missing(table: str, column: str, coltype: str):
        async with db.execute(f"PRAGMA table_info({table})") as cursor:
            cols = [row[1] for row in await cursor.fetchall()]
        if column not in cols:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")

    await add_column_if_missing("scrape_history", "mode", "TEXT DEFAULT 'simple'")
    await add_column_if_missing("scrape_history", "model", "TEXT")
    await add_column_if_missing("scheduled_jobs", "mode", "TEXT DEFAULT 'simple'")
    await db.commit()

async def save_to_history(
    db: aiosqlite.Connection,
    url: str, prompt: str, result: str,
    status: str = "success", mode: str = "simple", model: str = ""
) -> int:
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)
    cursor = await db.execute(
        "INSERT INTO scrape_history (url, prompt, result, status, mode, model) VALUES (?, ?, ?, ?, ?, ?)",
        (url, prompt, result, status, mode, model)
    )
    row_id = cursor.lastrowid
    await db.commit()
    return row_id

# ─── Lifespan ─────────────────────────────────────────────────────────────────
# OPTIM : le navigateur Chromium et le driver Playwright sont demarres UNE
# SEULE FOIS ici et reutilises pour toutes les requetes (via app.state.browser).
# Avant : chaque requete relancait tout Chromium (cold start ~1-3s a chaque
# fois), en plus d'etre gaspilleur en ressources sous charge.

@asynccontextmanager
async def lifespan(app: FastAPI):
    db = await aiosqlite.connect(DB_PATH)
    app.state.db = db
    await init_db(db)

    get_mistral_client()

    playwright_ctx = await async_playwright().start()
    app.state.playwright = playwright_ctx
    app.state.browser = await playwright_ctx.chromium.launch(
        headless=True,
        args=["--disable-blink-features=AutomationControlled"]
    )

    scheduler.start()
    async with db.execute(
        "SELECT id, name, url, prompt, cron, mode FROM scheduled_jobs WHERE active=1"
    ) as cursor:
        jobs = await cursor.fetchall()
    for job in jobs:
        job_id, name, url, prompt, cron, mode = job
        scheduler.add_job(
            scheduled_scrape,
            CronTrigger.from_crontab(cron),
            args=[app, url, prompt, mode or "simple"],
            id=f"job_{job_id}",
            name=name,
            replace_existing=True
        )

    yield

    scheduler.shutdown(wait=False)
    await app.state.browser.close()
    await playwright_ctx.stop()
    await db.close()

# ─── App ──────────────────────────────────────────────────────────────────────

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
API_KEY = os.getenv("API_KEY")

app = FastAPI(
    title="ScraperWeb V2 API",
    description="API de web scraping IA — Mistral + SQLite + Export + Planification + Agent navigateur",
    version="2.3.0",
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

# ─── Blocage de ressources (OPTIM) ─────────────────────────────────────────────
# Bloquer polices/medias (et images pour le mode "simple", sans vision) allege
# chaque page et reduit le trafic reseau residuel qui retardait les attentes.
# Pour l'agent (vision), on garde les images car Gemini regarde le screenshot.

async def _block_route(route: Route, block_images: bool):
    rtype = route.request.resource_type
    blocked_types = {"media", "font"}
    if block_images:
        blocked_types.add("image")
    if rtype in blocked_types:
        await route.abort()
    else:
        await route.continue_()

async def new_hardened_page(browser, *, block_images: bool, viewport: dict | None = None):
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        viewport=viewport or {"width": 1280, "height": 800}
    )
    page = await context.new_page()
    await page.route("**/*", functools.partial(_block_route, block_images=block_images))
    return context, page

async def dismiss_cookie_banner(page):
    for sel in [
        "button:has-text('Accept all')", "button:has-text('Tout accepter')",
        "button:has-text('Accept')", "button:has-text('Accepter')",
    ]:
        try:
            await page.click(sel, timeout=1200)
            break
        except Exception:
            pass

async def settle(page):
    """Attente courte et bornee, remplace les wait_for_load_state('networkidle')."""
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=4000)
    except Exception:
        pass
    await page.wait_for_timeout(POST_NAV_SETTLE_MS)

# ─── Scraping HTTP — mode simple (inchange) ───────────────────────────────────

def fetch_page_content(url: str) -> str:
    """Recupere le contenu HTML brut via requests (synchrone, executor)."""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    assert_public_url(url)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ScraperWebBot/2.0)"}
    response = http_requests.get(url, headers=headers, timeout=15, allow_redirects=True)
    response.raise_for_status()
    assert_public_url(response.url)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)[:12000]

# ─── Scraping navigateur — mode browser (URL connue) ─────────────────────────
# OPTIM : reutilise le navigateur persistant (app.state.browser), plus de
# chromium.launch() par requete. domcontentloaded + attente courte au lieu
# de networkidle. Images bloquees (pas de vision necessaire ici).

async def fetch_page_browser(app: FastAPI, url: str) -> str:
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    assert_public_url(url)
    browser = app.state.browser
    context, page = await new_hardened_page(browser, block_images=True)
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        await dismiss_cookie_banner(page)
        await settle(page)
        content = await page.content()
    finally:
        await context.close()
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)[:12000]

# ─── Analyse texte LLM (Gemini principal, Magistral fallback) ─────────────────

def extract_text_from_content(content) -> str:
    """
    Normalise le `message.content` retourne par Mistral en chaine de texte.

    Magistral (modeles "raisonneurs") renvoie une liste de chunks
    (TextChunk, ThinkChunk, ...) au lieu d'une simple chaine.
    On ignore le raisonnement (ThinkChunk) et on garde le texte final.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            chunk_type = getattr(chunk, "type", None)
            if chunk_type == "thinking" or chunk_type == "think":
                continue
            text = getattr(chunk, "text", None)
            if text:
                parts.append(text)
            elif isinstance(chunk, dict):
                if chunk.get("type") in ("thinking", "think"):
                    continue
                if "text" in chunk:
                    parts.append(chunk["text"])
        return "\n".join(parts).strip()
    return str(content)


def run_llm(content: str, prompt: str) -> tuple[str, str]:
    """
    Envoie le contenu au LLM pour analyse (synchrone, executor).

    1. Gemini Flash Lite (models/gemini-flash-lite-latest) — modele principal
    2. Gemini Flash (models/gemini-flash-latest) — fallback
    3. Gemma 4 (gemma-4-26b-a4b-it) — modele open-source, via la meme API Gemini
    4. Magistral (Mistral) — fallback final

    Retourne (reponse, nom_du_modele_utilise).
    """
    user_message = f"""Voici le contenu d'une page web :

{content}

---
Question : {prompt}

Reponds de maniere precise et structuree en te basant uniquement sur le contenu ci-dessus."""

    errors = []

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_LITE_MODEL,
            contents=user_message,
        )
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Reponse Gemini Flash Lite vide")
        return text, GEMINI_LITE_MODEL
    except Exception as e:
        errors.append(f"Gemini Flash Lite: {e}")

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
        )
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Reponse Gemini vide")
        return text, GEMINI_MODEL
    except Exception as e:
        errors.append(f"Gemini: {e}")

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMMA_MODEL,
            contents=user_message,
        )
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Reponse Gemma vide")
        return text, GEMMA_MODEL
    except Exception as e:
        errors.append(f"Gemma: {e}")

    try:
        client = get_mistral_client()
        messages = [{"role": "user", "content": user_message}]
        response = client.chat.complete(model=MAGISTRAL_MODEL, messages=messages)
        raw_content = response.choices[0].message.content
        text = extract_text_from_content(raw_content)
        return text, MAGISTRAL_MODEL
    except Exception as e:
        errors.append(f"Magistral: {e}")
        raise RuntimeError("Echec de tous les modeles LLM : " + " | ".join(errors))

# ─── Scraping unifie simple + browser ────────────────────────────────────────

async def scrape_single(app: FastAPI, url: str, prompt: str, mode: str = "simple") -> dict:
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    loop = asyncio.get_running_loop()
    if mode == "browser":
        content = await fetch_page_browser(app, url)
    else:
        content = await loop.run_in_executor(executor, fetch_page_content, url)
    answer, model_used = await loop.run_in_executor(executor, run_llm, content, prompt)
    return {"url": url, "prompt": prompt, "answer": answer, "mode": mode, "model": model_used}

# ─── Agent navigateur autonome ─────────────────────────────────────────────────
#
# Boucle :  screenshot -> Gemini (vision) decide -> Playwright execute -> recommence
# Arret :   action "extract", max_steps atteint, ou 2 echecs consecutifs sans progression
# Modele :  Gemini Flash Lite -> Gemma 4 (open-source) pour les decisions vision
#           Gemini -> Gemma 4 (open-source) -> Magistral pour l'analyse finale
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
- Utilise "extract" UNIQUEMENT si les donnees demandees par l'objectif sont VISIBLEMENT presentes a l'ecran (pas juste un score, mais les details/stats demandes)
- Si l'action precedente est en erreur (timeout/element introuvable), NE RECHOISIS PAS la meme action a l'identique : essaie un texte plus court/different (ex: juste "Argentina" ou "Iceland" au lieu du nom complet du match), ou scroll pour faire apparaitre l'element
- Si une popup bloque la vue, clique sur Accepter ou Fermer
- Ne repete pas la meme action sans raison
- Si apres plusieurs tentatives le clic echoue toujours, essaie de cliquer sur le score ou le statut du match (ex: "FT", "3 - 0") qui est souvent cliquable aussi"""


def decide_action_with_gemini(
    screenshot_b64: str,
    intent: str,
    url: str,
    step: int,
    max_steps: int,
    history: list
) -> dict:
    """
    Appelle Gemini (vision) avec le screenshot courant pour obtenir la prochaine action.
    Essaie Gemini Flash Lite d'abord (le plus rapide), puis Gemma 4 (open source)
    en secours si le premier echoue ou est rate-limite.
    Synchrone — exécuté dans le thread executor.
    """
    client = get_gemini_client()
    prompt = AGENT_DECISION_PROMPT.format(
        intent=intent,
        url=url,
        step=step,
        max_steps=max_steps,
        history=json.dumps(history, ensure_ascii=False) if history else "aucune action encore"
    )
    image_bytes = base64.b64decode(screenshot_b64)
    image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    errors = []
    for model_name in (GEMINI_LITE_MODEL, GEMMA_MODEL):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[image_part, prompt],
            )
            raw = (response.text or "").strip()
            if not raw:
                raise ValueError("Reponse vide")
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            errors.append(f"{model_name}: {e}")

    raise RuntimeError("Echec de la decision agent (vision) : " + " | ".join(errors))


async def _try_click(page, locator_factory, timeout_ms: int) -> None:
    await locator_factory().click(timeout=timeout_ms)


async def execute_action(page, action: dict) -> str:
    """
    Execute une action Playwright.

    OPTIM click : les strategies sont lancees EN CONCURRENCE (asyncio.wait /
    FIRST_COMPLETED) au lieu d'etre essayees en serie. Avant : jusqu'a 5
    strategies x 3-4s de timeout chacune (~16s pire cas). Apres : borne au
    timeout de la strategie la plus lente qui reussit (~2.5s pire cas typique).
    """
    action_type = action.get("type")

    if action_type == "click":
        text = action.get("text", "")
        strategy_factories = [
            lambda: page.get_by_text(text, exact=False).first,
            lambda: page.get_by_role("link", name=text, exact=False).first,
            lambda: page.locator(f"text={text}").first,
        ]

        tasks = [
            asyncio.create_task(_try_click(page, factory, CLICK_STRATEGY_TIMEOUT_MS))
            for factory in strategy_factories
        ]
        succeeded = False
        last_error = None
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=(CLICK_STRATEGY_TIMEOUT_MS / 1000) + 0.5,
                return_when=asyncio.FIRST_COMPLETED
            )
            # Cherche une tache reussie parmi celles deja terminees ; si aucune,
            # laisse une chance aux autres taches encore en vol de finir avant
            # de conclure a un echec total.
            while True:
                for t in done:
                    if t.exception() is None:
                        succeeded = True
                    else:
                        last_error = t.exception()
                if succeeded or not pending:
                    break
                done, pending = await asyncio.wait(
                    pending, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    break
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

        if not succeeded:
            raise last_error or RuntimeError(f"Aucune strategie de clic n'a fonctionne pour '{text}'")

        await settle(page)
        return f"Clique sur '{text}'"

    elif action_type == "type":
        text = action.get("text", "")
        await page.keyboard.type(text, delay=30)
        await page.keyboard.press("Enter")
        await settle(page)
        return f"Saisi '{text}' + Entree"

    elif action_type == "scroll":
        await page.evaluate("window.scrollBy(0, 600)")
        await page.wait_for_timeout(500)
        return "Scroll de 600px vers le bas"

    return f"Action non reconnue : {action_type}"


async def run_browser_agent(app: FastAPI, intent: str, source: str, max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """
    Agent navigateur autonome piloté par Gemini Vision.

    OPTIM :
    - navigateur persistant (app.state.browser), plus de chromium.launch() ici
    - domcontentloaded + attente courte au lieu de networkidle partout
    - arret anticipe si 2 echecs consecutifs sans progression (meme URL)
    """
    base_url = source if source.startswith("http") else f"https://{source}"
    loop = asyncio.get_running_loop()
    history = []
    steps_log = []
    consecutive_failures = 0

    browser = app.state.browser
    context, page = await new_hardened_page(browser, block_images=False, viewport={"width": 1280, "height": 800})

    try:
        await page.goto(base_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        await dismiss_cookie_banner(page)
        await settle(page)

        for step in range(1, max_steps + 1):

            screenshot_bytes = await page.screenshot(type="jpeg", quality=75, full_page=False)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
            current_url = page.url

            try:
                action = await loop.run_in_executor(
                    executor,
                    decide_action_with_gemini,
                    screenshot_b64, intent, current_url, step, max_steps, history
                )
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    steps_log.append({
                        "step": step, "url": current_url,
                        "error": f"Decision LLM rate-limitee, nouvelle tentative : {err_str}"
                    })
                    await asyncio.sleep(2)
                    try:
                        action = await loop.run_in_executor(
                            executor,
                            decide_action_with_gemini,
                            screenshot_b64, intent, current_url, step, max_steps, history
                        )
                    except Exception as e2:
                        steps_log.append({"step": step, "url": current_url, "error": f"Decision LLM echouee apres retry : {e2}"})
                        break
                else:
                    steps_log.append({"step": step, "url": current_url, "error": f"Decision LLM echouee : {err_str}"})
                    break

            log_entry = {"step": step, "action": action, "url": current_url}

            last_step_failed_click = (
                history
                and history[-1].get("type") == "click"
                and history[-1].get("error")
            )
            if action.get("type") == "extract" and last_step_failed_click:
                action = {"type": "scroll"}
                log_entry["action"] = action
                log_entry["note"] = "extract refuse : le clic precedent a echoue, scroll force a la place"

            if action.get("type") == "extract":
                log_entry["result"] = "Contenu extrait"
                steps_log.append(log_entry)
                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                return {
                    "success": True,
                    "final_url": current_url,
                    "steps": steps_log,
                    "content": soup.get_text(separator="\n", strip=True)[:12000]
                }

            try:
                description = await execute_action(page, action)
                log_entry["result"] = description
                consecutive_failures = 0
            except Exception as e:
                log_entry["error"] = str(e)
                consecutive_failures += 1

            steps_log.append(log_entry)
            history_entry = {"step": step, **action}
            if "error" in log_entry:
                history_entry["error"] = log_entry["error"]
            elif "result" in log_entry:
                history_entry["result"] = log_entry["result"]
            history.append(history_entry)

            # OPTIM : arret anticipe si 2 echecs consecutifs sans progression,
            # au lieu de consommer tout le budget max_steps inutilement.
            if consecutive_failures >= 2:
                steps_log.append({
                    "step": step, "url": page.url,
                    "note": "Arret anticipe : 2 echecs consecutifs sans progression"
                })
                break

        final_url = page.url
        content = await page.content()
    finally:
        await context.close()

    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    navigated_away = final_url.rstrip("/") != base_url.rstrip("/")
    return {
        "success": navigated_away,
        "reason": (
            "Contenu recupere apres navigation, mais arret avant 'extract' explicite "
            f"(limite de {max_steps} etapes ou erreur de decision)"
            if navigated_away else
            f"Aucune navigation utile : limite de {max_steps} etapes atteinte ou erreur de decision"
        ),
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
    intent: str
    source: str
    prompt: str
    max_steps: int = DEFAULT_MAX_STEPS

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "ScraperWeb V2 — Powered by Gemini Flash Lite + Gemini Flash + Gemma 4 (open-source) + Magistral (fallbacks)",
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
    return {"status": "healthy", "service": "ScraperWeb V2", "models": {"primary": GEMINI_LITE_MODEL, "fallback_1": GEMINI_MODEL, "fallback_2": GEMMA_MODEL, "fallback_3": MAGISTRAL_MODEL}}

# ── Scraping simple ──

@app.get("/scrape")
async def scrape(
    request: Request,
    url: str = Query(..., description="URL a scraper"),
    prompt: str = Query(..., description="Question sur la page"),
    mode: Literal["simple", "browser"] = Query(
        "simple", description="simple = HTTP | browser = Playwright URL connue"
    )
):
    app_ = request.app
    try:
        result = await scrape_single(app_, url, prompt, mode)
        history_id = await save_to_history(app_.state.db, url, prompt, result["answer"], mode=mode, model=result.get("model", ""))
        return {"status": "success", "id": history_id, **result}
    except Exception as e:
        await save_to_history(app_.state.db, url, prompt, str(e), status="error", mode=mode)
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

# ── Scraping en masse ──

@app.post("/scrape/bulk")
async def scrape_bulk(request: BulkScrapeRequest, http_request: Request):
    app_ = http_request.app
    results = []
    errors = []

    async def scrape_one(url):
        try:
            result = await scrape_single(app_, url, request.prompt, request.mode)
            hid = await save_to_history(app_.state.db, url, request.prompt, result["answer"], mode=request.mode, model=result.get("model", ""))
            results.append({"id": hid, **result})
        except Exception as e:
            errors.append({"url": url, "error": str(e)})
            await save_to_history(app_.state.db, url, request.prompt, str(e), status="error", mode=request.mode)

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
async def agent_search(request: AgentSearchRequest, http_request: Request):
    """
    Lance un agent navigateur autonome pilote par Gemini Vision.

    L'agent navigue de lui-meme sur le site source pour trouver le contenu
    correspondant a l'intention, sans URL ni ID a fournir.
    Gemini (fallback Gemma 4 puis Magistral) analyse ensuite le contenu extrait et repond a la question.

    Exemple :
    {
      "intent": "Stats du match PSG vs Barcelone du 7 mai 2025",
      "source": "sofascore.com",
      "prompt": "Extrais possession, xG et tirs cadres pour chaque equipe",
      "max_steps": 8
    }
    """
    app_ = http_request.app
    try:
        agent_result = await run_browser_agent(app_, request.intent, request.source, request.max_steps)

        if not agent_result.get("content"):
            raise HTTPException(status_code=500, detail="L'agent n'a pas pu extraire de contenu")

        loop = asyncio.get_running_loop()
        answer, model_used = await loop.run_in_executor(
            executor, run_llm, agent_result["content"], request.prompt
        )

        history_id = await save_to_history(
            app_.state.db, agent_result["final_url"], request.prompt, answer, mode="agent", model=model_used
        )

        warning = None
        steps = agent_result["steps"]
        ended_with_extract = bool(steps) and steps[-1].get("action", {}).get("type") == "extract"
        if not ended_with_extract:
            warning = (
                "L'agent a ete interrompu avant un 'extract' explicite "
                f"({agent_result.get('reason', 'raison inconnue')}). "
                "Le contenu recupere peut etre incomplet par rapport a l'intention demandee."
            )
        elif len(steps) >= 2 and (steps[-2].get("error") or steps[-1].get("note")):
            warning = (
                "L'agent n'a pas reussi a naviguer jusqu'a la page detaillee "
                "(dernier clic en echec) ; le contenu extrait peut etre incomplet "
                "par rapport a l'intention demandee."
            )

        return {
            "status": "success",
            "id": history_id,
            "intent": request.intent,
            "prompt": request.prompt,
            "answer": answer,
            "mode": "agent",
            "model": model_used,
            "agent_success": agent_result["success"],
            "warning": warning,
            "final_url": agent_result["final_url"],
            "steps": agent_result["steps"]
        }

    except HTTPException:
        raise
    except Exception as e:
        await save_to_history(app_.state.db, request.source, request.prompt, str(e), status="error", mode="agent")
        raise HTTPException(status_code=500, detail=f"Erreur agent : {str(e)}")

# ── Historique ──

@app.get("/history")
async def get_history(request: Request, limit: int = 50, offset: int = 0):
    db = request.app.state.db
    async with db.execute(
        "SELECT id, url, prompt, status, mode, model, created_at FROM scrape_history ORDER BY id DESC LIMIT ? OFFSET ?",
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
            {"id": r[0], "url": r[1], "prompt": r[2], "status": r[3], "mode": r[4], "model": r[5], "created_at": r[6]}
            for r in rows
        ]
    }

@app.get("/history/{item_id}")
async def get_history_item(item_id: int, request: Request):
    db = request.app.state.db
    async with db.execute(
        "SELECT id, url, prompt, result, status, mode, model, created_at FROM scrape_history WHERE id=?",
        (item_id,)
    ) as cursor:
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Entree introuvable")
    return {
        "id": row[0], "url": row[1], "prompt": row[2],
        "result": row[3],
        "status": row[4], "mode": row[5], "model": row[6], "created_at": row[7]
    }

# ── Export ──

@app.get("/export/{item_id}")
async def export_result(item_id: int, request: Request, fmt: str = Query("json", alias="format", description="json | csv | pdf")):
    db = request.app.state.db
    async with db.execute(
        "SELECT id, url, prompt, result, status, mode, model, created_at FROM scrape_history WHERE id=?",
        (item_id,)
    ) as cursor:
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Entree introuvable")

    data = {
        "id": row[0], "url": row[1], "prompt": row[2],
        "result": row[3] or "",
        "status": row[4], "mode": row[5], "model": row[6], "created_at": row[7]
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
        result_html = xml_escape(str(data["result"])).replace("\n", "<br/>")
        story = [
            Paragraph(f"ScraperWeb V2 — Export #{item_id}", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"<b>URL :</b> {xml_escape(str(data['url']))}", styles["Normal"]),
            Spacer(1, 6),
            Paragraph(f"<b>Prompt :</b> {xml_escape(str(data['prompt']))}", styles["Normal"]),
            Spacer(1, 6),
            Paragraph(f"<b>Mode :</b> {xml_escape(str(data['mode']))}", styles["Normal"]),
            Spacer(1, 6),
            Paragraph(f"<b>Modele :</b> {xml_escape(str(data['model'] or ''))}", styles["Normal"]),
            Spacer(1, 6),
            Paragraph(f"<b>Date :</b> {xml_escape(str(data['created_at']))}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("<b>Resultat :</b>", styles["Heading2"]),
            Spacer(1, 6),
            Paragraph(result_html, styles["Normal"]),
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

async def scheduled_scrape(app: FastAPI, url: str, prompt: str, mode: str = "simple"):
    try:
        result = await scrape_single(app, url, prompt, mode)
        await save_to_history(app.state.db, url, prompt, result["answer"], mode=mode, model=result.get("model", ""))
    except Exception as e:
        await save_to_history(app.state.db, url, prompt, str(e), status="error", mode=mode)

@app.post("/schedule")
async def create_schedule(request: ScheduleRequest, http_request: Request):
    app_ = http_request.app
    try:
        trigger = CronTrigger.from_crontab(request.cron)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Expression cron invalide : {e}")

    db = app_.state.db
    cursor = await db.execute(
        "INSERT INTO scheduled_jobs (name, url, prompt, cron, mode) VALUES (?, ?, ?, ?, ?)",
        (request.name, request.url, request.prompt, request.cron, request.mode)
    )
    job_id = cursor.lastrowid
    await db.commit()

    scheduler.add_job(
        scheduled_scrape,
        trigger,
        args=[app_, request.url, request.prompt, request.mode],
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
async def list_schedules(request: Request):
    db = request.app.state.db
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
async def delete_schedule(job_id: int, request: Request):
    db = request.app.state.db
    await db.execute("UPDATE scheduled_jobs SET active=0 WHERE id=?", (job_id,))
    await db.commit()
    try:
        scheduler.remove_job(f"job_{job_id}")
    except Exception:
        pass
    return {"status": "deleted", "id": job_id}

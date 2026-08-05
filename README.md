# ScraperWeb V2

API de web scraping assistée par IA — FastAPI + Playwright + SQLite, avec
chaîne de fallback LLM (Gemini → Gemma → Mistral), agent navigateur autonome,
recherche web (SERP), export, planification cron, et logs structurés.

Fichier unique : tout le code vit dans `main.py` (volontairement, pour rester
facile à lire/éditer via Termux ou l'interface web GitHub).

---

## Démarrage rapide

```bash
pip install -r requirements.txt
playwright install --with-deps chromium   # une seule fois
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Docker :
```bash
docker build -t scraperweb-v2 .
docker run -p 8000:8000 --env-file .env scraperweb-v2
```

Docs interactives une fois lancé : `http://localhost:8000/docs`

---

## Variables d'environnement

| Variable          | Requise | Défaut   | Rôle                                                        |
|--------------------|:-------:|----------|--------------------------------------------------------------|
| `GEMINI_API_KEY`   | oui     | —        | Modèles principaux (Flash Lite, Flash, Gemma 4) + agent vision |
| `MISTRAL_API_KEY`  | oui     | —        | Fallback final (Magistral)                                   |
| `API_KEY`          | non     | —        | Si définie, protège tous les endpoints sauf `/`, `/health`, `/docs` (header `X-API-Key` ou `?api_key=`) |
| `ALLOWED_ORIGINS`  | non     | `*`      | Liste d'origines CORS séparées par des virgules              |
| `LOG_LEVEL`        | non     | `INFO`   | `DEBUG`, `INFO`, `WARNING`, `ERROR`                           |
| `LOG_DIR`          | non     | `logs`   | Dossier des fichiers de log (créé automatiquement)            |

Créer un `.env` local avec ces clés (non commité).

---

## Endpoints

### Scraping simple
```
GET /scrape?url=...&prompt=...&mode=simple|browser
```
- `simple` : requête HTTP brute (`requests` + BeautifulSoup), rapide, ne charge pas le JS.
- `browser` : rendu Playwright (JS exécuté), plus lent, pour les pages dynamiques.

### Scraping en masse
```
POST /scrape/bulk
{ "urls": ["..."], "prompt": "...", "mode": "simple" }
```
Toutes les URLs sont scrapées en parallèle (`asyncio.gather`).

### Agent navigateur autonome
```
POST /agent/search
{ "intent": "...", "source": "sofascore.com", "prompt": "...", "max_steps": 8 }
```
Boucle screenshot → décision Gemini (vision) → action Playwright, jusqu'à un
`extract` explicite ou 2 échecs consécutifs. À utiliser quand on ne connaît
pas l'URL exacte mais seulement le site et l'intention.

### Recherche web (SERP)
```
GET /serp?query=...&num_results=10
GET /serp/analyze?query=...&prompt=...&num_results=10
```
- `/serp` : résultats bruts (titre, url, extrait) via DuckDuckGo HTML, sans clé API.
- `/serp/analyze` : agrège les résultats et fait répondre le LLM à `prompt` en
  se basant dessus (mêmes fallbacks que `/scrape`). Utile pour une question
  ouverte qui ne vise pas un site précis.

### Historique & export
```
GET /history?limit=50&offset=0
GET /history/{id}
GET /export/{id}?format=json|csv|pdf
```

### Planification (cron)
```
POST   /schedule   { "name": "...", "url": "...", "prompt": "...", "cron": "0 8 * * *", "mode": "simple" }
GET    /schedule
DELETE /schedule/{id}
```
Syntaxe cron standard (5 champs). Les jobs actifs sont rechargés au démarrage
de l'app (voir `lifespan`).

### Divers
```
GET /        → liste des endpoints
GET /health  → statut + modèles configurés
```

---

## Chaîne de fallback LLM

Utilisée par `run_llm()` (analyse de contenu) et `decide_action_with_gemini()`
(décision de l'agent, chaîne plus courte car besoin de vision) :

1. **Gemini Flash Lite** (`models/gemini-flash-lite-latest`) — rapide, modèle principal
2. **Gemini Flash** (`models/gemini-2.5-flash-lite`) — fallback texte
3. **Gemma 4** (`gemma-4-26b-a4b-it`) — open-source, via l'API Gemini
4. **Magistral** (`mistral-small-latest`) — dernier recours (texte uniquement, pas de vision)

Chaque tentative est loggée (modèle essayé, durée, succès/échec) — voir
section Logs.

---

## Logs

Configurés une fois en haut de `main.py`, deux sorties :
- Console (stdout) — visible dans les logs Render/Docker.
- Fichier rotatif `logs/scraperweb.log` (5 Mo × 5 fichiers).

Format : `date | niveau | req=<request_id> | logger | message`

Chaque requête HTTP reçoit un `request_id` court (généré dans
`request_logging_middleware`), retourné aussi dans le header `X-Request-ID`
de la réponse. Ce même id est propagé explicitement (en paramètre de
fonction, pas de contextvar — plus fiable à travers les threads executor et
les étapes de l'agent) à travers tout l'appel : scraping, appels LLM,
étapes de l'agent, jobs cron. **Pour retrouver tout ce qui concerne un appel
précis, grep sur son `request_id`.**

Réglage du niveau de détail : `LOG_LEVEL=DEBUG` (variable d'env).

Convention à garder pour toute nouvelle fonction interne : accepter un
paramètre `request_id: str = "-"`, récupérer un logger via
`log = get_logger(request_id)`, et logguer au minimum le début, la fin
(avec durée) et les erreurs (`log.error` / `log.exception`).

---

## Structure de la base SQLite (`scraper.db`)

- `scrape_history` : chaque appel (`/scrape`, `/scrape/bulk`, `/agent/search`,
  `/serp`, `/serp/analyze`, jobs cron) y enregistre une ligne — url, prompt,
  résultat, statut (`success`/`error`), mode, modèle utilisé, date.
- `scheduled_jobs` : jobs cron définis via `/schedule`.

Le schéma est migré automatiquement au démarrage (`add_column_if_missing`),
donc pas besoin de migration manuelle en ajoutant une colonne — juste
l'ajouter à `init_db()`.

---

## Sécurité

- **SSRF** : `assert_public_url()` résout le hostname et bloque IP privées/
  loopback/link-local avant tout scraping (simple et browser). Ne pas
  contourner cette fonction pour un nouvel endpoint qui accepte une URL.
- **Auth optionnelle** : si `API_KEY` est définie, tous les endpoints sauf
  `/`, `/health`, `/docs`, `/openapi.json`, `/redoc` exigent
  `X-API-Key` (header) ou `?api_key=`.
- Le navigateur Playwright tourne avec des ressources bloquées (polices,
  médias, images en mode `simple`/scraping) pour rester rapide.

---

## Pièges connus / notes pour la prochaine session

- **Un seul fichier `main.py`** : c'est voulu (facilite l'édition via
  heredoc/GitHub web). Avant de le scinder en modules, vérifier que c'est
  vraiment nécessaire.
- **Navigateur et connexion DB persistants** : `app.state.browser` et
  `app.state.db` sont initialisés une fois dans `lifespan()`, jamais par
  requête. Ne pas réintroduire de `chromium.launch()` ou
  `aiosqlite.connect()` par endpoint (régression de perf déjà corrigée).
- **`domcontentloaded` + attente courte**, jamais `networkidle` (beaucoup de
  sites scrapés ont du polling live-score qui ne devient jamais idle).
- **SERP via DuckDuckGo HTML** (`fetch_serp_results`) : pas de clé API, mais
  fragile aux changements de structure HTML de DuckDuckGo. Si `/serp` renvoie
  soudainement 0 résultat, vérifier d'abord les sélecteurs CSS
  (`div.result`, `a.result__a`, `.result__snippet`) avant de suspecter autre
  chose — un `[fetch_serp_results] AUCUN resultat` dans les logs est le
  signal à chercher.
- **`requirements.txt`** : `mistralai` est explicitement désinstallé puis
  réinstallé dans le `Dockerfile` (conflit avec un package `mistral` fantôme
  sur certains environnements) — ne pas retirer cette ligne sans tester le
  build Docker complet.

---

## Historique des versions

- **2.4.0** — Logs structurés (request_id de bout en bout, fichier rotatif) +
  fonctionnalité SERP (`/serp`, `/serp/analyze`).
- **2.3.0** — Base : scraping simple/browser, agent navigateur, historique,
  export, planification cron.

# Apple Store Review Analysis

Collects reviews from the Apple App Store, runs VADER sentiment analysis and TF-IDF keyword extraction, and uses Claude to produce structured domain insights — all surfaced through a web UI and a REST API.

---

## Approach & Design Decisions

### Why fan out across all 155 storefronts?

Apple's RSS feed caps at roughly 50 reviews per storefront, and a single country gives a geographically narrow, often English-biased sample. A global app like Spotify has a very different complaint profile in the US vs Japan vs Brazil. Fetching only one storefront produces a misleading picture. The solution is to fan out across every storefront in parallel with a concurrency semaphore (so Apple's servers aren't flooded), deduplicate by review ID, sort by date, and return the N most recent. This gives a genuinely representative global sample in one request.

### Why three separate analysis layers instead of sending everything to Claude?

Sending raw reviews straight to an LLM and asking "what's wrong?" is expensive, slow, and produces inconsistent output. The pipeline is structured to use the cheapest tool that can do the job at each step:

1. **VADER sentiment first** — lexicon-based, runs in milliseconds, costs nothing, and handles the majority of English reviews reliably. There's no reason to pay for a model call on a review that says "app keeps crashing 1/5". VADER catches it. A Claude Haiku 4.5 call is only triggered when VADER's confidence falls below a threshold on a non-English review — fast tiebreaker, not the primary path.

2. **TF-IDF on segmented corpora** — running keyword extraction on *negative-only* and *positive-only* reviews separately is the key decision here. Generic TF-IDF across all reviews gets drowned out by filler words. Segmenting first means the words that surface are actually the ones users associate with frustration or delight, not just the most common words overall.

3. **Claude Opus 4.7 for structured domain insights** — this is where a human analyst would spend hours: reading hundreds of reviews, grouping complaints thematically, judging severity, and writing recommendations. Opus handles this in one pass and returns structured output (severity level, summary, specific issues, recommendation) per domain. The fixed taxonomy of seven domains (UX, monetisation, features, stability, content, auth, support) keeps the output consistent and directly actionable.

### Why three different Claude models?

Cost and task complexity should match. Global domain analysis is the hardest task — synthesising 100 reviews across seven dimensions requires genuine reasoning. That gets Opus 4.7. Per-region analysis is the same structure applied to a subset, so Sonnet 4.6 handles it — cheaper, and the narrower scope doesn't need Opus-level reasoning. Haiku 4.5 is used only for the binary sentiment tiebreaker where the task is trivial. Running everything on Opus would work but would be 5–10× more expensive per request.

### Why Redis and not just in-process memory?

An in-process dict works fine for a single-process dev server, but breaks down in production for two reasons. First, most production deployments run multiple worker processes or containers — each with its own memory space. Without a shared cache, a review collected by worker A is invisible to worker B, so the same Apple RSS fan-out and the same Opus call happen again on the very next request that lands on a different process. Second, in-process memory is lost on every restart or redeploy, which means cold-starting the LLM call on the first request after every deploy — exactly when latency is most visible.

Redis gives a single cache that all processes share and that survives restarts. The app probes the connection at startup and falls back gracefully to in-process memory if Redis is unavailable, so local development without Docker still works. In production (or the Docker Compose setup), Redis persistence means a 30-second Opus call from yesterday is still cached today.

### Why separate cache keys for reviews and insights?

Reviews and Claude insights have very different compute costs. If they shared a cache namespace, a metrics-only request could invalidate an expensive LLM result. Reviews and insights are stored under separate keys (`reviews:{app_id}:{country}` and `insights:{app_id}:{country}:{llm|raw}`), so each can expire independently. The retrieval logic also checks the richer LLM key first — if insights were already computed with Claude, a subsequent raw-only request gets them for free rather than recomputing.

### Incremental refresh: fetching only what changed

A full re-collection is wasteful if only a handful of reviews were published since the last fetch. The `GET /api/v1/reviews?incremental=true` endpoint handles this: it reads the most recent `updated` timestamp from the cached pool, then fans out across storefronts again — but each storefront's page walk stops the moment it encounters an entry older than that timestamp. Only the delta is returned, merged into the existing pool (deduped by review ID), sorted, and trimmed back to the requested count.

Whether to re-run the analysis pipeline after a merge is a deliberate decision rather than an automatic one. One or two new reviews won't shift sentiment scores, keyword frequencies, or domain severity levels in any meaningful way — running Opus 4.7 on an almost-identical corpus produces an almost-identical result at non-trivial cost. The insights cache is therefore only invalidated when the number of new reviews reaches the `MIN_NEW_REVIEWS_RERUN` threshold (default: 10). Below that threshold the cached insights stay in place; the next insights request will pick them up as-is. Above the threshold the cache is cleared and re-analysis happens on the next request for insights, not eagerly on the refresh call itself — keeping the incremental update fast regardless of whether the caller actually needs updated analysis.

### API structure

A combined `GET /api/v1/report/{app_id}` endpoint returns everything in one shot for the UI. Separate `/reviews`, `/metrics`, and `/insights` endpoints exist so callers can fetch only what they need — a dashboard that only wants metrics shouldn't trigger a Claude call. The `country` / `countries` query parameters accept a single code, a comma-separated list, or nothing (full global pool), all resolved through the same code path.

---

## Sample Report

The web UI at **http://localhost:8080** is itself the sample report. Type any app name or App Store ID, click **Analyse**, and the four-tab interface shows:

- **Metrics** — average rating, median, star distribution
- **Insights & Domain Analysis** — VADER sentiment, TF-IDF keywords, and Claude domain cards with severity levels and recommendations
- **Dashboard** — rating distribution, sentiment breakdown, review volume over time, and top countries by review count
- **Reviews** — paginated, searchable, filterable list with CSV download

---

## Prerequisites

- Python 3.11+
- Docker + Docker Compose (for the recommended setup)
- An Anthropic API key (optional — the app works without it, Claude features just turn off)

---

## Running Locally

### Option A — Docker Compose (recommended)

```bash
# 1. Clone and enter the repo
git clone https://github.com/smazepaa/review-analysis.git
cd review-analysis

# 2. Add your Anthropic key (optional)
cp .env.example .env
# open .env and set ANTHROPIC_API_KEY=sk-ant-...

# 3. Start the API + Redis
docker compose up --build
```

The API will be available at **http://localhost:8080**.
Redis runs alongside it and caches reviews and Claude insights across restarts.

---

### Option B — Local Python

```bash
# 1. Create a virtualenv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your Anthropic key (optional)
cp .env.example .env
# open .env and set ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the API
uvicorn app.main:app --reload --port 8080
```

> Without Docker, caching is in-process only (lost on restart). Start Redis separately and set `REDIS_URL=redis://localhost:6379` in `.env` to get persistence.

---

## API

| Surface | URL |
|---|---|
| **Web UI** | http://localhost:8080 |
| **Interactive API docs** (Swagger) | http://localhost:8080/docs |

The web UI is a thin layer over the REST API — everything it shows is also available as JSON directly.

### Key endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/api/v1/reviews` | Collect and return reviews. Pass `name=Spotify` or `app_id=324684580`. Add `country=us` or `countries=us,gb,jp` to target specific storefronts, or omit for the global pool. `refresh=true` forces a full re-fetch; `incremental=true` fetches only reviews newer than the cached ones — but falls back to a full re-fetch if `count` exceeds the cached pool size, since Apple's RSS gives no way to page into history from a specific date. |
| `GET` | `/api/v1/metrics/{app_id}` | Rating statistics for cached reviews — average, median, star distribution. |
| `GET` | `/api/v1/insights/{app_id}` | VADER sentiment, TF-IDF keywords, and Claude domain analysis. `use_llm=false` skips the Claude call if you only need the fast stats. |
| `GET` | `/api/v1/report/{app_id}` | Everything in one response: app info, metrics, and insights. What the UI calls on load. |
| `GET` | `/api/v1/reviews/{app_id}/download` | Download the cached review set as CSV or JSON (`fmt=csv` / `fmt=json`). |

All endpoints that accept `country` / `countries` follow the same resolution logic: a single two-letter code targets one storefront, a comma-separated list targets several, and omitting both fans out globally across all 155 storefronts.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Enables Claude-powered domain insights. Without it the app still works. |
| `REDIS_URL` | — | Redis connection string. Omit to use in-process cache only. |
| `CACHE_TTL_SECONDS` | `21600` | How long reviews and insights are cached (default: 6 hours). |
| `MIN_NEW_REVIEWS_RERUN` | `10` | Minimum number of new reviews (from an incremental fetch) required to invalidate the insights cache and trigger re-analysis. |
| `ANTHROPIC_MODEL` | `claude-opus-4-7` | Model for overall domain insights. |
| `ANTHROPIC_REGIONAL_MODEL` | `claude-sonnet-4-6` | Model for per-region insights. |
| `ANTHROPIC_SENTIMENT_MODEL` | `claude-haiku-4-5` | Model for multilingual sentiment tiebreaker. |
| `LOG_LEVEL` | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`). |
| `PORT` | `8080` | Port the server listens on. |

---

## Deploy to Google Cloud Run

```bash
gcloud run deploy review-analysis-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 512Mi \
  --port 8080
```

Set `ANTHROPIC_API_KEY` as a Cloud Run secret or environment variable in the Console after deploying.

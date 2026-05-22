import csv
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import cache
from .collector import (
    AppNotFoundError,
    CollectorError,
    NoReviewsError,
    fetch_new_reviews,
    fetch_reviews,
    resolve_app,
)
from .insights import compute_insights
from .metrics import compute_metrics
from .models import (
    AppInfo,
    FullReport,
    Insights,
    Metrics,
    Review,
    ReviewsResponse,
)

logger = logging.getLogger(__name__)

# Minimum number of new reviews required to invalidate the insights cache
# and signal that downstream callers should recompute analysis.
_MIN_NEW_FOR_RERUN: int = int(os.getenv("MIN_NEW_REVIEWS_RERUN", "10"))

app = FastAPI(
    title="Apple Store Review Analysis API",
    version="0.4.0",
    description=(
        "Collects reviews from the Apple App Store, computes rating metrics, "
        "runs VADER sentiment analysis, extracts keywords from negative reviews, "
        "and returns actionable insights. Pass a single storefront code, a "
        "comma-separated list of codes, or omit to pool every Apple storefront."
    ),
)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    start = time.perf_counter()

    qs = f"?{request.url.query}" if request.url.query else ""
    logger.info("→ %s %s%s", request.method, request.url.path, qs)

    try:
        response: Response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        logger.exception("✗ %s %s%s  UNHANDLED ERROR  (%.0f ms)", request.method, request.url.path, qs, elapsed)
        raise

    elapsed = (time.perf_counter() - start) * 1000
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(
        level,
        "← %s %s%s  %d  %.0f ms",
        request.method,
        request.url.path,
        qs,
        response.status_code,
        elapsed,
    )
    return response


COUNTRY_QUERY = Query(
    default=None,
    min_length=2,
    max_length=2,
    description=(
        "Single two-letter storefront code (e.g. 'us'). "
        "Ignored when `countries` is provided."
    ),
)
COUNTRIES_QUERY = Query(
    default=None,
    description=(
        "Comma-separated storefront codes (e.g. 'us,gb,jp'). "
        "Overrides `country`. Omit both to pool every storefront globally."
    ),
)
COUNT_QUERY = Query(default=100, ge=1, le=500)


def _normalize_country(country: Optional[str]) -> Optional[str]:
    if country is None:
        return None
    country = country.strip().lower()
    return country or None


def _effective_country(
    country: Optional[str],
    countries: Optional[str],
) -> Optional[str]:
    """Return the canonical country spec for this request"""
    if countries:
        codes = sorted({
            c.strip().lower()
            for c in countries.split(",")
            if len(c.strip()) == 2
        })
        if codes:
            return ",".join(codes)
    return _normalize_country(country)


async def _collect_and_cache(
    app_id: Optional[str],
    name: Optional[str],
    country: Optional[str],
    count: int,
) -> tuple[AppInfo, list[Review]]:
    try:
        info = await resolve_app(app_id=app_id, name=name, country=country)
        reviews = await fetch_reviews(info.app_id, country=country, count=count)
    except AppNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NoReviewsError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CollectorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}") from exc

    await cache.put_reviews(info.app_id, country, cache.CachedCollection(app=info, reviews=reviews))
    return info, reviews


async def _get_or_collect(
    app_id: str,
    country: Optional[str],
    count: int = 100,
) -> tuple[AppInfo, list[Review]]:
    cached = await cache.get_reviews(app_id, country)
    if cached is not None:
        return cached.app, cached.reviews
    return await _collect_and_cache(app_id=app_id, name=None, country=country, count=count)


async def _incremental_update(
    app_id: str,
    country: Optional[str],
    count: int,
    cached: cache.CachedCollection,
) -> tuple[AppInfo, list[Review]]:
    """Fetch only reviews newer than the most recent cached one.

    If ``count`` is larger than the cached pool, Apple's RSS gives no way to
    page into history from a specific date — so a full re-collection is done
    instead to fill the gap.

    Otherwise fetches only the delta, merges into the existing pool, and trims
    to ``count``. Invalidates the insights cache only when the number of new
    reviews meets or exceeds ``_MIN_NEW_FOR_RERUN`` — a small trickle won't
    meaningfully change the analysis.
    """
    if not cached.reviews:
        return cached.app, cached.reviews

    # If the caller wants more reviews than we have cached, we can't fill the
    # gap with a forward-only fetch — fall back to a full collection.
    if count > len(cached.reviews):
        logger.info(
            "Incremental update: requested %d reviews but cache has %d — "
            "falling back to full re-collection for app_id=%r",
            count, len(cached.reviews), app_id,
        )
        return await _collect_and_cache(app_id=app_id, name=None, country=country, count=count)

    latest_date = max(
        (r.updated for r in cached.reviews if r.updated),
        default=None,
    )
    if latest_date is None:
        logger.info("Incremental update skipped: no dated reviews in cache for app_id=%r", app_id)
        return cached.app, cached.reviews

    try:
        new_reviews = await fetch_new_reviews(app_id, since_date=latest_date, country=country)
    except Exception as exc:
        logger.warning("Incremental fetch failed (%s) — returning cached for app_id=%r", exc, app_id)
        return cached.app, cached.reviews

    if not new_reviews:
        logger.info("Incremental update: no new reviews for app_id=%r", app_id)
        return cached.app, cached.reviews

    # Merge: prepend new reviews, drop duplicates, sort, trim to requested count
    existing_ids = {r.review_id for r in cached.reviews if r.review_id}
    merged = new_reviews + [r for r in cached.reviews if not r.review_id or r.review_id not in existing_ids]
    merged.sort(key=lambda r: r.updated or "", reverse=True)
    merged = merged[:count]

    updated_collection = cache.CachedCollection(app=cached.app, reviews=merged)
    await cache.put_reviews(app_id, country, updated_collection)

    if len(new_reviews) >= _MIN_NEW_FOR_RERUN:
        logger.info(
            "Invalidating insights for app_id=%r (%d new reviews ≥ threshold %d)",
            app_id, len(new_reviews), _MIN_NEW_FOR_RERUN,
        )
        await cache.invalidate_insights(app_id, country)
    else:
        logger.info(
            "Keeping insights for app_id=%r (%d new reviews < threshold %d)",
            app_id, len(new_reviews), _MIN_NEW_FOR_RERUN,
        )

    return cached.app, merged


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/reviews", response_model=ReviewsResponse, tags=["reviews"])
async def get_reviews(
    app_id: Optional[str] = Query(
        default=None,
        description="Apple App Store numeric id. One of `app_id` or `name` is required.",
    ),
    name: Optional[str] = Query(
        default=None,
        description="App name to search for. Used when `app_id` is not known.",
    ),
    country: Optional[str] = COUNTRY_QUERY,
    countries: Optional[str] = COUNTRIES_QUERY,
    count: int = COUNT_QUERY,
    refresh: bool = Query(
        default=False,
        description="Force a full re-fetch from the App Store, replacing the cache entirely.",
    ),
    incremental: bool = Query(
        default=False,
        description=(
            "Fetch only reviews newer than the most recent cached one and merge them "
            "into the existing pool. The insights cache is invalidated only when the "
            f"number of new reviews reaches the MIN_NEW_REVIEWS_RERUN threshold "
            f"(default {_MIN_NEW_FOR_RERUN}). Ignored when `refresh=true`."
        ),
    ),
) -> ReviewsResponse:
    if not app_id and not name:
        raise HTTPException(
            status_code=400, detail="Provide either 'app_id' or 'name'."
        )

    country_spec = _effective_country(country, countries)

    # If only `name` was given, resolve it to an app_id up front
    if not app_id:
        try:
            info = await resolve_app(name=name, country=country_spec)
        except AppNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CollectorError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Upstream error: {exc}") from exc
        app_id = info.app_id

    if refresh:
        # Full re-collection — replaces cache entirely
        info, reviews = await _collect_and_cache(
            app_id=app_id, name=None, country=country_spec, count=count
        )
    elif incremental:
        # Diff-only fetch: pull what's new, merge, conditionally invalidate insights
        cached = await cache.get_reviews(app_id, country_spec)
        if cached is not None:
            info, reviews = await _incremental_update(app_id, country_spec, count, cached)
        else:
            # Nothing cached yet — fall back to a full initial collection
            info, reviews = await _collect_and_cache(
                app_id=app_id, name=None, country=country_spec, count=count
            )
    else:
        info, reviews = await _get_or_collect(app_id, country_spec, count)

    return ReviewsResponse(app=info, reviews=reviews)


@app.get("/api/v1/reviews/{app_id}/download", tags=["reviews"])
async def download_reviews(
    app_id: str,
    country: Optional[str] = COUNTRY_QUERY,
    countries: Optional[str] = COUNTRIES_QUERY,
    count: int = COUNT_QUERY,
    fmt: str = Query("csv", pattern="^(csv|json)$"),
):
    country_spec = _effective_country(country, countries)
    info, reviews = await _get_or_collect(app_id, country_spec, count)
    country_tag = (country_spec or "all").replace(",", "-")

    if fmt == "json":
        body = "[\n" + ",\n".join(r.model_dump_json() for r in reviews) + "\n]"
        filename = f"{info.app_id}-{country_tag}-reviews.json"
        return Response(
            content=body,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["review_id", "author", "rating", "title", "text", "version", "updated"])
    for r in reviews:
        writer.writerow([r.review_id, r.author, r.rating, r.title, r.text, r.version or "", r.updated or ""])
    filename = f"{info.app_id}-{country_tag}-reviews.csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/v1/metrics/{app_id}", response_model=Metrics, tags=["analytics"])
async def get_metrics(
    app_id: str,
    country: Optional[str] = COUNTRY_QUERY,
    countries: Optional[str] = COUNTRIES_QUERY,
    count: int = COUNT_QUERY,
) -> Metrics:
    _, reviews = await _get_or_collect(app_id, _effective_country(country, countries), count)
    return compute_metrics(reviews)


@app.get("/api/v1/insights/{app_id}", response_model=Insights, tags=["analytics"])
async def get_insights(
    app_id: str,
    country: Optional[str] = COUNTRY_QUERY,
    countries: Optional[str] = COUNTRIES_QUERY,
    count: int = COUNT_QUERY,
    use_llm: Optional[bool] = Query(
        default=None,
        description=(
            "Force the Claude-powered domain insights on or off. Defaults to on "
            "when ANTHROPIC_API_KEY is set, off otherwise."
        ),
    ),
) -> Insights:
    country_spec = _effective_country(country, countries)
    info, reviews = await _get_or_collect(app_id, country_spec, count)

    want_llm = use_llm is not False  # None (auto) or True both want LLM
    cached = await cache.get_insights(app_id, country_spec, want_llm=want_llm)
    if cached is not None:
        logger.info("Insights cache hit for app_id=%r country=%r", app_id, country_spec)
        return cached

    result = compute_insights(reviews, app_name=info.name, include_llm=use_llm)
    await cache.put_insights(app_id, country_spec, result)
    return result


@app.get("/api/v1/report/{app_id}", response_model=FullReport, tags=["analytics"])
async def get_full_report(
    app_id: str,
    country: Optional[str] = COUNTRY_QUERY,
    countries: Optional[str] = COUNTRIES_QUERY,
    count: int = COUNT_QUERY,
    use_llm: Optional[bool] = Query(default=None),
) -> FullReport:
    country_spec = _effective_country(country, countries)
    info, reviews = await _get_or_collect(app_id, country_spec, count)

    want_llm = use_llm is not False
    insights = await cache.get_insights(app_id, country_spec, want_llm=want_llm)
    if insights is None:
        insights = compute_insights(reviews, app_name=info.name, include_llm=use_llm)
        await cache.put_insights(app_id, country_spec, insights)
    else:
        logger.info("Insights cache hit for app_id=%r country=%r (report)", app_id, country_spec)

    return FullReport(
        app=info,
        metrics=compute_metrics(reviews),
        insights=insights,
    )


_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

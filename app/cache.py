import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from .models import AppInfo, Insights, Review

logger = logging.getLogger(__name__)

TTL: int = int(os.getenv("CACHE_TTL_SECONDS", str(6 * 3600)))

@dataclass
class CachedCollection:
    app: AppInfo
    reviews: list[Review]


_mem_reviews: dict[str, CachedCollection] = {}
_mem_insights: dict[str, Insights] = {}
_lock = asyncio.Lock()

_redis_client = None
_redis_available: Optional[bool] = None   # None = not yet probed


async def _redis():
    """Return a live Redis client, or None if unavailable."""
    global _redis_client, _redis_available

    if _redis_available is False:
        return None
    if _redis_client is not None:
        return _redis_client

    url = os.getenv("REDIS_URL")
    if not url:
        _redis_available = False
        logger.info("REDIS_URL not set — using in-process cache")
        return None

    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        await client.ping()
        _redis_client = client
        _redis_available = True
        logger.info("Redis cache connected: %s", url)
        return client
    except Exception as exc:
        _redis_available = False
        logger.warning("Redis unavailable (%s) — falling back to in-process cache", exc)
        return None


def _reviews_key(app_id: str, country: Optional[str]) -> str:
    return f"reviews:{app_id}:{(country or 'all').lower()}"


def _insights_key(app_id: str, country: Optional[str], *, llm: bool) -> str:
    return f"insights:{app_id}:{(country or 'all').lower()}:{'llm' if llm else 'raw'}"


async def get_reviews(app_id: str, country: Optional[str]) -> Optional[CachedCollection]:
    k = _reviews_key(app_id, country)
    r = await _redis()
    if r:
        try:
            raw = await r.get(k)
            if raw:
                data = json.loads(raw)
                logger.debug("Redis cache hit: %s", k)
                return CachedCollection(
                    app=AppInfo.model_validate(data["app"]),
                    reviews=[Review.model_validate(rv) for rv in data["reviews"]],
                )
        except Exception as exc:
            logger.warning("Redis get_reviews error: %s", exc)

    async with _lock:
        hit = _mem_reviews.get(k)
    if hit is not None:
        logger.debug("In-process cache hit: %s", k)
    return hit


async def put_reviews(app_id: str, country: Optional[str], collection: CachedCollection) -> None:
    k = _reviews_key(app_id, country)
    r = await _redis()
    if r:
        try:
            payload = json.dumps({
                "app": collection.app.model_dump(),
                "reviews": [rv.model_dump() for rv in collection.reviews],
            })
            await r.setex(k, TTL, payload)
            logger.debug("Redis cache set (TTL %ds): %s", TTL, k)
            return
        except Exception as exc:
            logger.warning("Redis put_reviews error: %s", exc)

    async with _lock:
        _mem_reviews[k] = collection
        logger.debug("In-process cache set: %s", k)


async def get_insights(
    app_id: str,
    country: Optional[str],
    *,
    want_llm: bool,
) -> Optional[Insights]:
    """Return cached Insights, preferring the richer LLM variant.

    Checks :llm first regardless of want_llm — if LLM results were already
    computed and stored they are returned for free.  Falls through to :raw
    only when want_llm is False and no LLM cache exists.
    """
    keys_to_try = [True, False] if not want_llm else [True]
    for llm_flag in keys_to_try:
        k = _insights_key(app_id, country, llm=llm_flag)
        r = await _redis()
        if r:
            try:
                raw = await r.get(k)
                if raw:
                    logger.debug("Redis cache hit: %s", k)
                    return Insights.model_validate_json(raw)
            except Exception as exc:
                logger.warning("Redis get_insights error: %s", exc)

        async with _lock:
            cached = _mem_insights.get(k)
        if cached is not None:
            logger.debug("In-process cache hit: %s", k)
            return cached

    return None


async def invalidate_insights(app_id: str, country: Optional[str]) -> None:
    """Delete cached insights for both the :llm and :raw variants."""
    for llm_flag in (True, False):
        k = _insights_key(app_id, country, llm=llm_flag)
        r = await _redis()
        if r:
            try:
                await r.delete(k)
                logger.debug("Redis cache deleted: %s", k)
            except Exception as exc:
                logger.warning("Redis invalidate_insights error: %s", exc)
        async with _lock:
            _mem_insights.pop(k, None)
            logger.debug("In-process cache deleted: %s", k)


async def put_insights(
    app_id: str,
    country: Optional[str],
    insights: Insights,
) -> None:
    """Persist Insights. The :llm vs :raw key is chosen automatically."""
    had_llm = insights.llm_model is not None
    k = _insights_key(app_id, country, llm=had_llm)
    r = await _redis()
    if r:
        try:
            await r.setex(k, TTL, insights.model_dump_json())
            logger.debug("Redis cache set (TTL %ds): %s", TTL, k)
            return
        except Exception as exc:
            logger.warning("Redis put_insights error: %s", exc)

    async with _lock:
        _mem_insights[k] = insights
        logger.debug("In-process cache set: %s", k)

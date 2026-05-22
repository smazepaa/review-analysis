import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from .models import AppInfo, Review
from .storefronts import ALL_STOREFRONTS, DEFAULT_CONCURRENCY

logger = logging.getLogger(__name__)

ITUNES_REVIEWS_URL = (
    "https://itunes.apple.com/{country}/rss/customerreviews"
    "/page={page}/id={app_id}/sortby=mostrecent/json"
)
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_LOOKUP_URL = "https://itunes.apple.com/lookup"

DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
PAGE_SIZE = 50
MAX_PAGES = 10

# Used as the cache key and AppInfo.country when no country was specified.
ALL_COUNTRIES = "all"

# When fanning out across every storefront, only fetch this many pages per
# storefront to keep the total request count tractable. One page = up to 50
# reviews per country, which across ~155 storefronts is already a deeper
# pool than any caller would need.
PAGES_PER_STOREFRONT_GLOBAL = 1


class CollectorError(Exception):
    """Raised for any collection failure surfaced to the API layer."""

class AppNotFoundError(CollectorError):
    pass


class NoReviewsError(CollectorError):
    pass


def _entry_to_review(entry: dict, country: str) -> Optional[Review]:
    """Convert a single RSS feed entry into a Review."""
    try:
        rating_str = entry["im:rating"]["label"]
        rating = int(rating_str)
    except (KeyError, TypeError, ValueError):
        return None

    if not (1 <= rating <= 5):
        return None

    return Review(
        review_id=entry.get("id", {}).get("label", ""),
        author=entry.get("author", {}).get("name", {}).get("label", "anonymous"),
        title=entry.get("title", {}).get("label", ""),
        text=entry.get("content", {}).get("label", ""),
        rating=rating,
        version=entry.get("im:version", {}).get("label"),
        updated=entry.get("updated", {}).get("label"),
        country=country,
    )


async def lookup_app(app_id: str, country: str = "us") -> AppInfo:
    """Resolve an App Store ID to AppInfo via the iTunes lookup endpoint."""
    logger.debug("Looking up app_id=%r in country=%r", app_id, country)
    params = {"id": app_id, "country": country}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        resp = await client.get(ITUNES_LOOKUP_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

    results = payload.get("results") or []
    if not results:
        raise AppNotFoundError(f"App with id={app_id!r} not found in country={country!r}")

    item = results[0]
    info = AppInfo(
        app_id=str(item.get("trackId", app_id)),
        name=item.get("trackName", "Unknown"),
        country=country,
        artist=item.get("artistName"),
    )
    logger.info("Resolved app_id=%r → %r (%r)", app_id, info.name, info.artist)
    return info


async def search_app(name: str, country: str = "us") -> AppInfo:
    """Resolve an app name to AppInfo via the iTunes search endpoint."""
    logger.debug("Searching for app name=%r in country=%r", name, country)
    params = {
        "term": name,
        "country": country,
        "entity": "software",
        "limit": 1,
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        resp = await client.get(ITUNES_SEARCH_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

    results = payload.get("results") or []
    if not results:
        raise AppNotFoundError(f"No app found matching name={name!r}")

    item = results[0]
    info = AppInfo(
        app_id=str(item["trackId"]),
        name=item.get("trackName", name),
        country=country,
        artist=item.get("artistName"),
    )
    logger.info("Found app name=%r → app_id=%r %r (%r)", name, info.app_id, info.name, info.artist)
    return info


async def resolve_app(
    app_id: Optional[str] = None,
    name: Optional[str] = None,
    country: Optional[str] = None,
) -> AppInfo:
    if not app_id and not name:
        raise CollectorError("Either app_id or name must be provided")

    if country and "," in country:
        lookup_country = country.split(",")[0].strip() or "us"
    else:
        lookup_country = country or "us"

    if app_id:
        info = await lookup_app(app_id, lookup_country)
    else:
        info = await search_app(name, lookup_country)

    if country is None:
        info = info.model_copy(update={"country": ALL_COUNTRIES})
    elif "," in country:
        info = info.model_copy(update={"country": country})
    return info


async def _fetch_page(
    client: httpx.AsyncClient, app_id: str, country: str, page: int
) -> list[dict]:
    url = ITUNES_REVIEWS_URL.format(country=country, page=page, app_id=app_id)
    try:
        resp = await client.get(url)
    except httpx.HTTPError:
        return []
    if resp.status_code in (403, 404):
        return []
    if resp.status_code >= 500:
        return []
    try:
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError):
        return []
    feed = body.get("feed") or {}
    entries = feed.get("entry") or []
    if isinstance(entries, dict):
        entries = [entries]
    return entries


async def _fetch_storefront(
    client: httpx.AsyncClient,
    app_id: str,
    country: str,
    max_pages: int,
) -> list[Review]:
    """Pull all pages for one storefront. Stops on the first empty/short page."""
    out: list[Review] = []
    for page in range(1, max_pages + 1):
        entries = await _fetch_page(client, app_id, country, page)
        if not entries:
            break
        for entry in entries:
            review = _entry_to_review(entry, country=country)
            if review is not None:
                out.append(review)
        if len(entries) < PAGE_SIZE:
            break
    return out


async def _fetch_storefronts_parallel(
    app_id: str,
    countries: tuple[str, ...],
    max_pages: int,
    concurrency: int,
) -> list[Review]:
    logger.info(
        "Fanning out across %d storefronts (concurrency=%d, max_pages=%d)",
        len(countries),
        concurrency,
        max_pages,
    )
    sem = asyncio.Semaphore(concurrency)
    pool: list[Review] = []

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:

        async def worker(country: str) -> list[Review]:
            async with sem:
                return await _fetch_storefront(client, app_id, country, max_pages)

        results = await asyncio.gather(*(worker(c) for c in countries))

    empty_storefronts = sum(1 for batch in results if not batch)
    for batch in results:
        pool.extend(batch)

    logger.info(
        "Storefront fan-out complete: %d storefronts returned reviews, "
        "%d returned nothing, %d raw reviews total",
        len(countries) - empty_storefronts,
        empty_storefronts,
        len(pool),
    )
    return pool


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 date string into a timezone-aware datetime, or None."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


async def _fetch_storefront_since(
    client: httpx.AsyncClient,
    app_id: str,
    country: str,
    since: datetime,
) -> list[Review]:
    """Fetch reviews newer than `since` from one storefront.

    Apple's RSS feed returns entries sorted newest-first, so we stop fetching
    further pages the moment we encounter an entry older than `since`.
    """
    out: list[Review] = []
    for page in range(1, MAX_PAGES + 1):
        entries = await _fetch_page(client, app_id, country, page)
        if not entries:
            break
        all_old = True
        for entry in entries:
            review = _entry_to_review(entry, country=country)
            if review is None:
                continue
            review_date = _parse_date(review.updated)
            if review_date is not None and review_date <= since:
                # Hit something older — everything from here on is also older
                return out
            all_old = False
            out.append(review)
        if all_old:
            break
    return out


async def fetch_new_reviews(
    app_id: str,
    since_date: str,
    country: Optional[str] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[Review]:
    """Return only reviews published after `since_date` (ISO 8601 string).

    Fans out across the same storefront set as ``fetch_reviews`` but stops
    each storefront's page walk as soon as older entries are encountered,
    making incremental fetches much cheaper than a full re-collection.
    """
    since = _parse_date(since_date)
    if since is None:
        raise CollectorError(f"Invalid since_date: {since_date!r}")

    if country and "," in country:
        codes: tuple[str, ...] = tuple(c.strip() for c in country.split(",") if c.strip())
    elif country is None:
        codes = ALL_STOREFRONTS
    else:
        codes = (country,)

    sem = asyncio.Semaphore(min(concurrency, len(codes)))
    pool: list[Review] = []

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:

        async def worker(sf: str) -> list[Review]:
            async with sem:
                return await _fetch_storefront_since(client, app_id, sf, since)

        results = await asyncio.gather(*(worker(c) for c in codes))

    for batch in results:
        pool.extend(batch)

    # Deduplicate
    seen_ids: set[str] = set()
    deduped: list[Review] = []
    for r in pool:
        if r.review_id and r.review_id in seen_ids:
            continue
        if r.review_id:
            seen_ids.add(r.review_id)
        deduped.append(r)

    logger.info(
        "Incremental fetch for app_id=%r since=%s → %d new review(s) across %d storefront(s)",
        app_id, since_date, len(deduped), len(codes),
    )
    return deduped


async def fetch_reviews(
    app_id: str,
    country: Optional[str] = None,
    count: int = 100,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[Review]:
    if count < 1:
        raise CollectorError("count must be >= 1")

    if country and "," in country:
        codes = tuple(c.strip() for c in country.split(",") if c.strip())
        if len(codes) <= 5:
            pages = MAX_PAGES
        elif len(codes) <= 20:
            pages = 5
        else:
            pages = PAGES_PER_STOREFRONT_GLOBAL
        logger.info(
            "Fetching reviews for app_id=%r across %d selected storefronts "
            "(pages=%d, want=%d)",
            app_id, len(codes), pages, count,
        )
        raw_pool = await _fetch_storefronts_parallel(
            app_id,
            codes,
            max_pages=pages,
            concurrency=min(concurrency, len(codes)),
        )
        scope_label = f"{len(codes)} selected storefronts"

    elif country is None:
        logger.info("Fetching reviews for app_id=%r (global fan-out, want=%d)", app_id, count)
        raw_pool = await _fetch_storefronts_parallel(
            app_id,
            ALL_STOREFRONTS,
            max_pages=PAGES_PER_STOREFRONT_GLOBAL,
            concurrency=concurrency,
        )
        scope_label = "any storefront"

    else:
        logger.info("Fetching reviews for app_id=%r country=%r (want=%d)", app_id, country, count)
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            raw_pool = await _fetch_storefront(client, app_id, country, MAX_PAGES)
        logger.debug("Single-storefront fetch returned %d raw reviews", len(raw_pool))
        scope_label = f"country={country!r}"

    seen_ids: set[str] = set()
    pool: list[Review] = []
    for review in raw_pool:
        if review.review_id and review.review_id in seen_ids:
            continue
        if review.review_id:
            seen_ids.add(review.review_id)
        pool.append(review)

    dupes = len(raw_pool) - len(pool)
    if dupes:
        logger.debug("Deduplicated %d duplicate review(s); %d unique remain", dupes, len(pool))

    if not pool:
        raise NoReviewsError(f"No reviews available for app_id={app_id!r} in {scope_label}")

    pool.sort(key=lambda r: r.updated or "", reverse=True)

    if len(pool) <= count:
        logger.info("Returning all %d reviews (pool ≤ requested %d)", len(pool), count)
        return pool

    logger.info("Returning %d most-recent reviews from a pool of %d", count, len(pool))
    return pool[:count]

import logging
from collections import Counter, defaultdict
from typing import Iterable, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import llm

logger = logging.getLogger(__name__)
from .models import (
    DomainInsight,
    Insights,
    RegionalSummary,
    Review,
    SentimentBreakdown,
)
from .processor import clean_text, normalize_for_keywords
from .regions import region_for

_MIN_REVIEWS_PER_REGION = 5
_MIN_REVIEWS_FOR_LLM = 10
_MAX_REGIONS_FOR_LLM = 4

_ANALYZER = SentimentIntensityAnalyzer()

# VADER compound score thresholds
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# Tiebreaking thresholds for 3-star reviews
_TIEBREAK_POSITIVE = 0.3
_TIEBREAK_NEGATIVE = -0.3

_ISSUE_RULES: list[tuple[tuple[str, ...], str]] = [
    (
        ("crash", "freeze", "freezes", "frozen", "force close", "keeps closing"),
        "Investigate stability issues — multiple negative reviews report crashes or freezes.",
    ),
    (
        ("slow", "lag", "laggy", "loading", "buffer", "buffering"),
        "Improve performance — users complain about slowness, lag, or loading times.",
    ),
    (
        ("ads", "ad ", "advertisement", "too many ads"),
        "Reduce or rebalance ad load — ads are a major friction point in negative feedback.",
    ),
    (
        ("subscription", "premium", "paywall", "expensive", "pricing", "price", "charge"),
        "Revisit pricing and subscription messaging — users feel value or transparency is lacking.",
    ),
    (
        ("login", "log in", "sign in", "signin", "password", "account", "logged out"),
        "Fix authentication friction — login, sign-in or account problems appear in negative reviews.",
    ),
    (
        ("update", "new version", "latest update", "after update"),
        "Audit the latest update — sentiment regression appears tied to a recent release.",
    ),
    (
        ("bug", "glitch", "broken", "doesn't work", "not working", "error"),
        "Triage reported bugs — a meaningful share of negative reviews cite broken functionality.",
    ),
    (
        ("ui", "interface", "design", "ux", "layout", "confusing"),
        "Review UI/UX clarity — users describe the interface as confusing or hard to navigate.",
    ),
    (
        ("logo", "icon", "rebrand", "new look", "new design", "redesign"),
        "Re-evaluate the recent visual redesign / rebrand — logo or icon changes are drawing strong negative reaction.",
    ),
    (
        ("offline", "download", "downloaded songs", "no internet"),
        "Strengthen offline behaviour — offline mode and downloads are recurring pain points.",
    ),
    (
        ("customer service", "support", "help", "response"),
        "Improve customer support responsiveness — reviewers feel unheard when reaching out.",
    ),
]

# Augment scikit-learn's English stop list with terms that are uninformative
# for app reviews specifically
_EXTRA_STOPWORDS = {
    "app",
    "apps",
    "phone",
    "iphone",
    "ipad",
    "ios",
    "apple",
    "store",
    "really",
    "just",
    "like",
    "love",
    "good",
    "great",
    "best",
    "worst",
    "bad",
    "thing",
    "things",
    "way",
    "time",
    "times",
    "use",
    "used",
    "using",
    "user",
    "users",
    "people",
    "make",
    "makes",
    "made",
    "want",
    "wanted",
    "wants",
    "even",
    "still",
    "also",
    "would",
    "could",
    "im",
    "ive",
    "dont",
    "doesnt",
    "didnt",
    "wont",
    "isnt",
    "5",
    "stars",
    "star",
    "review",
    "reviews",
}


def classify(text: str, rating: Optional[int] = None) -> tuple[str, float]:
    """Return (label, compound) for one review."""
    compound = _ANALYZER.polarity_scores(text)["compound"] if text else 0.0

    if rating is None:
        if compound >= POSITIVE_THRESHOLD:
            return ("positive", compound)
        if compound <= NEGATIVE_THRESHOLD:
            return ("negative", compound)
        return ("neutral", compound)

    # Rating-aware path: trust the user's self-report, refine ambiguous 3★.
    if rating >= 4:
        return ("positive", compound)
    if rating <= 2:
        return ("negative", compound)

    # 3★: genuinely ambivalent. Lean on VADER only for clear text signal,
    # with stricter thresholds so we don't flip the label on a single
    # exclamation point.
    if compound >= _TIEBREAK_POSITIVE:
        return ("positive", compound)
    if compound <= _TIEBREAK_NEGATIVE:
        return ("negative", compound)
    return ("neutral", compound)


def _review_text(review: Review) -> str:
    return clean_text(f"{review.title}. {review.text}")


def _sentiment_breakdown(scored: list[tuple[Review, str, float]]) -> SentimentBreakdown:
    counts = Counter(label for _, label, _ in scored)
    counts.setdefault("positive", 0)
    counts.setdefault("neutral", 0)
    counts.setdefault("negative", 0)
    total = sum(counts.values()) or 1
    percentages = {k: round(100 * v / total, 2) for k, v in counts.items()}
    avg_compound = round(
        sum(score for _, _, score in scored) / total if scored else 0.0,
        3,
    )
    return SentimentBreakdown(
        counts=dict(counts),
        percentages=percentages,
        average_compound=avg_compound,
    )


def _top_keywords(docs: list[str], top_n: int = 10) -> list[str]:
    """Top TF-IDF terms across a small document set."""
    docs = [d for d in docs if d.strip()]
    if not docs:
        return []

    # Small corpora: allow singletons so rare issues still surface.
    # Larger corpora: require at least 2 occurrences to filter noise.
    min_df = 1 if len(docs) < 15 else 2
    max_df = 1.0 if len(docs) < 4 else 0.85

    try:
        vec = TfidfVectorizer(
            stop_words=list(_sklearn_stopwords()),
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            token_pattern=r"(?u)\b[a-z][a-z]+\b",
        )
        matrix = vec.fit_transform(docs)
    except ValueError:
        return []

    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = scores.argsort()[::-1]
    return [terms[i] for i in order[:top_n]]


def _sklearn_stopwords() -> set[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return set(ENGLISH_STOP_WORDS) | _EXTRA_STOPWORDS


def _actionable_insights(
    negative_keywords: list[str],
    sentiment: SentimentBreakdown,
    avg_rating: float,
) -> list[str]:
    """Map keyword patterns into human-readable recommendations."""
    findings: list[str] = []
    seen: set[str] = set()

    blob = " ".join(negative_keywords).lower()
    for triggers, advice in _ISSUE_RULES:
        if any(t in blob for t in triggers) and advice not in seen:
            findings.append(advice)
            seen.add(advice)

    if not findings:
        # No specific pattern matched — fall back to a general note.
        if sentiment.counts.get("negative", 0) > 0:
            findings.append(
                "Review the lowest-rated reviews directly — negative sentiment exists "
                "but does not cluster around a single recurring theme."
            )

    neg_pct = sentiment.percentages.get("negative", 0.0)
    if neg_pct >= 30:
        findings.insert(
            0,
            f"{neg_pct:.1f}% of reviews are negative — treat this as a priority signal, "
            "not background noise.",
        )

    if avg_rating < 3.5:
        findings.append(
            f"Average rating is {avg_rating:.2f}/5 — overall satisfaction is below the "
            "threshold typically required to drive organic growth on the App Store."
        )

    return findings


def _basic_metrics_for(
    reviews: list[Review],
    *,
    llm_overrides: Optional[dict[str, str]] = None,
) -> tuple[SentimentBreakdown, list[str], list[str], float]:
    """Return (sentiment, neg_keywords, pos_keywords, avg_rating) for a review subset"""
    overrides = llm_overrides or {}
    scored: list[tuple[Review, str, float]] = []
    for r in reviews:
        label, compound = classify(_review_text(r), rating=r.rating)
        if r.review_id and r.review_id in overrides:
            label = overrides[r.review_id]
        scored.append((r, label, compound))
    sentiment = _sentiment_breakdown(scored)

    # Rating-based pools: language-independent and capture mixed signals.
    negative_docs = [
        normalize_for_keywords(_review_text(r)) for r in reviews if r.rating <= 3
    ]
    positive_docs = [
        normalize_for_keywords(_review_text(r)) for r in reviews if r.rating >= 4
    ]
    neg_keywords = _top_keywords(negative_docs, top_n=10)
    pos_keywords = _top_keywords(positive_docs, top_n=10)
    avg_rating = sum(r.rating for r in reviews) / len(reviews)
    return sentiment, neg_keywords, pos_keywords, avg_rating


def _is_global_scope(reviews: list[Review]) -> bool:
    countries = {r.country for r in reviews if r.country}
    return len(countries) > 1


def _regional_breakdown(
    reviews: list[Review],
    app_name: str,
    *,
    include_llm: bool,
    llm_overrides: Optional[dict[str, str]] = None,
) -> list[RegionalSummary]:
    """Group reviews by region and compute per-region stats (+ optional LLM)"""
    by_region: dict[str, list[Review]] = defaultdict(list)
    for r in reviews:
        by_region[region_for(r.country)].append(r)

    # Pre-filter to regions with enough reviews to be meaningful, sort by size.
    qualified: list[tuple[str, list[Review]]] = sorted(
        ((region, rs) for region, rs in by_region.items() if len(rs) >= _MIN_REVIEWS_PER_REGION),
        key=lambda kv: -len(kv[1]),
    )

    # Decide which regions get an LLM call (largest N first, above threshold).
    llm_regions: set[str] = set()
    if include_llm:
        for region, rs in qualified[:_MAX_REGIONS_FOR_LLM]:
            if len(rs) >= _MIN_REVIEWS_FOR_LLM:
                llm_regions.add(region)

    # Per-region calls use the cheaper regional tier model.
    regional_model = llm.DEFAULT_REGIONAL_MODEL

    summaries: list[RegionalSummary] = []
    for region, rs in qualified:
        sentiment, neg_kw, _, avg_rating = _basic_metrics_for(rs, llm_overrides=llm_overrides)
        domain: list = []
        used_model: Optional[str] = None
        if region in llm_regions:
            logger.info(
                "Requesting regional domain insights: region=%r, n=%d, model=%s",
                region,
                len(rs),
                regional_model,
            )
            domain = llm.generate_domain_insights(
                rs,
                app_name=app_name,
                scope_label=f"region: {region}",
                model=regional_model,
            )
            if domain:
                used_model = regional_model
        else:
            logger.debug(
                "Region %r: %d reviews (stats only, no LLM call)", region, len(rs)
            )
        countries = sorted({r.country for r in rs if r.country})
        summaries.append(
            RegionalSummary(
                region=region,
                countries=countries,
                review_count=len(rs),
                average_rating=round(avg_rating, 2),
                sentiment_percentages=sentiment.percentages,
                top_negative_keywords=neg_kw,
                domain_insights=domain,
                llm_model=used_model,
            )
        )
    return summaries


def compute_insights(
    reviews: Iterable[Review],
    *,
    app_name: str = "this app",
    include_llm: Optional[bool] = None,
) -> Insights:
    reviews = list(reviews)
    if not reviews:
        raise ValueError("Cannot compute insights over an empty review set")

    if include_llm is None:
        include_llm = llm.is_enabled()

    logger.info(
        "Computing insights for %r: %d reviews, LLM=%s",
        app_name,
        len(reviews),
        "enabled" if include_llm else "disabled",
    )

    sentiment_overrides: dict[str, str] = {}
    if include_llm:
        ambiguous = [
            r for r in reviews
            if r.rating == 3
            and classify(_review_text(r), rating=r.rating)[0] == "neutral"
        ]
        if ambiguous:
            logger.info(
                "Running Haiku multilingual tiebreaker on %d ambiguous 3★ review(s)",
                len(ambiguous),
            )
            haiku_labels = llm.classify_sentiment_bulk(ambiguous)
            if haiku_labels:
                sentiment_overrides = {
                    r.review_id: lbl
                    for r, lbl in zip(ambiguous, haiku_labels)
                    if r.review_id
                }
                logger.info(
                    "Haiku tiebreaker produced %d override(s)", len(sentiment_overrides)
                )
        else:
            logger.debug("No ambiguous 3★ reviews — skipping Haiku tiebreaker")
    else:
        # Count ambiguous reviews anyway so the log is informative even when LLM is off.
        ambiguous_count = sum(
            1 for r in reviews
            if r.rating == 3
            and classify(_review_text(r), rating=r.rating)[0] == "neutral"
        )
        if ambiguous_count:
            logger.debug(
                "%d ambiguous 3★ review(s) will stay as VADER-neutral (LLM disabled)",
                ambiguous_count,
            )

    logger.debug("Running VADER + rating-aware classifier on %d reviews", len(reviews))
    sentiment, neg_keywords, pos_keywords, avg_rating = _basic_metrics_for(
        reviews, llm_overrides=sentiment_overrides
    )
    logger.info(
        "Sentiment breakdown: positive=%.1f%%, neutral=%.1f%%, negative=%.1f%%, avg_rating=%.2f",
        sentiment.percentages.get("positive", 0),
        sentiment.percentages.get("neutral", 0),
        sentiment.percentages.get("negative", 0),
        avg_rating,
    )
    actionable = _actionable_insights(neg_keywords, sentiment, avg_rating)

    domain_insights: list[DomainInsight] = []
    llm_model: Optional[str] = None
    if include_llm:
        overall_model = llm.DEFAULT_MODEL
        logger.info("Requesting domain insights from Claude (%s, scope=overall)", overall_model)
        domain_insights = llm.generate_domain_insights(
            reviews,
            app_name=app_name,
            scope_label="overall sample",
            model=overall_model,
        )
        if domain_insights:
            llm_model = overall_model
            logger.info("Domain insights received: %d domain(s)", len(domain_insights))
        else:
            logger.warning("Domain insights call returned no results")

    regional: list[RegionalSummary] = []
    if _is_global_scope(reviews):
        logger.info("Global scope detected — computing regional breakdown")
        regional = _regional_breakdown(
            reviews,
            app_name=app_name,
            include_llm=include_llm,
            llm_overrides=sentiment_overrides,
        )
        logger.info("Regional breakdown: %d region(s) with enough reviews", len(regional))
    else:
        logger.debug("Single-country scope — skipping regional breakdown")

    return Insights(
        sentiment=sentiment,
        top_negative_keywords=neg_keywords,
        top_positive_keywords=pos_keywords,
        actionable_insights=actionable,
        domain_insights=domain_insights,
        regional_breakdown=regional,
        llm_model=llm_model,
    )

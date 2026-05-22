import logging
import os
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, Field

from .models import DomainInsight, Review

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")
DEFAULT_REGIONAL_MODEL = os.environ.get("ANTHROPIC_REGIONAL_MODEL", "claude-sonnet-4-6")
DEFAULT_SENTIMENT_MODEL = os.environ.get("ANTHROPIC_SENTIMENT_MODEL", "claude-haiku-4-5")

_THINKING_AND_EFFORT_MODELS = (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
)


def _supports_thinking_and_effort(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _THINKING_AND_EFFORT_MODELS)


def _call_kwargs(model: str) -> dict:
    if _supports_thinking_and_effort(model):
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        }
    return {}

MAX_REVIEWS_PER_CALL = 100
MAX_REVIEW_CHARS = 600

DOMAINS = (
    "Design & UX",
    "Pricing & Monetization",
    "Features",
    "Performance & Stability",
    "Content & Recommendations",
    "Account & Authentication",
    "Customer Support",
)

Severity = Literal["none", "low", "medium", "high"]


class _DomainInsightLLM(BaseModel):
    domain: str = Field(description="One of the fixed taxonomy slots.")
    summary: str = Field(description="1-2 sentence neutral observation.")
    key_issues: list[str] = Field(
        default_factory=list,
        description="Up to 5 concrete problems users raised. Empty if severity='none'.",
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="One concrete action the product team should take. Null if severity='none'.",
    )
    severity: Severity
    representative_quotes: list[str] = Field(
        default_factory=list,
        description="1-3 verbatim quotes drawn from the reviews. Empty if severity='none'.",
    )


class _LLMResponse(BaseModel):
    domains: list[_DomainInsightLLM]


SYSTEM_PROMPT = """\
You analyze Apple App Store reviews and produce structured, actionable product insights for the product team.

For each business domain below, examine ONLY the user reviews provided in the message and return:
- `summary`: 1-2 sentence neutral observation of what users are saying in this domain (empty string if no signal).
- `key_issues`: specific, concrete problems users raised (max 5, each <= 12 words). Empty list if no issues.
- `recommendation`: ONE specific action the product team should take. Null if severity is "none".
- `severity`: "none" | "low" | "medium" | "high" based on frequency and intensity in the sample.
- `representative_quotes`: 1-3 short verbatim quotes (each <= 25 words) drawn from the reviews. Empty if severity is "none".

Domains (use these EXACT names, in this order):
1. "Design & UX" — visual design, icon/logo changes, redesigns, navigation, clarity.
2. "Pricing & Monetization" — subscriptions, ads, paywall, perceived value.
3. "Features" — missing or requested features, feature quality, capability gaps.
4. "Performance & Stability" — crashes, freezes, lag, bugs, errors.
5. "Content & Recommendations" — content available, recommendation quality, library breadth.
6. "Account & Authentication" — login, sign-in, password, account access.
7. "Customer Support" — responsiveness, help articles, support quality.

Rules:
- Return ALL seven domains, in the order above, even if some have severity "none".
- Be CONSERVATIVE: "high" requires multiple reviews showing strong negative reaction.
- Do NOT fabricate quotes — every quote must appear verbatim in the input reviews.
- Do NOT score domains that aren't actually represented in the reviews — set severity "none" and leave fields empty.
- Recommendations must be concrete and product-team-actionable ("Reduce mid-session ad frequency" — not "improve ads").
"""


def is_enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def current_model() -> str:
    return DEFAULT_MODEL


def _format_review(idx: int, review: Review) -> str:
    stars = "★" * review.rating + "☆" * (5 - review.rating)
    body = (review.text or "").strip().replace("\n", " ")
    if len(body) > MAX_REVIEW_CHARS:
        body = body[:MAX_REVIEW_CHARS].rstrip() + "…"
    title = (review.title or "").strip().replace("\n", " ")
    country_tag = f" [{review.country.upper()}]" if review.country else ""
    return f"{idx}. {stars}{country_tag} \"{title}\"\n   {body}"


def _build_user_message(reviews: list[Review], app_name: str, scope_label: str) -> str:
    bodies = "\n".join(_format_review(i + 1, r) for i, r in enumerate(reviews))
    return (
        f"App: {app_name}\n"
        f"Sample scope: {scope_label}\n"
        f"Reviews ({len(reviews)} total):\n\n"
        f"{bodies}\n\n"
        "Return your analysis using the required schema, covering all seven domains "
        "in order. Be conservative — score severity 'none' for any domain not "
        "actually represented in this sample."
    )


def generate_domain_insights(
    reviews: Iterable[Review],
    app_name: str,
    scope_label: str = "all storefronts",
    *,
    model: Optional[str] = None,
) -> list[DomainInsight]:
    if not is_enabled():
        return []

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed; skipping LLM insights")
        return []

    reviews = list(reviews)[:MAX_REVIEWS_PER_CALL]
    if not reviews:
        return []

    chosen_model = model or DEFAULT_MODEL
    client = anthropic.Anthropic()

    system_blocks = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    try:
        response = client.messages.parse(
            model=chosen_model,
            max_tokens=8192,
            system=system_blocks,
            messages=[{"role": "user", "content": _build_user_message(reviews, app_name, scope_label)}],
            output_format=_LLMResponse,
            **_call_kwargs(chosen_model),
        )
    except anthropic.APIError as exc:
        logger.warning("Claude API call failed (%s); falling back to rule-based insights", exc)
        return []
    except Exception as exc:  # noqa: BLE001 — never let LLM break the API
        logger.exception("Unexpected error calling Claude (%s); skipping LLM insights", exc)
        return []

    try:
        usage = response.usage
        logger.info(
            "Claude call complete: model=%s, input=%s, output=%s, cache_read=%s, cache_write=%s",
            chosen_model,
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_read_input_tokens", 0),
            getattr(usage, "cache_creation_input_tokens", 0),
        )
    except AttributeError:
        pass

    parsed = response.parsed_output
    if parsed is None:
        logger.warning(
            "Claude returned no parsed output (stop_reason=%s); skipping LLM insights",
            getattr(response, "stop_reason", None),
        )
        return []

    return [
        DomainInsight(
            domain=item.domain,
            summary=item.summary,
            key_issues=list(item.key_issues),
            recommendation=item.recommendation,
            severity=item.severity,
            representative_quotes=list(item.representative_quotes),
        )
        for item in parsed.domains
    ]


# Multilingual sentiment classification

_SENTIMENT_SYSTEM_PROMPT = """\
You classify Apple App Store reviews into one of three sentiment categories:
positive, negative, or neutral.

For each input review, return:
- review_index: the 1-based index of the review in the input.
- sentiment: "positive" | "negative" | "neutral"

Rules:
- Judge based on the TEXT content. The star rating is shown for context but is
  NOT the sentiment — many reviews disagree with their own rating.
- Detect language automatically and judge sentiment in the review's own
  language. The input includes reviews in many languages, not just English.
- "positive" — clear enthusiasm, appreciation, or recommendation.
- "negative" — clear complaint, frustration, disappointment, or anger.
- "neutral" — factual, descriptive, mixed praise-and-complaint, or genuinely
  ambivalent ("it's okay", "works as expected").
- Be conservative: when the text doesn't lean strongly, choose "neutral".
- Return exactly one item per input review.
"""


_SentimentLabel = Literal["positive", "negative", "neutral"]


class _SentimentItem(BaseModel):
    review_index: int = Field(description="1-based index of the review in the input.")
    sentiment: _SentimentLabel


class _SentimentBulkResponse(BaseModel):
    items: list[_SentimentItem]


def _format_sentiment_review(idx: int, review: Review) -> str:
    stars = "★" * review.rating + "☆" * (5 - review.rating)
    body = (review.text or "").strip().replace("\n", " ")
    if len(body) > 400:
        body = body[:400].rstrip() + "…"
    title = (review.title or "").strip().replace("\n", " ")
    return f"{idx}. {stars} \"{title}\" — {body}".strip()


def classify_sentiment_bulk(
    reviews: Iterable[Review],
    *,
    model: Optional[str] = None,
) -> list[_SentimentLabel]:
    """Bulk multilingual sentiment classification via Haiku 4.5"""
    reviews = list(reviews)
    if not reviews:
        return []
    if not is_enabled():
        return []

    try:
        import anthropic
    except ImportError:
        return []

    chosen_model = model or DEFAULT_SENTIMENT_MODEL
    client = anthropic.Anthropic()

    user_msg = (
        f"Classify each of the following {len(reviews)} reviews. "
        "Reviews are numbered 1.." + str(len(reviews)) + ".\n\n"
        + "\n".join(_format_sentiment_review(i + 1, r) for i, r in enumerate(reviews))
        + "\n\nReturn one item per input review with its 1-based index and sentiment."
    )

    try:
        response = client.messages.parse(
            model=chosen_model,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": _SENTIMENT_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_msg}],
            output_format=_SentimentBulkResponse,
            **_call_kwargs(chosen_model),
        )
    except anthropic.APIError as exc:
        logger.warning(
            "Bulk sentiment call failed (%s); leaving existing labels in place", exc
        )
        return []
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in classify_sentiment_bulk (%s)", exc)
        return []

    parsed = response.parsed_output
    if parsed is None or not parsed.items:
        logger.warning(
            "Bulk sentiment call produced no parsed output (stop_reason=%s)",
            getattr(response, "stop_reason", None),
        )
        return []

    labels: list[_SentimentLabel] = ["neutral"] * len(reviews)
    for item in parsed.items:
        if 1 <= item.review_index <= len(reviews):
            labels[item.review_index - 1] = item.sentiment

    try:
        usage = response.usage
        logger.info(
            "Haiku sentiment call: model=%s, n=%d, input=%s, output=%s, "
            "cache_read=%s, cache_write=%s",
            chosen_model,
            len(reviews),
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_read_input_tokens", 0),
            getattr(usage, "cache_creation_input_tokens", 0),
        )
    except AttributeError:
        pass

    return labels

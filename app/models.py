from typing import Optional

from pydantic import BaseModel, Field


class Review(BaseModel):
    review_id: str
    author: str
    title: str
    text: str
    rating: int = Field(ge=1, le=5)
    version: Optional[str] = None
    updated: Optional[str] = None
    country: Optional[str] = Field(
        default=None,
        description="Two-letter storefront code the review was collected from.",
    )


class AppInfo(BaseModel):
    app_id: str
    name: str
    country: str = Field(
        description=(
            "Two-letter storefront code the reviews came from, or the sentinel "
            "'all' when reviews were pooled across every Apple storefront."
        )
    )
    artist: Optional[str] = None


class ReviewsResponse(BaseModel):
    app: AppInfo
    reviews: list[Review]


class RatingDistribution(BaseModel):
    counts: dict[int, int]
    percentages: dict[int, float]


class Metrics(BaseModel):
    total_reviews: int
    average_rating: float
    median_rating: float
    rating_distribution: RatingDistribution


class SentimentBreakdown(BaseModel):
    counts: dict[str, int]
    percentages: dict[str, float]
    average_compound: float


class DomainInsight(BaseModel):
    domain: str = Field(
        description=(
            "Stable taxonomy slot. One of: 'Design & UX', 'Pricing & Monetization', "
            "'Features', 'Performance & Stability', 'Content & Recommendations', "
            "'Account & Authentication', 'Customer Support'."
        )
    )
    summary: str
    key_issues: list[str] = []
    recommendation: Optional[str] = None
    severity: str = Field(description="'none' | 'low' | 'medium' | 'high'")
    representative_quotes: list[str] = []


class RegionalSummary(BaseModel):
    region: str
    countries: list[str]
    review_count: int
    average_rating: float
    sentiment_percentages: dict[str, float]
    top_negative_keywords: list[str] = []
    domain_insights: list[DomainInsight] = []
    llm_model: Optional[str] = Field(
        default=None,
        description="Claude model that produced this region's domain_insights, or null if no LLM call was made for this region.",
    )


class Insights(BaseModel):
    sentiment: SentimentBreakdown
    top_negative_keywords: list[str]
    top_positive_keywords: list[str]
    actionable_insights: list[str]
    domain_insights: list[DomainInsight] = Field(
        default_factory=list,
        description=(
            "Domain-organized insights produced by Claude. Empty when "
            "ANTHROPIC_API_KEY is not configured or the LLM call fails."
        ),
    )
    regional_breakdown: list[RegionalSummary] = Field(
        default_factory=list,
        description=(
            "Per-region rollup. Populated only when reviews were collected with "
            "country='all' (i.e. pooled across every storefront)."
        ),
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Claude model that produced domain_insights, or null if LLM was not used.",
    )


class FullReport(BaseModel):
    app: AppInfo
    metrics: Metrics
    insights: Insights

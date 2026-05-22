from __future__ import annotations

from statistics import mean, median
from typing import Iterable

from .models import Metrics, RatingDistribution, Review


def compute_metrics(reviews: Iterable[Review]) -> Metrics:
    reviews = list(reviews)
    if not reviews:
        raise ValueError("Cannot compute metrics over an empty review set")

    ratings = [r.rating for r in reviews]
    counts = {star: 0 for star in range(1, 6)}
    for r in ratings:
        counts[r] += 1

    total = len(ratings)
    percentages = {star: round(100 * counts[star] / total, 2) for star in counts}

    return Metrics(
        total_reviews=total,
        average_rating=round(mean(ratings), 3),
        median_rating=float(median(ratings)),
        rating_distribution=RatingDistribution(counts=counts, percentages=percentages),
    )

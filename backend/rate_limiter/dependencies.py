from __future__ import annotations

"""
FastAPI dependency bridge for feature-based rate limiting.

Responsibilities:
- detect whether the caller is anonymous or authenticated
- map each FeatureType to the correct light/heavy limiter
- keep router modules thin and free of tier-selection logic

Notes:
- anonymous users use anonymous light/heavy wrappers
- authenticated users currently map to the authenticated-free tier
- authenticated-paid is intentionally not handled yet
"""

from collections.abc import Callable
from fastapi import Depends, Request

from backend.auth0_dependencies import AuthenticatedUser, get_current_user_optional

from src.schema import FeatureType
from backend.rate_limiter.anonymous.light import rate_limit_anonymous_light
from backend.rate_limiter.anonymous.heavy import rate_limit_anonymous_heavy
from backend.rate_limiter.authenticated_free.light import rate_limit_authenticated_free_light
from backend.rate_limiter.authenticated_free.heavy import rate_limit_authenticated_free_heavy
from backend.rate_limiter.shared import LIGHT_FEATURES, HEAVY_FEATURES

def rate_limit_for_feature(feature: FeatureType) -> Callable[..., None]:
    """
    Build a FastAPI dependency that enforces the correct rate limiter for a feature.

    Usage:
        @router.post(
            "/summarize",
            dependencies=[Depends(rate_limit_for_feature(FeatureType.summarize))]
        )
        async def summarize_route(...):
            ...
    """

    def dependency(
        request: Request,
        current_user: AuthenticatedUser | None = Depends(get_current_user_optional),
    ) -> None:
        if feature in LIGHT_FEATURES:
            if current_user is None:
                rate_limit_anonymous_light(
                    request=request,
                    feature=feature,
                )
            else:
                rate_limit_authenticated_free_light(
                    request=request,
                    user_id=current_user.user_id,
                    feature=feature,
                )
            return

        if feature in HEAVY_FEATURES:
            if current_user is None:
                rate_limit_anonymous_heavy(
                    request=request,
                    feature=feature,
                )
            else:
                rate_limit_authenticated_free_heavy(
                    request=request,
                    user_id=current_user.user_id,
                    feature=feature,
                )
            return

        raise ValueError(f"Unsupported feature for rate limiting: {feature}")

    return dependency


__all__ = ["rate_limit_for_feature"]
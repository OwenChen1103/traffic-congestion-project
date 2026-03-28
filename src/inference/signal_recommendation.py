"""
signal_recommendation.py
------------------------
Rule-based signal timing recommendation module.
Takes a predicted congestion level and returns a recommended action.

This is a PROTOTYPE decision-support module — not an optimised control system.
It is intentionally simple and rule-based to remain within project scope.

Usage:
    from src.inference.signal_recommendation import recommend

    result = recommend("high")
    print(result)
"""

from dataclasses import dataclass


# Default signal timings (seconds) per phase
# These are illustrative values — can be tuned or loaded from config
BASE_GREEN_DURATION = 30     # seconds
BASE_RED_DURATION = 30       # seconds


@dataclass
class SignalRecommendation:
    congestion_level: str
    recommended_green_extension: int   # additional seconds to add to green
    action_label: str
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "congestion_level": self.congestion_level,
            "recommended_green_extension_sec": self.recommended_green_extension,
            "action_label": self.action_label,
            "reasoning": self.reasoning,
        }

    def __str__(self) -> str:
        return (
            f"Congestion: {self.congestion_level.upper()}\n"
            f"Action:     {self.action_label}\n"
            f"Green ext.: +{self.recommended_green_extension}s\n"
            f"Reasoning:  {self.reasoning}"
        )


# Rule table
_RULES: dict[str, SignalRecommendation] = {
    "low": SignalRecommendation(
        congestion_level="low",
        recommended_green_extension=0,
        action_label="Maintain standard timing",
        reasoning="Traffic is flowing freely. No adjustment needed.",
    ),
    "medium": SignalRecommendation(
        congestion_level="medium",
        recommended_green_extension=10,
        action_label="Slightly extend green phase",
        reasoning="Moderate congestion detected. Extend green by ~10s to clear queued vehicles.",
    ),
    "high": SignalRecommendation(
        congestion_level="high",
        recommended_green_extension=20,
        action_label="Significantly extend green phase",
        reasoning="High congestion detected. Extend green by ~20s to prioritise the congested approach.",
    ),
}


def recommend(congestion_level: str) -> SignalRecommendation:
    """
    Return a signal timing recommendation for a given congestion level.

    Args:
        congestion_level: One of 'low', 'medium', 'high'

    Returns:
        SignalRecommendation dataclass

    Raises:
        ValueError if congestion_level is not recognised
    """
    key = congestion_level.lower().strip()
    if key not in _RULES:
        raise ValueError(
            f"Unknown congestion level '{congestion_level}'. "
            f"Expected one of: {list(_RULES.keys())}"
        )
    return _RULES[key]

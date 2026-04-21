from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SearchConfig:
    coarse_interval_sec: float = 0.2
    refine_window_sec: float = 3.0
    top_n: int = 10
    max_coarse_samples: int = 50000
    use_keyframes: bool = True
    max_refine_regions: int = 40



@dataclass(slots=True)
class Candidate:
    timestamp_seconds: float
    score: float
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MatchResult:
    timestamp_seconds: float
    timestamp_human: str
    confidence: float
    method_used: str
    notes: str = ""
    top_matches: list[dict[str, Any]] = field(default_factory=list)

    def as_json_dict(self) -> dict[str, Any]:
        payload = {
            "timestamp_seconds": float(self.timestamp_seconds),
            "timestamp_human": self.timestamp_human,
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "method_used": self.method_used,
            "notes": self.notes,
        }
        if self.top_matches:
            payload["top_matches"] = self.top_matches
        return payload

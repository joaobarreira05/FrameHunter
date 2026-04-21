"""FrameHunter: robust frame-to-video timestamp finder."""

from .models import MatchResult, SearchConfig
from .search import FrameHunter

__all__ = ["FrameHunter", "MatchResult", "SearchConfig"]

from framehunter.models import Candidate
from framehunter.search import _select_diverse_candidates


def test_select_diverse_candidates_prefers_time_gap():
    candidates = [
        Candidate(timestamp_seconds=10.0, score=0.9, method="hybrid"),
        Candidate(timestamp_seconds=11.0, score=0.89, method="hybrid"),
        Candidate(timestamp_seconds=40.0, score=0.8, method="hybrid"),
        Candidate(timestamp_seconds=80.0, score=0.7, method="hybrid"),
    ]

    selected = _select_diverse_candidates(candidates, max_count=3, min_gap_sec=5.0)

    assert len(selected) == 3
    assert selected[0].timestamp_seconds == 10.0
    assert selected[1].timestamp_seconds == 40.0
    assert selected[2].timestamp_seconds == 80.0

import heapq
import multiprocessing as mp
import os
import sys
from collections.abc import Callable
from functools import partial

import cv2
import numpy as np

from .models import Candidate, MatchResult, SearchConfig
from .similarity import HybridMatcher
from .utils import format_timestamp, load_image_bgr, resize_keep_aspect
from .video_decoder import VideoDecoder, get_keyframe_timestamps


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_s, prev_e = merged[-1]
        if start <= prev_e:
            merged[-1] = (prev_s, max(prev_e, end))
        else:
            merged.append((start, end))
    return merged


def _select_diverse_candidates(
    candidates: list[Candidate],
    max_count: int,
    min_gap_sec: float,
) -> list[Candidate]:
    selected: list[Candidate] = []
    for cand in candidates:
        if all(abs(cand.timestamp_seconds - s.timestamp_seconds) >= min_gap_sec for s in selected):
            selected.append(cand)
            if len(selected) >= max_count:
                break
    if len(selected) < max_count:
        for cand in candidates:
            if cand in selected: continue
            selected.append(cand)
            if len(selected) >= max_count: break
    return selected


def _coarse_worker(
    timestamps: list[float],
    image_path: str,
    video_path: str,
    fast_mode: bool,
) -> list[Candidate]:
    # Disable OpenCV's internal threading to avoid conflict with multiprocessing
    cv2.setNumThreads(0)
    
    ref = resize_keep_aspect(load_image_bgr(image_path), max_side=800)
    matcher = HybridMatcher(ref, max_side=800, fast_mode=fast_mode)
    decoder = VideoDecoder(video_path)
    
    results = []
    for ts in timestamps:
        frame = decoder.get_frame_at_time(ts)
        if frame is None:
            continue
        sim = matcher.compare(frame)
        cand = Candidate(
            timestamp_seconds=float(ts),
            score=sim.score,
            method="hybrid",
            diagnostics={
                "stage": "coarse",
                "sift": sim.sift_score,
                "ssim": sim.ssim_score,
                "hist": sim.hist_score,
                "phash": sim.phash_score,
                "tmpl": sim.tmpl_score,
                "edge": sim.edge_score,
            },
        )
        results.append(cand)
    return results


class FrameHunter:
    def __init__(self, config: SearchConfig | None = None):
        self.config = config or SearchConfig()

    def _build_coarse_timestamps(self, decoder: VideoDecoder) -> list[float]:
        duration = decoder.info.duration_seconds
        if duration <= 0:
            return [0.0]

        uniform = list(np.arange(0.0, duration, self.config.coarse_interval_sec, dtype=np.float64))
        points = set(float(x) for x in uniform)

        if self.config.use_keyframes:
            kfs = get_keyframe_timestamps(decoder.video_path, max_count=self.config.max_coarse_samples)
            points.update(kfs)

        timestamps = sorted(points)
        if len(timestamps) > self.config.max_coarse_samples:
            step = len(timestamps) / float(self.config.max_coarse_samples)
            sampled = []
            i = 0.0
            while int(i) < len(timestamps):
                sampled.append(timestamps[int(i)])
                i += step
            timestamps = sampled

        return timestamps

    @staticmethod
    def _push_candidate(heap: list[tuple[float, int, Candidate]], cand: Candidate, keep: int):
        # Use an integer tie-breaker to avoid comparing Candidate objects on equal scores.
        item = (cand.score, id(cand), cand)
        if len(heap) < keep:
            heapq.heappush(heap, item)
            return
        if cand.score > heap[0][0]:
            heapq.heapreplace(heap, item)

    def search(
        self,
        image_path: str,
        video_path: str,
        top_n: int | None = None,
        workers: int | None = None,
        fast_mode_coarse: bool = False,
        progress_callback: Callable[[str, int, int], None] | None = None,
        live_callback: Callable[[MatchResult], None] | None = None,
    ) -> MatchResult:
        top_n = top_n if top_n is not None else self.config.top_n
        if workers is None:
            workers = max(1, os.cpu_count() - 1)

        coarse_points = self._build_coarse_timestamps(VideoDecoder(video_path))
        coarse_total = len(coarse_points)
        candidate_heap: list[tuple[float, int, Candidate]] = []
        best_overall: Candidate | None = None

        def update_best(new_cand: Candidate):
            nonlocal best_overall
            if best_overall is None or new_cand.score > best_overall.score:
                best_overall = new_cand
                if live_callback:
                    # Construct a temporary MatchResult for the live update
                    res = MatchResult(
                        timestamp_seconds=new_cand.timestamp_seconds,
                        timestamp_human=format_timestamp(new_cand.timestamp_seconds),
                        confidence=new_cand.score,
                        method_used=new_cand.method,
                        notes=f"Live update from {new_cand.diagnostics['stage']} stage",
                        top_matches=[],
                    )
                    live_callback(res)

        if progress_callback:
            progress_callback("coarse", 0, max(1, coarse_total))

        if workers > 1 and coarse_total > workers:
            chunk_size = max(1, coarse_total // (workers * 4))
            chunks = [coarse_points[i:i + chunk_size] for i in range(0, coarse_total, chunk_size)]
            worker_fn = partial(_coarse_worker, image_path=image_path, video_path=video_path, fast_mode=fast_mode_coarse)
            
            processed_count = 0
            with mp.Pool(processes=workers) as pool:
                for chunk_results in pool.imap_unordered(worker_fn, chunks):
                    for cand in chunk_results:
                        self._push_candidate(candidate_heap, cand, keep=max(30, top_n * 10))
                        update_best(cand)
                    
                    processed_count += len(chunk_results)
                    if progress_callback:
                        progress_callback("coarse", min(processed_count, coarse_total), max(1, coarse_total))
        else:
            ref = resize_keep_aspect(load_image_bgr(image_path), max_side=800)
            matcher = HybridMatcher(ref, max_side=800, fast_mode=fast_mode_coarse)
            decoder = VideoDecoder(video_path)
            for i, ts in enumerate(coarse_points, start=1):
                frame = decoder.get_frame_at_time(ts)
                if frame is not None:
                    sim = matcher.compare(frame)
                    cand = Candidate(timestamp_seconds=float(ts), score=sim.score, method="hybrid", diagnostics={"stage": "coarse", "sift": sim.sift_score, "ssim": sim.ssim_score, "hist": sim.hist_score, "phash": sim.phash_score, "tmpl": sim.tmpl_score, "edge": sim.edge_score})
                    self._push_candidate(candidate_heap, cand, keep=max(30, top_n * 10))
                    update_best(cand)
                if progress_callback: progress_callback("coarse", i, coarse_total)

        coarse_best = [c for _, _, c in sorted(candidate_heap, key=lambda x: x[0], reverse=True)]
        if not coarse_best:
            if progress_callback: progress_callback("done", 1, 1)
            return MatchResult(0.0, format_timestamp(0.0), 0.0, "hybrid", "No readable frames found.", [])

        ref = resize_keep_aspect(load_image_bgr(image_path), max_side=800)
        matcher = HybridMatcher(ref, max_side=800, fast_mode=False)
        decoder = VideoDecoder(video_path)
        duration = decoder.info.duration_seconds
        
        intervals = []
        diverse_coarse = _select_diverse_candidates(coarse_best, max_count=self.config.max_refine_regions, min_gap_sec=max(1.0, self.config.refine_window_sec))
        for cand in diverse_coarse:
            s = max(0.0, cand.timestamp_seconds - self.config.refine_window_sec)
            e = min(duration, cand.timestamp_seconds + self.config.refine_window_sec)
            intervals.append((s, e))

        intervals = _merge_intervals(intervals)
        fps = max(decoder.info.fps, 1.0)
        fine_total = sum(max(0, int((end - start) * fps) + 1) for start, end in intervals)
        fine_done = 0

        if progress_callback:
            progress_callback("fine", 0, max(1, fine_total))

        for start, end in intervals:
            for ts, frame in decoder.iter_frames_between(start, end, frame_stride=1):
                fine_done += 1
                sim = matcher.compare(frame)
                cand = Candidate(
                    timestamp_seconds=float(ts),
                    score=sim.score,
                    method="hybrid",
                    diagnostics={
                        "stage": "fine", "sift": sim.sift_score, "ssim": sim.ssim_score, "hist": sim.hist_score, "phash": sim.phash_score, "tmpl": sim.tmpl_score, "edge": sim.edge_score, "fps": fps,
                    },
                )
                self._push_candidate(candidate_heap, cand, keep=max(100, top_n * 20))
                update_best(cand)

                if progress_callback and (fine_done % 10 == 0 or fine_done >= fine_total):
                    progress_callback("fine", fine_done, max(1, fine_total))

        ranked = [c for _, _, c in sorted(candidate_heap, key=lambda x: x[0], reverse=True)]
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        margin = max(0.0, best.score - second.score) if second else best.score
        confidence = float(np.clip(0.75 * best.score + 0.25 * margin, 0.0, 1.0))

        top_matches = []
        seen = set()
        for c in ranked:
            t_rounded = round(c.timestamp_seconds, 3)
            if t_rounded in seen: continue
            seen.add(t_rounded)
            top_matches.append({"timestamp_seconds": c.timestamp_seconds, "timestamp_human": format_timestamp(c.timestamp_seconds), "confidence": c.score, "method_used": c.method, "diagnostics": c.diagnostics})
            if len(top_matches) >= top_n: break

        result = MatchResult(best.timestamp_seconds, format_timestamp(best.timestamp_seconds), confidence, best.method, f"coarse-to-fine parallel (workers={workers})", top_matches)
        if progress_callback: progress_callback("done", 1, 1)
        return result

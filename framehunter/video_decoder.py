from __future__ import annotations

import subprocess
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class VideoInfo:
    fps: float
    frame_count: int
    duration_seconds: float


class VideoDecoder:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._info = self._probe_video_info()

    @property
    def info(self) -> VideoInfo:
        return self._info

    def _probe_video_info(self) -> VideoInfo:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0

        duration = frame_count / fps if frame_count > 0 else 0.0
        cap.release()
        return VideoInfo(fps=float(fps), frame_count=frame_count, duration_seconds=float(duration))

    def get_frame_at_time(self, timestamp_seconds: float) -> np.ndarray | None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_seconds) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None

    def get_frame_at_index(self, frame_index: int) -> tuple[float, np.ndarray] | None:
        frame_index = max(0, min(frame_index, max(0, self._info.frame_count - 1)))
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        ts = frame_index / self._info.fps
        return ts, frame

    def iter_frames_between(self, start_sec: float, end_sec: float, frame_stride: int = 1):
        if frame_stride < 1:
            frame_stride = 1

        fps = self._info.fps
        start_idx = int(max(0.0, start_sec) * fps)
        end_idx = int(max(0.0, end_sec) * fps)
        if end_idx < start_idx:
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        idx = start_idx

        while idx <= end_idx:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start_idx) % frame_stride == 0:
                yield (idx / fps, frame)
            idx += 1

        cap.release()


def get_keyframe_timestamps(video_path: str, max_count: int | None = None) -> list[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-skip_frame",
        "nokey",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return []

    if proc.returncode != 0:
        return []

    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(float(line))
        except ValueError:
            continue

    out = sorted(set(out))
    if max_count is not None and max_count > 0 and len(out) > max_count:
        step = len(out) / float(max_count)
        sampled = []
        i = 0.0
        while int(i) < len(out):
            sampled.append(out[int(i)])
            i += step
        out = sampled

    return out

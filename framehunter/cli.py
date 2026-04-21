from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2

from .models import SearchConfig
from .search import FrameHunter
from .utils import load_image_bgr, resize_keep_aspect
from .video_decoder import VideoDecoder


class _ProgressRenderer:
    def __init__(self):
        self.start = time.time()

    def __call__(self, stage: str, current: int, total: int) -> None:
        total = max(1, total)
        current = max(0, min(current, total))
        elapsed = time.time() - self.start

        if stage == "done":
            print(f"\r[done] 100.0% elapsed={elapsed:.1f}s" + " " * 20, file=sys.stderr)
            return

        pct = (current / total) * 100.0
        print(
            f"\r[{stage:6}] {pct:6.2f}% ({current}/{total}) elapsed={elapsed:.1f}s",
            end="",
            file=sys.stderr,
            flush=True,
        )


def _save_visualization(image_path: str, video_path: str, ts: float, out_path: str) -> None:
    ref = resize_keep_aspect(load_image_bgr(image_path), max_side=720)
    decoder = VideoDecoder(video_path)
    frame = decoder.get_frame_at_time(ts)
    if frame is None:
        return

    frame = resize_keep_aspect(frame, max_side=720)

    # Match geometry for side-by-side display.
    h = max(ref.shape[0], frame.shape[0])
    if ref.shape[0] != h:
        ref = cv2.resize(ref, (int(ref.shape[1] * h / ref.shape[0]), h), interpolation=cv2.INTER_AREA)
    if frame.shape[0] != h:
        frame = cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h), interpolation=cv2.INTER_AREA)

    canvas = cv2.hconcat([ref, frame])
    cv2.putText(canvas, "Reference", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Best Match",
        (ref.shape[1] + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(out_path, canvas)


def _export_top_frames(video_path: str, top_matches: list[dict], out_dir: str) -> list[str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    decoder = VideoDecoder(video_path)
    exported: list[str] = []

    for index, match in enumerate(top_matches, start=1):
        ts = float(match.get("timestamp_seconds", 0.0))
        conf = float(match.get("confidence", 0.0))
        frame = decoder.get_frame_at_time(ts)
        if frame is None:
            continue

        ts_label = match.get("timestamp_human", "00:00:00.000").replace(":", "-")
        file_name = f"rank_{index:02d}_{ts_label}_{conf:.4f}.jpg"
        file_path = out_path / file_name
        ok = cv2.imwrite(str(file_path), frame)
        if ok:
            exported.append(str(file_path))

    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="framehunter",
        description="Find the exact/closest timestamp in a video where an image frame appears.",
    )
    parser.add_argument("--image", required=True, help="Path to reference image (PNG/JPG/BMP/...) ")
    parser.add_argument("--video", required=True, help="Path to video file (MP4 and others supported by OpenCV/FFmpeg)")
    parser.add_argument("--top-n", type=int, default=5, help="Return top-N closest matches")
    parser.add_argument("--coarse-interval", type=float, default=2.0, help="Coarse scan interval in seconds")
    parser.add_argument("--refine-window", type=float, default=3.0, help="Fine search window around candidates (seconds)")
    parser.add_argument("--max-coarse", type=int, default=5000, help="Cap on coarse sample points")
    parser.add_argument("--no-keyframes", action="store_true", help="Disable keyframe-assisted coarse scan")
    parser.add_argument("--no-progress", action="store_true", help="Disable live CLI progress output")
    parser.add_argument("--visualize", help="Optional output image path for side-by-side reference vs best match")
    parser.add_argument(
        "--export-top-frames-dir",
        help="Optional directory to export image files for the returned top-N matches",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = SearchConfig(
        coarse_interval_sec=max(0.1, args.coarse_interval),
        refine_window_sec=max(0.5, args.refine_window),
        top_n=max(1, args.top_n),
        max_coarse_samples=max(100, args.max_coarse),
        use_keyframes=not args.no_keyframes,
    )

    hunter = FrameHunter(config=config)
    progress_cb = None if args.no_progress else _ProgressRenderer()
    result = hunter.search(args.image, args.video, top_n=config.top_n, progress_callback=progress_cb)

    if args.visualize:
        _save_visualization(args.image, args.video, result.timestamp_seconds, args.visualize)

    payload = result.as_json_dict()
    if args.export_top_frames_dir and result.top_matches:
        exported = _export_top_frames(args.video, result.top_matches, args.export_top_frames_dir)
        payload["exported_top_frames"] = exported

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

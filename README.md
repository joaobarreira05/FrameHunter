# FrameHunter

High-performance frame-to-video timestamp finder for CTF and forensic workflows.

Given a reference image and a video, FrameHunter finds the exact or closest timestamp where that frame appears.

## Features

- Coarse-to-fine search strategy for speed and precision.
- Keyframe-aware coarse scan (via `ffprobe`) when available.
- Hybrid similarity model:
	- ORB feature matching (robust to compression, moderate scale/color changes)
	- SSIM (structural similarity)
	- HSV histogram correlation (color distribution sanity check)
- Handles approximate matches (compression, minor noise, color shifts).
- Returns top-N matches with confidence and diagnostics.
- Optional side-by-side visualization output.

## Architecture

- `framehunter/video_decoder.py`: video probing, frame access, keyframe timestamp extraction.
- `framehunter/similarity.py`: hybrid frame similarity scoring.
- `framehunter/search.py`: coarse -> refine search orchestration.
- `framehunter/cli.py`: CLI and JSON output.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Dependencies:

- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- Optional but recommended: FFmpeg/ffprobe in PATH for keyframe scanning

## Usage

```bash
python -m framehunter --image frame.png --video video.mp4
```

or (after `pip install -e .`):

```bash
framehunter --image frame.png --video video.mp4
```

Top-N matches + tuned scanning:

```bash
python -m framehunter \
	--image frame.png \
	--video video.mp4 \
	--top-n 10 \
	--coarse-interval 1.5 \
	--refine-window 4.0
```

Disable keyframe assistance:

```bash
python -m framehunter --image frame.png --video video.mp4 --no-keyframes
```

Disable live progress output:

```bash
python -m framehunter --image frame.png --video video.mp4 --no-progress
```

Generate side-by-side match image:

```bash
python -m framehunter \
	--image frame.png \
	--video video.mp4 \
	--visualize match_preview.jpg
```

## Docker (Cross-Platform)

Run the tool the same way on macOS, Linux, or Windows (with Docker Desktop/Engine).

Build image:

```bash
docker build -t framehunter:latest .
```

Run with host folder mounted (inputs + outputs):

```bash
docker run --rm \
	-v "$(pwd)/testes:/data" \
	framehunter:latest \
	--image /data/frame.png \
	--video /data/video.mp4 \
	--top-n 5 \
	--visualize /data/match_preview.jpg
```

Windows PowerShell volume syntax example:

```powershell
docker run --rm `
	-v "${PWD}/testes:/data" `
	framehunter:latest `
	--image /data/frame.png `
	--video /data/video.mp4
```

Using Docker Compose:

```bash
docker compose build
docker compose run --rm framehunter --image /data/frame.png --video /data/video.mp4
```

Notes:

- Container includes FFmpeg (`ffprobe`) for keyframe-assisted coarse scan.
- Place files in `testes/` (default compose mount) or adjust the mount path.

## Output Format

FrameHunter prints JSON:

```json
{
	"timestamp_seconds": 123.456,
	"timestamp_human": "00:02:03.456",
	"confidence": 0.87,
	"method_used": "hybrid",
	"notes": "coarse-to-fine hybrid (ORB + SSIM + HSV histogram)",
	"top_matches": [
		{
			"timestamp_seconds": 123.456,
			"timestamp_human": "00:02:03.456",
			"confidence": 0.91,
			"method_used": "hybrid",
			"diagnostics": {
				"stage": "fine",
				"orb": 0.84,
				"ssim": 0.89,
				"hist": 0.86,
				"fps": 29.97
			}
		}
	]
}
```

## Performance Notes

- Coarse pass combines interval sampling and keyframe timestamps.
- Refine pass only decodes around best coarse candidates.
- Keep `--coarse-interval` larger for huge videos; lower it for precision.
- Increase `--refine-window` when the coarse stage may land far from the true frame.

## Edge Cases

- Frame absent: returns best approximate timestamp with low confidence.
- Heavy transformations: ORB may degrade; SSIM + histogram still provide a fallback.
- Variable FPS: timestamp-based probing and frame-index refinement are both used.

## Next Improvements

- CLIP/CNN embedding backend for stronger semantic matching.
- Batch mode for multiple query images.
- Multi-process search workers for long videos.
- Optional GPU-accelerated decode path.

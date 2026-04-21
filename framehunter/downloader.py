from __future__ import annotations

import sys
from pathlib import Path
import yt_dlp


def download_video(url: str, output_dir: str | Path) -> str:
    """
    Downloads a video from a URL (YouTube, etc.) to the output_dir.
    Returns the absolute path to the downloaded MP4 file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # We want the best single-file MP4 format if possible, 
    # or the best video + best audio merged into MP4.
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': True,
        'noprogress': False,
    }

    print(f"[*] Downloading video from: {url}", file=sys.stderr)
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        
        # Ensure the filename extension is corrected if it was merged to mp4
        final_path = Path(filename).with_suffix('.mp4')
        if not final_path.exists() and Path(filename).exists():
            return str(Path(filename).absolute())
            
        return str(final_path.absolute())

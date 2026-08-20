#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from provenance import video_key  # noqa: E402


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def resolve_local(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        print(
            f"[watch] warning: {p.suffix} is not a known video extension, proceeding anyway",
            file=sys.stderr,
        )
    return {
        "video_path": str(p),
        "subtitle_path": None,
        "info": {"title": p.name, "url": str(p)},
        "downloaded": False,
        "video_error": None,
    }


def _pick_subtitle(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    preferred = [c for c in candidates if ".en" in c.name]
    return preferred[0] if preferred else candidates[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def cache_root() -> Path:
    """Where downloaded videos persist between runs.

    Overridable with $WATCH_CACHE_DIR. Chunked step-by-step watching of a long
    video re-invokes /watch many times over the same source; without this the
    same multi-hundred-megabyte file is fetched once per chunk.
    """
    env = os.environ.get("WATCH_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "watch" / "downloads"


def _cache_dir_for(url: str) -> Path:
    return cache_root() / video_key(url)


def _cache_probe(cache_dir: Path) -> dict | None:
    """Return cached artefacts if this directory holds a usable download."""
    if not cache_dir.is_dir():
        return None
    video = _pick_video(cache_dir)
    subtitle = _pick_subtitle(cache_dir)
    if video is None and subtitle is None:
        return None
    if video is not None and video.stat().st_size < 1024:
        return None          # truncated / failed download, do not trust it
    return {"video": video, "subtitle": subtitle}


def cache_size_bytes() -> int:
    root = cache_root()
    if not root.is_dir():
        return 0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


def download_url(url: str, out_dir: Path, use_cache: bool = True) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    # Download into a persistent per-video cache, not the throwaway workdir,
    # so repeated chunked watches of one source fetch the stream exactly once.
    if use_cache:
        out_dir = _cache_dir_for(url)
        hit = _cache_probe(out_dir)
        if hit is not None:
            size = hit['video'].stat().st_size / 1048576 if hit['video'] else 0
            print('[watch] cache hit: reusing %.0f MB download from %s'
                  % (size, out_dir), file=sys.stderr)
            info = {}
            info_path = out_dir / 'video.info.json'
            if info_path.exists():
                try:
                    raw = json.loads(info_path.read_text(encoding='utf-8'))
                    info = {
                        'title': raw.get('title'),
                        'uploader': raw.get('uploader') or raw.get('channel'),
                        'duration': raw.get('duration'),
                        'url': raw.get('webpage_url') or url,
                    }
                except Exception:
                    info = {'url': url}
            return {
                'video_path': str(hit['video']) if hit['video'] else None,
                'subtitle_path': str(hit['subtitle']) if hit['subtitle'] else None,
                'info': info or {'url': url},
                'downloaded': False,
                'cached': True,
                'video_error': None if hit['video'] else 'cached subtitles only',
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    base = [
        "yt-dlp",
        "-N", "8",
        "-f", "bv*[height<=720]+ba/b[height<=720]/bv+ba/b",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en,en-US,en-GB,en-orig",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
    ]

    # yt-dlp may exit non-zero if a subtitle variant fails (e.g. 429) even when
    # the video itself downloaded fine. Treat "video file present" as success.
    result = subprocess.run(base + ["--", url], stdout=sys.stderr, stderr=sys.stderr)
    video = _pick_video(out_dir)

    # YouTube increasingly serves 403 / "confirm you're not a bot" to the default
    # player client while still handing over subtitles. Retry the stream with
    # other clients before giving up on frames.
    if video is None and is_url(url):
        for client in ("android", "web_safari", "ios", "mweb", "tv_embedded"):
            print(f"[watch] stream failed; retrying player_client={client}…", file=sys.stderr)
            subprocess.run(
                base + ["--extractor-args", f"youtube:player_client={client}", "--", url],
                stdout=sys.stderr, stderr=sys.stderr,
            )
            video = _pick_video(out_dir)
            if video is not None:
                print(f"[watch] stream recovered via player_client={client}", file=sys.stderr)
                break

    subtitle = _pick_subtitle(out_dir)

    # Degrade instead of dying: if the stream is unavailable but captions came
    # through, a transcript-only watch is still worth emitting a report for.
    video_error = None
    if video is None:
        if subtitle is None:
            raise SystemExit(
                f"yt-dlp produced neither a video nor subtitles in {out_dir} "
                f"(exit {result.returncode}). If this is age-gated, private, or "
                f"bot-checked, pass cookies via yt-dlp --cookies-from-browser."
            )
        video_error = (
            "video stream unavailable (403 / bot-check / DRM) - captions were "
            "retrieved, continuing transcript-only with no frames"
        )
        print(f"[watch] WARNING: {video_error}", file=sys.stderr)
    info_path = out_dir / "video.info.json"
    info: dict = {}
    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
            info = {
                "title": raw.get("title"),
                "uploader": raw.get("uploader") or raw.get("channel"),
                "duration": raw.get("duration"),
                "url": raw.get("webpage_url") or url,
            }
        except Exception as exc:
            print(f"[watch] info.json parse failed: {exc}", file=sys.stderr)
            info = {"url": url}

    return {
        "video_path": str(video) if video else None,
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": True,
        "video_error": video_error,
    }


def download(source: str, out_dir: Path, use_cache: bool = True) -> dict:
    if is_url(source):
        return download_url(source, out_dir, use_cache=use_cache)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))

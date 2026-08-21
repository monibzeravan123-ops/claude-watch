#!/usr/bin/env python3
"""Provenance and regression guards for /watch.

Exists because of a measured failure: the same video was watched three times,
each pass rewrote the summary from the same transcript, and by the third pass a
performance warning the author stated out loud had been compressed out of the
summary while the recommendation it qualified had become *more* confident. A
build was made from that third summary and cost the user a 103x slowdown.

The warning was never missing from the data. It sat in the transcript dump two
hundred lines below the summary that contradicted it. Nothing compared them.

Four guards, all deterministic (no model in the loop, so they cannot drift):

  1. extract_caveats()      - pull every hedge/warning out of the transcript
  2. check_caveat_coverage() - assert the narrative actually mentions each one
  3. find_prior_watches()   - locate earlier watches of the SAME video
  4. validate_citations()   - every frame_NNNN cited must exist on disk

Guard 1+2 are absolute: they run against the transcript every time and do not
need a prior watch to exist. Guard 3 is relative: it catches things a previous
pass said that this one dropped.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 1. Caveat extraction
# --------------------------------------------------------------------------

# Phrases that mark a claim as qualified, risky, or costly. Deliberately
# tuned for *technical tutorial* speech - the register where "you can, but"
# is doing real work and is exactly what gets lost in summary.
WARNING_MARKERS = [
    r"be careful",
    r"\bcareful\b",
    r"\bbeware\b",
    r"\bcaution\b",
    r"watch out",
    r"\bwarning\b",
    r"\bavoid\b",
    r"\bdo not\b",
    r"kill your machine",
    r"\bcrash(es|ed|ing)?\b",
    r"\bhangs?\b",
    r"memory load",
    r"\bmemory\b.{0,20}\b(load|usage|hog|heavy)\b",
    r"\bexpensive\b",
    r"\bheavy\b",
    r"\bbog(s|ged)?\b",
    r"\bhates?\b",
    r"too much",
    r"\btoo many\b",
    r"\brisky?\b",
    r"\bgotcha\b",
    r"\bpitfall\b",
    r"if you'?re not careful",
    r"at your own risk",
    r"\bkill(s)? your\b",
    r"\bslow(s|er) (it|things|everything|down)\b",
]

# Phrases that mark a step as OPTIONAL. A step promoted from optional to
# mandatory is the same class of failure as a dropped warning.
OPTIONAL_MARKERS = [
    r"if you want", r"\boptional(ly)?\b", r"you don'?t have to",
    r"up to you", r"you could also", r"you can also", r"doesn'?t have to",
    r"feel free", r"i won'?t", r"i'?m not going to", r"that part is up to you",
    r"\bor you can\b", r"\bif you'?d like\b", r"\bplay around\b",
]

_WARN_RE = re.compile("|".join(WARNING_MARKERS), re.I)
_OPT_RE = re.compile("|".join(OPTIONAL_MARKERS), re.I)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "you", "your", "yours", "i",
    "it", "its", "this", "that", "these", "those", "is", "are", "was", "were",
    "be", "been", "to", "of", "in", "on", "at", "for", "with", "as", "so",
    "do", "does", "did", "can", "could", "will", "would", "just", "really",
    "too", "also", "here", "there", "then", "than", "them", "they", "we",
    "my", "me", "not", "no", "yes", "up", "down", "out", "into", "some",
    "what", "when", "where", "which", "who", "how", "all", "any", "more",
    "want", "like", "going", "get", "got", "make", "made", "one", "now",
}


def _fmt_time(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _shingles(text: str, n: int = 3) -> set[str]:
    cw = _content_words(text)
    if len(cw) < n:
        return {" ".join(cw)} if cw else set()
    return {" ".join(cw[i:i + n]) for i in range(len(cw) - n + 1)}


def _dedupe_rolling(segments: list[dict]) -> list[dict]:
    """Undo YouTube's rolling auto-captions.

    Auto-captions repeat the previous line's tail before adding new words:
        [03:49] careful this can really increase the memory load and blender hates that and
        [03:51] memory load and blender hates that and so do kelp forests
    Left alone, that inflates every caveat window with duplicated text and
    makes coverage checks match on the repetition instead of the content.
    """
    out: list[dict] = []
    prev = ""
    for seg in segments:
        text = re.sub(r"\s+", " ", (seg.get("text") or "").strip())
        if not text:
            continue
        if prev:
            lo = min(len(prev), len(text))
            for k in range(lo, 2, -1):
                if prev.endswith(text[:k]):
                    text = text[k:].strip()
                    break
        prev = re.sub(r"\s+", " ", (seg.get("text") or "").strip())
        if text:
            out.append({"start": float(seg.get("start", 0.0)), "text": text})
    return out


def extract_caveats(segments: list[dict], lookahead: int = 3) -> list[dict]:
    """Find every hedged / warning / optional statement in the transcript.

    Each hit is anchored at the segment containing the marker and extends a
    bounded number of segments forward to capture the marker's object - the
    thing being warned about. Windows are bounded and never chain, so one
    caveat stays one caveat instead of merging into an unsearchable blob.
    """
    if not segments:
        return []

    segs = _dedupe_rolling(segments)
    hits: list[dict] = []
    last_end = -1

    for i, seg in enumerate(segs):
        text = seg["text"]
        warn = _WARN_RE.search(text)
        opt = _OPT_RE.search(text)
        if not (warn or opt):
            continue
        if i <= last_end:          # already inside the previous caveat's window
            continue

        hi = min(len(segs), i + lookahead + 1)
        last_end = hi - 1
        # key_text is what coverage is judged on: the marker plus its object,
        # nothing else. full_text is the wider window, for a human to read.
        key_text = " ".join(segs[j]["text"] for j in range(i, min(len(segs), i + 2)))
        full_text = " ".join(segs[j]["text"] for j in range(max(0, i - 1), hi))
        hits.append({
            "start": seg["start"],
            "timestamp": _fmt_time(seg["start"]),
            "kind": "warning" if warn else "optional",
            "marker": (warn or opt).group(0).lower(),
            "key_text": re.sub(r"\s+", " ", key_text).strip(),
            "text": re.sub(r"\s+", " ", full_text).strip(),
        })

    return hits


# --------------------------------------------------------------------------
# 2. Coverage: does the NARRATIVE actually carry each caveat?
# --------------------------------------------------------------------------

def split_narrative(report_text: str) -> str:
    """Everything above the raw Transcript dump.

    A caveat sitting only in the transcript is exactly the failure this
    module exists to catch, so the transcript must not count as coverage.
    """
    for header in ("\n## Transcript", "\n## All frames"):
        idx = report_text.find(header)
        if idx != -1:
            report_text = report_text[:idx]
    return report_text


def _strip_caveats_section(text: str) -> str:
    """Remove the auto-generated caveats block before judging coverage.

    That section quotes every caveat verbatim, so leaving it in would make
    each caveat satisfy its own coverage check and turn guard 2 into a no-op.
    Coverage must be judged on what the *human-written* summary says.
    """
    start = text.find("## Caveats and warnings")
    if start == -1:
        return text
    nxt = text.find(chr(10) + "## ", start + 5)
    return text[:start] + (text[nxt:] if nxt != -1 else "")


def check_caveat_coverage(report_text: str, caveats: list[dict]) -> list[dict]:
    """Return the caveats NOT represented anywhere in the narrative.

    Judged on `key_text` (the marker and its object), not the wider window,
    and requires real overlap: a shared 3-gram, or two distinct uncommon
    words. A single incidental word match is not coverage - that leniency is
    what let a dropped warning pass once already.
    """
    narrative = _strip_caveats_section(split_narrative(report_text)).lower()
    narrative_shingles = _shingles(narrative, 3)
    uncovered: list[dict] = []

    for cav in caveats:
        probe = cav.get("key_text") or cav.get("text", "")
        if cav["timestamp"] in narrative:
            continue
        if _shingles(probe, 3) & narrative_shingles:
            continue
        rare = {w for w in _content_words(probe) if len(w) > 6}
        if len(rare & set(re.findall(r"[a-z']+", narrative))) >= 2:
            continue
        uncovered.append(cav)

    return uncovered


# --------------------------------------------------------------------------
# 3. Prior watches of the same video
# --------------------------------------------------------------------------

def video_key(source: str) -> str:
    """Stable identity for a video across watches.

    YouTube IDs survive URL-shortener / query-param churn, which is what
    made three folders for one video look like three different sources.
    """
    m = re.search(r"(?:youtu\.be/|[?&]v=|/embed/|/shorts/)([A-Za-z0-9_-]{11})", source)
    if m:
        return m.group(1)
    m = re.search(r"(?:vimeo\.com/)(\d+)", source)
    if m:
        return f"vimeo:{m.group(1)}"
    return Path(source).stem.lower()


def resolve_vault() -> Path | None:
    """Locate the Obsidian vault the same way SKILL.md documents.

    $WATCH_VAULT_DIR wins; otherwise a `.obsidian/` marker one level under
    $HOME or $HOME/Documents; otherwise the historical name fallbacks.
    """
    import os
    env = os.environ.get("WATCH_VAULT_DIR", "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    home = Path.home()
    for base in (home, home / "Documents"):
        if not base.is_dir():
            continue
        try:
            for child in base.iterdir():
                if (child / ".obsidian").is_dir():
                    return child
        except OSError:
            continue
    for name in ("SecondBrain", "Second brain", "Second Brain",
                 "Documents/Obsidian", "Obsidian"):
        cand = home / name
        if cand.is_dir():
            return cand
    return None


def _frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    out = {}
    for line in text[3:end].splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def find_prior_watches(vault_dir: Path, source: str, exclude: Path | None = None) -> list[dict]:
    """All earlier report.md files in the vault for the same video."""
    key = video_key(source)
    root = Path(vault_dir) / "raw" / "watched"
    if not root.is_dir():
        return []

    found = []
    for report in sorted(root.glob("*/report.md")):
        if exclude and report.resolve() == Path(exclude).resolve():
            continue
        try:
            text = report.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fm = _frontmatter(text)
        prior_source = fm.get("source", "")
        if not prior_source or video_key(prior_source) != key:
            continue
        found.append({
            "path": str(report),
            "slug": report.parent.name,
            "watched_at": fm.get("watched_at", "?"),
            "intent": fm.get("intent", ""),
            "transcript_source": fm.get("transcript_source", "?"),
            "text": text,
        })
    found.sort(key=lambda d: d["watched_at"])
    return found


def diff_against_prior(new_report_text: str, prior: dict) -> list[dict]:
    """Claims the PRIOR narrative made that the new narrative dropped.

    Only lines carrying a warning/optional marker are compared. A summary
    losing a decorative sentence is fine; losing a hedge is a regression.
    """
    prior_narr = split_narrative(prior["text"])
    new_narr = split_narrative(new_report_text)
    new_shingles = _shingles(new_narr, 3)
    new_lower = new_narr.lower()

    dropped = []
    for line in prior_narr.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "---", "```", "<!--")):
            continue
        if not (_WARN_RE.search(stripped) or _OPT_RE.search(stripped)):
            continue
        if _shingles(stripped, 3) & new_shingles:
            continue
        rare = [w for w in _content_words(stripped) if len(w) > 6]
        if rare and any(w in new_lower for w in rare):
            continue
        dropped.append({"slug": prior["slug"], "line": stripped[:400]})
    return dropped


# --------------------------------------------------------------------------
# 4. Frame citations must be checkable
# --------------------------------------------------------------------------

def validate_citations(report_text: str, frames_dir: Path) -> dict:
    """Every frame_NNNN named in the report must exist in frames_dir."""
    cited = sorted(set(re.findall(r"frame_(\d{4})", report_text)))
    frames_dir = Path(frames_dir)
    on_disk = set()
    if frames_dir.is_dir():
        on_disk = {m.group(1) for f in frames_dir.glob("*.jpg")
                   for m in [re.search(r"frame_(\d{4})", f.name)] if m}
    missing = [c for c in cited if c not in on_disk]
    return {
        "cited": len(cited),
        "on_disk": len(on_disk),
        "missing": missing,
        "ok": not missing,
    }


# --------------------------------------------------------------------------
# Report section rendering (called by report.py)
# --------------------------------------------------------------------------

def render_caveats_section(caveats: list[dict], transcript_source: str | None) -> list[str]:
    lines = ["## Caveats and warnings (verbatim, auto-extracted)", ""]
    if not transcript_source or transcript_source == "none":
        lines += ["_No transcript - caveats could not be extracted. "
                  "Treat every claim in this report as unverified._", ""]
        return lines
    if not caveats:
        lines += ["_None detected in the transcript._", ""]
        return lines

    lines += [
        "**This section is generated from the transcript, not written by Claude, "
        "and must not be edited, shortened, or removed.** Each entry qualifies a "
        "step somewhere above. If a step here is marked optional or risky, it must "
        "read that way in the summary too.",
        "",
    ]
    for c in caveats:
        tag = "⚠️ WARNING" if c["kind"] == "warning" else "○ OPTIONAL"
        lines.append(f"- **[{c['timestamp']}] {tag}** — \"{c['text']}\"")
    lines.append("")
    return lines


def render_provenance_section(
    transcript_source: str | None,
    priors: list[dict],
    frames_kept: int,
    frames_total: int,
) -> list[str]:
    captions_only = (transcript_source or "none") == "captions"
    grade = {
        "captions": "frames + CAPTIONS (machine-generated subtitles)",
        "whisper": "frames + audio transcription",
        "none": "frames only",
    }.get(transcript_source or "none", str(transcript_source))

    lines = ["## Provenance", "", f"- **Evidence grade:** {grade}"]
    if captions_only:
        lines.append(
            "- ⚠️ **Captions are not evidence for visual claims.** Anything in this "
            "report about what is *on screen* (a panel value, a modifier order, a "
            "camera angle) must be traceable to a frame, not to this transcript."
        )
    lines.append(f"- **Frames retained:** {frames_kept} of {frames_total} extracted")
    if frames_kept < frames_total:
        lines.append(
            "- ⚠️ **Not all frames retained** — any citation to a discarded frame "
            "is uncheckable. Either keep them all or cite only what was kept."
        )
    if priors:
        lines.append(f"- **Prior watches of this video:** {len(priors)}")
        for p in priors:
            lines.append(
                f"  - `{p['slug']}` ({p['watched_at'][:10]}, {p['transcript_source']})"
            )
        lines.append(
            "  - ⚠️ **This is a re-watch. Read the prior report(s) before writing the "
            "summary.** A re-watch must *diff*, not replace: if an earlier pass "
            "recorded a caveat this one does not, that is a regression."
        )
    else:
        lines.append("- **Prior watches of this video:** none (first watch)")
    lines.append("")
    lines.append("### Values not stated in the source")
    lines.append("")
    lines.append(
        "<!-- pending Claude fill: list every number/setting you supply that the "
        "video does NOT state, as `- <name>: <value> — INVENTED, not in source`. "
        "Write 'none' only if you introduced no such value. An invented value that "
        "reaches a build unlabelled is the failure this section exists to prevent. -->"
    )
    lines.append("")
    return lines


# --------------------------------------------------------------------------
# CLI: the gate Claude must pass before ingest
# --------------------------------------------------------------------------

def _cmd_check(report_path: Path, frames_dir: Path | None, vault: Path | None) -> int:
    text = report_path.read_text(encoding="utf-8", errors="replace")
    fm = _frontmatter(text)
    problems = 0

    print(f"[provenance] checking {report_path}")

    # a. unfilled markers
    pending = text.count("pending Claude fill")
    if pending:
        print(f"  FAIL  {pending} unfilled '<!-- pending Claude fill -->' marker(s)")
        problems += 1
    else:
        print("  ok    all pending markers filled")

    # b. caveat coverage
    cav_path = report_path.parent / "caveats.json"
    if cav_path.exists():
        caveats = json.loads(cav_path.read_text(encoding="utf-8"))
        uncovered = check_caveat_coverage(text, caveats)
        warns = [c for c in uncovered if c["kind"] == "warning"]
        opts = [c for c in uncovered if c["kind"] != "warning"]
        if warns:
            print(f"  FAIL  {len(warns)} WARNING(s) from the transcript absent from the narrative:")
            for c in warns:
                print(f"          [{c['timestamp']}] \"{(c.get('key_text') or c['text'])[:110]}\"")
            print("          -> a warning the author stated must survive into the summary.")
            problems += 1
        else:
            print(f"  ok    every transcript warning is represented ({len(caveats)} caveat(s) total)")
        if opts:
            print(f"  note  {len(opts)} optional-step hedge(s) not mentioned (advisory, not a failure):")
            for c in opts[:5]:
                print(f"          [{c['timestamp']}] \"{(c.get('key_text') or c['text'])[:90]}\"")
    else:
        print("  warn  no caveats.json beside the report - coverage not checked")

    # c. frame citations
    if frames_dir is None:
        guess = report_path.parent / "frames"
        frames_dir = guess if guess.is_dir() else report_path.parent
    res = validate_citations(text, frames_dir)
    if res["ok"]:
        print(f"  ok    all {res['cited']} cited frame(s) present ({res['on_disk']} on disk)")
    else:
        print(f"  FAIL  {len(res['missing'])} cited frame(s) missing from {frames_dir}:")
        print("          " + ", ".join("frame_" + m for m in res["missing"][:15])
              + (" ..." if len(res["missing"]) > 15 else ""))
        problems += 1

    # d. regression vs prior watches
    if vault and fm.get("source"):
        priors = find_prior_watches(vault, fm["source"], exclude=report_path)
        if priors:
            total = 0
            for p in priors:
                dropped = diff_against_prior(text, p)
                for d in dropped:
                    if total == 0:
                        print("  FAIL  hedges present in a prior watch but absent here:")
                    print(f"          [{d['slug']}] {d['line'][:110]}")
                    total += 1
            if total:
                problems += 1
            else:
                print(f"  ok    no hedges dropped vs {len(priors)} prior watch(es)")
        else:
            print("  ok    first watch of this video - nothing to diff")

    print(f"[provenance] {'PASS' if not problems else 'FAIL (%d category)' % problems}")
    return 0 if not problems else 1


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        print("usage:")
        print("  provenance.py check <report.md> [--frames DIR] [--vault DIR]")
        print("  provenance.py priors <vault-dir> <source-url>")
        return 2

    cmd = argv[1]
    if cmd == "check":
        if len(argv) < 3:
            print("usage: provenance.py check <report.md> [--frames DIR] [--vault DIR]",
                  file=sys.stderr)
            return 2
        report = Path(argv[2]).expanduser().resolve()
        frames = vault = None
        rest = argv[3:]
        for i, a in enumerate(rest):
            if a == "--frames" and i + 1 < len(rest):
                frames = Path(rest[i + 1]).expanduser()
            elif a == "--vault" and i + 1 < len(rest):
                vault = Path(rest[i + 1]).expanduser()
        return _cmd_check(report, frames, vault)

    if cmd == "priors":
        if len(argv) < 4:
            print("usage: provenance.py priors <vault-dir> <source-url>", file=sys.stderr)
            return 2
        priors = find_prior_watches(Path(argv[2]).expanduser(), argv[3])
        if not priors:
            print("(no prior watches of this video)")
            return 0
        for p in priors:
            print(f"{p['watched_at'][:10]}  {p['slug']}  [{p['transcript_source']}]")
            print(f"    intent: {p['intent'][:100]}")
            print(f"    path:   {p['path']}")
        return 0

    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

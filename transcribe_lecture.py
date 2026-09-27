"""
Transcribe Persian lecture audio/video to timestamped Markdown
Uses faster-whisper — good balance of speed & accuracy for Farsi.

Usage:
    py -u transcribe_lecture.py path/to/file.mp3
    py -u transcribe_lecture.py path/to/file.mp3 --model tiny
    py -u transcribe_lecture.py path/to/file.mp3 --preview 180   # test first 3 min
    py -u transcribe_lecture.py path/to/file.mp3 --verbose       # show each word
"""

import argparse, os, sys, subprocess, time
from pathlib import Path


def trim_audio(audio_path: str, seconds: int) -> str:
    """Trim first N seconds of audio for testing. Returns path to trimmed file."""
    trimmed = audio_path.rsplit(".", 1)[0] + f"_trim{seconds}s.wav"
    print(f"Trimming first {seconds}s for preview...", flush=True)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            audio_path,
            "-t",
            str(seconds),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            trimmed,
            "-y",
        ],
        check=True,
        capture_output=True,
    )
    return trimmed


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Persian lecture to Markdown"
    )
    parser.add_argument("input", help="Path to audio or video file")
    parser.add_argument(
        "--model",
        default="medium",
        help="Model size: tiny/base/small/medium/large-v3 (default: medium)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output .md path")
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Only transcribe first N seconds (e.g. --preview 180 for 3min test)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show transcribed text live (default: progress %% only)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    ext = os.path.splitext(input_path)[1].lower()
    audio_path = input_path

    # Trim for preview mode
    if args.preview > 0:
        audio_path = trim_audio(audio_path, args.preview)

    # Output path
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        suffix = f"_preview{args.preview}s" if args.preview > 0 else ""
        today = time.strftime("%Y-%m-%d")
        out_dir = os.path.expanduser(f"~/Desktop/hermes/{today}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}{suffix}_transcript.md")

    print(f"Loading model '{args.model}'...", flush=True)
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=8)

    print(f"Transcribing...", flush=True)
    start = time.time()
    segments, info = model.transcribe(
        audio_path, language="fa", beam_size=3, vad_filter=True
    )

    dur = info.duration
    print(f"Language: {info.language} | Duration: {int(dur//60)}m {int(dur%60)}s")
    print(f"{'─'*50}", flush=True)

    md = []
    md.append(f"# {os.path.splitext(os.path.basename(input_path))[0]} — Transcript")
    md.append("")
    label = f"Language: Persian | Model: {args.model} | Duration: {int(dur//60)}m {int(dur%60)}s"
    if args.preview > 0:
        label += f" | First {args.preview}s preview"
    md.append(label)
    md.append("")

    seg_count = 0
    for seg in segments:
        ts = f"{int(seg.start//60):02d}:{int(seg.start%60):02d}"
        text = seg.text.strip()
        line = f"**[{ts}]** {text}"
        md.append(line)
        seg_count += 1

        pct = seg.end / dur * 100
        if args.verbose:
            preview = text[:80] + ("..." if len(text) > 80 else "")
            print(f"[{ts}] ({pct:4.0f}%) {preview}", flush=True)
        else:
            # Progress only — rewrite same line
            bar_len = 20
            filled = int(pct / 100 * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\rProgress: |{bar}| {pct:4.0f}%  [{ts}]  ", end="", flush=True)

    # Final save
    text = "\n".join(md)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    elapsed = time.time() - start
    print(f"\n{'─'*50}")
    print(f"Done! {seg_count} segments in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Saved: {out_path}", flush=True)

    # Clean temp files
    if audio_path != input_path:
        os.remove(audio_path)
        print(f"Cleaned up: {audio_path}")


if __name__ == "__main__":
    main()

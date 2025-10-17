#!/usr/bin/env python3
"""
map_srt_to_diarized_speakers.py

Run Demucs (optional) -> pyannote diarization -> map each SRT subtitle to a diarized speaker
(by largest overlap, or nearest speaker if no overlap) -> extract exact subtitle audio and save
into out_dir/speaker_{N}/ with filenames containing index and timestamps.

Usage:
    export HF_TOKEN=hf_xxx
    python map_srt_to_diarized_speakers.py input_audio.mp3 --srt kan2.srt --out-dir my_voices --tmp /tmp/demucs_out --device cuda

Outputs:
  my_voices/speaker_1/001_0000000_0007000.wav
  my_voices/speaker_2/002_0007000_0010000.wav
  ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from pydub import AudioSegment
from huggingface_hub import login
from pyannote.audio import Pipeline

# ---------- helpers ----------
def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    print("Running:", " ".join(cmd))
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.stdout:
            print("--- STDOUT ---")
            print(p.stdout)
        if p.stderr:
            print("--- STDERR ---")
            print(p.stderr)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError as e:
        print("FileNotFoundError:", e)
        return -1, "", str(e)

def demucs_separate(input_path: Path, out_dir: Path, device: str = "cuda") -> Tuple[bool, Path]:
    base_args = [
        "demucs",
        "--two-stems=vocals",
        "--name", "htdemucs",
        "--device", device,
        "--out", str(out_dir),
        str(input_path),
    ]
    rc, out, err = run_cmd(base_args)
    if rc != 0:
        # try python -m demucs fallback
        py_cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "--name", "htdemucs", "--device", device, "--out", str(out_dir), str(input_path)]
        rc2, out2, err2 = run_cmd(py_cmd)
        if rc2 != 0:
            return False, Path("")
    # find vocals file
    matches = list(Path(out_dir).rglob("vocals.wav")) + list(Path(out_dir).rglob("*_vocals.wav"))
    if not matches:
        return False, Path("")
    chosen = matches[0]
    for m in matches:
        if input_path.stem in m.name:
            chosen = m
            break
    return True, chosen

# ---------- SRT parser ----------
def parse_srt_simple(srt_path: str) -> List[Dict]:
    text = Path(srt_path).read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"\n\s*\n", text.strip())
    blocks = []
    for p in parts:
        lines = [l.rstrip() for l in p.splitlines() if l.strip()]
        if not lines:
            continue
        time_line = next((l for l in lines if "-->" in l), None)
        if not time_line:
            continue
        # accept optional meta after '|' but we ignore
        if "|" in time_line:
            time_part, *_ = [s.strip() for s in time_line.split("|")]
        else:
            time_part = time_line.strip()
        m = re.search(r"(\d{2}:\d{2}:\d{2}[,\.]\d+)\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d+)", time_part)
        if not m:
            continue
        def to_sec(t):
            hh, mm, ss_ms = t.split(":")
            ss, ms = re.split(r"[,.]", ss_ms)
            return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
        start = to_sec(m.group(1))
        end = to_sec(m.group(2))
        idx = None
        if lines and re.fullmatch(r"\d+", lines[0]):
            idx = int(lines[0])
        idx = idx or -1
        # subtitle body after time line
        idx_timepos = lines.index(time_line)
        text_body = " ".join(lines[idx_timepos + 1 :]).strip()
        blocks.append({"index": idx, "start": start, "end": end, "text": text_body})
    return blocks

# ---------- diarize ----------
def diarize(vocals_path: Path, hf_token: str):
    login(token=hf_token)
    print("Loading pyannote/speaker-diarization pipeline (this may take a while)...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    print("Running diarization on:", vocals_path)
    diarization = pipeline({"uri": "audio", "audio": str(vocals_path)})

    speaker_segments = {}  # label -> list of (start_s, end_s)
    first_appearance = {}
    try:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            s = float(turn.start); e = float(turn.end)
            speaker_segments.setdefault(speaker, []).append((s,e))
            if speaker not in first_appearance or s < first_appearance[speaker]:
                first_appearance[speaker] = s
    except Exception:
        # fallback iteration
        for seg, label in diarization.items():
            s = float(seg.start); e = float(seg.end)
            speaker_segments.setdefault(label, []).append((s,e))
            if label not in first_appearance or s < first_appearance[label]:
                first_appearance[label] = s

    if not speaker_segments:
        raise RuntimeError("No speakers found by diarization")

    # order speakers by first appearance -> label_to_dir
    ordered = sorted(first_appearance.items(), key=lambda kv: kv[1])
    label_to_dir = {}
    for i, (label, _) in enumerate(ordered, start=1):
        label_to_dir[label] = f"speaker_{i}"
    print("Detected speakers:", label_to_dir)
    return speaker_segments, label_to_dir

# ---------- overlap utilities ----------
def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    return max(0.0, e - s)

def nearest_distance(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    # if overlap -> distance 0
    ov = overlap_seconds(a_start, a_end, b_start, b_end)
    if ov > 0:
        return 0.0
    # else distance between intervals
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return float('inf')

# ---------- main mapping and export ----------
def map_subs_to_speakers_and_export(subs, speaker_segments, label_to_dir, vocals_path: Path, out_dir: Path):
    audio = AudioSegment.from_file(str(vocals_path))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # create speaker dirs
    for d in sorted(set(label_to_dir.values())):
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    assigned = {}  # idx -> (speaker_dir, out_path)
    for block in subs:
        idx = int(block["index"])
        s_start = float(block["start"]); s_end = float(block["end"])
        # pick speaker with largest overlap
        best_label = None; best_overlap = 0.0
        for label, segments in speaker_segments.items():
            tot_ov = 0.0
            for (a,b) in segments:
                tot_ov += overlap_seconds(s_start, s_end, a, b)
            if tot_ov > best_overlap:
                best_overlap = tot_ov
                best_label = label
        if best_label is None or best_overlap <= 0.0:
            # pick nearest speaker by minimal distance
            best_label = None; best_dist = float('inf')
            for label, segments in speaker_segments.items():
                # compute min distance over all segments
                for (a,b) in segments:
                    d = nearest_distance(s_start, s_end, a, b)
                    if d < best_dist:
                        best_dist = d; best_label = label
        speaker_dir = out_dir / label_to_dir[best_label]
        # extract audio for the exact subtitle time window
        start_ms = max(0, int(round(s_start * 1000)))
        end_ms = max(0, int(round(s_end * 1000)))
        audio_len = len(audio)
        start_ms_cl = min(max(0, start_ms), audio_len)
        end_ms_cl = min(max(0, end_ms), audio_len)
        if start_ms_cl >= end_ms_cl:
            # produce a tiny silence as fallback (shouldn't happen with valid SRT)
            seg_audio = AudioSegment.silent(duration=50)
        else:
            seg_audio = audio[start_ms_cl:end_ms_cl]
        filename = f"{idx:03d}_{start_ms_cl:07d}_{end_ms_cl:07d}.wav"
        out_path = speaker_dir / filename
        seg_audio.export(out_path, format="wav")
        assigned[idx] = (label_to_dir[best_label], str(out_path))
        print(f"Subtitle {idx:03d} -> {label_to_dir[best_label]} (overlap_s={best_overlap:.3f}) saved: {out_path}")

    # summary counts
    counts: Dict[str,int] = {}
    for idx, (spk, p) in assigned.items():
        counts[spk] = counts.get(spk, 0) + 1
    print("\nSummary (files per speaker):")
    total = 0
    for spk in sorted(counts.keys()):
        print(f"  {spk}: {counts[spk]}")
        total += counts[spk]
    print("Total subtitle files exported:", total, " (should equal number of SRT entries)")

    return assigned

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Map SRT subtitles to diarized speakers and export per-subtitle WAVs in speaker_N/ folders.")
    ap.add_argument("input", help="Input audio/video path (wav/mp3/mp4 etc).")
    ap.add_argument("--srt", required=True, help="Reference SRT (used only for subtitle times/text).")
    ap.add_argument("--out-dir", default="my_voices", help="Output directory containing speaker_N/ folders.")
    ap.add_argument("--tmp", default="/tmp/demucs_out", help="Temporary demucs output dir.")
    ap.add_argument("--device", default="cuda", help="Device for demucs (cuda or cpu).")
    ap.add_argument("--no-demucs", action="store_true", help="Skip Demucs separation and run diarization on input audio directly.")
    ap.add_argument("--hf-token-env", default="HF_TOKEN", help="Environment variable name that holds HF token.")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    if not inp.exists():
        print("Input not found:", inp); sys.exit(1)

    hf_token = os.environ.get(args.hf_token_env) or os.environ.get("hf_token") or os.environ.get("HF_TOKEN".lower())
    if not hf_token:
        print("Set HF token env, e.g.: export HF_TOKEN=hf_xxx"); sys.exit(1)

    # Prepare vocals_path
    if args.no_demucs:
        vocals_path = inp
        print("Skipping Demucs; using input for diarization:", vocals_path)
    else:
        tmp_out = Path(args.tmp)
        if tmp_out.exists():
            try: shutil.rmtree(tmp_out)
            except Exception: pass
        tmp_out.mkdir(parents=True, exist_ok=True)
        print("Running Demucs to extract vocals (device=%s)..." % args.device)
        ok, chosen = demucs_separate(inp, tmp_out, device=args.device)
        if not ok:
            print("Demucs failed; aborting."); sys.exit(1)
        vocals_path = chosen
        # copy to working filename
        try:
            working_vocals = Path.cwd() / f"{inp.stem}_vocals.wav"
            shutil.copy2(vocals_path, working_vocals)
            vocals_path = working_vocals
        except Exception:
            pass
        print("Vocals for diarization:", vocals_path)

    # Run diarization
    try:
        speaker_segments, label_to_dir = diarize(Path(vocals_path), hf_token)
    except Exception as e:
        print("Diarization failed:", e); sys.exit(1)

    # Parse SRT (reference only)
    subs = parse_srt_simple(args.srt)
    if not subs:
        print("No subtitles parsed from SRT"); sys.exit(1)
    print("Parsed", len(subs), "subtitles from SRT")

    # Map each subtitle to speaker and export
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    assigned = map_subs_to_speakers_and_export(subs, speaker_segments, label_to_dir, Path(vocals_path), out_dir)

    print("\nDone. Files are in", out_dir)
    sys.exit(0)

if __name__ == "__main__":
    main()

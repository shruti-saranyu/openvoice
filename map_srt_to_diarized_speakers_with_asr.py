#!/usr/bin/env python3
"""
map_srt_to_diarized_speakers_full.py

Full pipeline:
 - (optional) Demucs -> vocals extraction
 - pyannote speaker diarization (if available) or a local MFCC+clustering fallback
 - parse SRT (reference only)
 - for each subtitle, pick diarized speaker by largest overlap (or nearest if none)
 - export exact subtitle audio into out_dir/speaker_N/{idx}_{startms}_{endms}.wav
 - run ASR on each exported file and add 'ref_text' to mapping.json

This version includes:
 - robust ASR helper (faster-whisper preferred, openai-whisper fallback)
 - extensive ASR debug logging and context-extension for short segments
 - safe diarization that falls back to a local clustering method when pyannote/tfcodec/HF gated model fails
 - keep temp ASR files if ASR_KEEP_TEMPFILES=1 for debugging

Usage example:
  export HF_TOKEN=hf_xxx
  pip install demucs pydub pyannote.audio huggingface_hub faster-whisper openai-whisper librosa soundfile scikit-learn
  python map_srt_to_diarized_speakers_full.py input.mp4 --srt kan2.srt --out-dir my_voices --mapping-out mapping.json --device cuda --asr-models large-v2,medium --asr-device cuda --asr-lang kn --asr-context-ms 4000 --asr-beam 7
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Core audio libraries
try:
    from pydub import AudioSegment
except Exception as e:
    print("Missing dependency: pydub. Install with `pip install pydub` and ensure ffmpeg is installed.", file=sys.stderr)
    raise

# Diarization (optional)
try:
    from huggingface_hub import login as hf_login  # used if available
    from pyannote.audio import Pipeline  # optional; we will try and fall back if it fails
    HAS_PYANNOTE = True
except Exception:
    HAS_PYANNOTE = False

# ASR dependencies (optional)
HAS_FASTER = False
HAS_WHISPER = False
try:
    from faster_whisper import WhisperModel as FWWhisperModel  # type: ignore
    HAS_FASTER = True
except Exception:
    HAS_FASTER = False
try:
    import whisper  # type: ignore
    HAS_WHISPER = True
except Exception:
    HAS_WHISPER = False

# audio conversion libs
try:
    import soundfile as sf
    import librosa
    import numpy as np
except Exception:
    print("Missing dependency: soundfile or librosa or numpy. Install with `pip install soundfile librosa numpy`", file=sys.stderr)
    raise

# clustering fallback
try:
    from sklearn.cluster import AgglomerativeClustering
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# ------------------------------ helper shell runner ------------------------------
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
        return -1, "", str(e)

# ------------------------------ demucs separation ------------------------------
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
        py_cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "--name", "htdemucs", "--device", device, "--out", str(out_dir), str(input_path)]
        rc2, out2, err2 = run_cmd(py_cmd)
        if rc2 != 0:
            return False, Path("")
    matches = list(Path(out_dir).rglob("vocals.wav")) + list(Path(out_dir).rglob("*_vocals.wav"))
    if not matches:
        return False, Path("")
    chosen = matches[0]
    for m in matches:
        if input_path.stem in m.name:
            chosen = m
            break
    return True, chosen

# ------------------------------ SRT parser (simple) ------------------------------
def parse_srt_simple(srt_path: str) -> List[Dict]:
    txt = Path(srt_path).read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"\n\s*\n", txt.strip())
    blocks = []
    for p in parts:
        lines = [l.rstrip() for l in p.splitlines() if l.strip()]
        if not lines:
            continue
        time_line = next((l for l in lines if "-->" in l), None)
        if not time_line:
            continue
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
        idx_timepos = lines.index(time_line)
        text_body = " ".join(lines[idx_timepos + 1 :]).strip()
        blocks.append({"index": idx, "start": start, "end": end, "text": text_body})
    return blocks

# ------------------------------ diarize using pyannote (robust) ------------------------------
def diarize_pyannote(vocals_path: Path, hf_token: str):
    import torch
    from huggingface_hub import login as hf_login
    from pyannote.audio import Pipeline

    if hf_token:
        try:
            hf_login(token=hf_token)
        except Exception as e:
            print("Warning: huggingface_hub.login() raised:", e)

    repo_id = "pyannote/speaker-diarization"
    pipeline = None
    errors = []

    try:
        pipeline = Pipeline.from_pretrained(repo_id)
        print("Loaded pipeline without explicit token.")
    except Exception as e:
        errors.append(("no_token", e))
        print("Load without token failed:", type(e).__name__, e)

    if pipeline is None and hf_token:
        try:
            pipeline = Pipeline.from_pretrained(repo_id, use_auth_token=hf_token)
            print("Loaded pipeline with use_auth_token=HF_TOKEN.")
        except Exception as e:
            errors.append(("use_auth_token", e))
            print("use_auth_token failed:", type(e).__name__, e)

    if pipeline is None and hf_token:
        try:
            pipeline = Pipeline.from_pretrained(repo_id, token=hf_token)
            print("Loaded pipeline with token=HF_TOKEN.")
        except Exception as e:
            errors.append(("token", e))
            print("token failed:", type(e).__name__, e)

    if pipeline is None:
        print("Failed to load pyannote pipeline. Diagnostics:")
        for tag, ex in errors:
            print(f"  Attempt [{tag}]: {type(ex).__name__}: {ex}")
        raise RuntimeError("Could not load pyannote/speaker-diarization; see diagnostics above.")

    print("Pyannote pipeline loaded successfully.")
    y, sr = librosa.load(str(vocals_path), sr=None, mono=True)
    wav = torch.from_numpy(y.astype("float32"))
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() == 2 and wav.shape[0] != 1:
        wav = wav.mean(dim=0, keepdim=True)

    print("Running diarization (in-memory)...")
    diarization = pipeline({"waveform": wav, "sample_rate": sr})

    speaker_segments = {}
    first_appearance = {}
    try:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            s = float(turn.start); e = float(turn.end)
            speaker_segments.setdefault(speaker, []).append((s, e))
            if speaker not in first_appearance or s < first_appearance[speaker]:
                first_appearance[speaker] = s
    except Exception:
        for seg, label in diarization.items():
            s = float(seg.start); e = float(seg.end)
            speaker_segments.setdefault(label, []).append((s, e))
            if label not in first_appearance or s < first_appearance[label]:
                first_appearance[label] = s

    if not speaker_segments:
        raise RuntimeError("No speakers found by diarization")

    ordered = sorted(first_appearance.items(), key=lambda kv: kv[1])
    label_to_dir = {}
    for i, (label, _) in enumerate(ordered, start=1):
        label_to_dir[label] = f"speaker_{i}"

    print("Detected speakers (pyannote):", label_to_dir)
    return speaker_segments, label_to_dir

# ------------------------------ fallback diarization (MFCC + clustering) ------------------------------
def diarize_fallback_clustering(vocals_path: Path,
                                max_speakers: int | None = None,
                                window_sec: float = 1.0,
                                hop_sec: float = 0.5,
                                n_mfcc: int = 13,
                                sample_rate: int | None = None):
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required for fallback diarizer. Install with `pip install scikit-learn`.")
    print("Running fallback clustering diarization (mfcc -> agglomerative).")
    y, sr = librosa.load(str(vocals_path), sr=sample_rate, mono=True)
    duration_s = len(y) / sr
    if duration_s <= 0:
        raise RuntimeError("Empty audio in fallback diarizer")

    win = int(round(window_sec * sr))
    hop = int(round(hop_sec * sr))
    if hop <= 0:
        hop = min(win, 16000)

    frames = []
    frame_times = []
    for start in range(0, max(1, len(y) - win + 1), hop):
        end = start + win
        frame = y[start:end]
        if len(frame) < win:
            frame = np.pad(frame, (0, win - len(frame)))
        mf = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=n_mfcc)
        feat = np.concatenate([np.mean(mf, axis=1), np.std(mf, axis=1)])
        frames.append(feat)
        frame_times.append((start / sr, min(end / sr, duration_s)))

    if not frames:
        mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        feat = np.concatenate([np.mean(mf, axis=1), np.std(mf, axis=1)])
        frames = [feat]
        frame_times = [(0.0, duration_s)]

    X = np.vstack(frames)

    if max_speakers is None:
        est = 2 if duration_s < 30 else (3 if duration_s < 120 else 4)
        n_clusters = est
    else:
        n_clusters = int(max_speakers)
    n_clusters = max(1, n_clusters)

    try:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="euclidean", linkage="ward")
        labels = clustering.fit_predict(X)
    except Exception as e:
        print("Clustering failed, marking all frames as single speaker. Error:", e)
        labels = np.zeros(len(X), dtype=int)

    speaker_segments = {}
    cur_label = labels[0]
    cur_start = frame_times[0][0]
    cur_end = frame_times[0][1]
    for i in range(1, len(labels)):
        if labels[i] == cur_label:
            cur_end = frame_times[i][1]
        else:
            speaker_segments.setdefault(f"spk_{int(cur_label)}", []).append((cur_start, cur_end))
            cur_label = labels[i]
            cur_start = frame_times[i][0]
            cur_end = frame_times[i][1]
    speaker_segments.setdefault(f"spk_{int(cur_label)}", []).append((cur_start, cur_end))

    gap_tol = 0.35
    for lbl, segs in list(speaker_segments.items()):
        merged = []
        for s, e in segs:
            if not merged:
                merged.append((s, e))
            else:
                ps, pe = merged[-1]
                if s - pe <= gap_tol:
                    merged[-1] = (ps, max(pe, e))
                else:
                    merged.append((s, e))
        speaker_segments[lbl] = merged

    first_appearance = {lbl: min(s for s, _ in segs) for lbl, segs in speaker_segments.items()}
    ordered = sorted(first_appearance.items(), key=lambda kv: kv[1])
    label_to_dir = {}
    for i, (lbl, _) in enumerate(ordered, start=1):
        label_to_dir[lbl] = f"speaker_{i}"
    print("Fallback diarization produced", len(label_to_dir), "speakers.")
    return speaker_segments, label_to_dir

# --------------------------- safe wrapper: try pyannote then fallback ---------------------------
def safe_diarize(vocals_path: Path, hf_token: str, fallback_max_speakers: int | None = None):
    if HAS_PYANNOTE:
        try:
            return diarize_pyannote(vocals_path, hf_token)
        except Exception as e:
            print("pyannote diarize failed with:", type(e).__name__, e)
            print("Falling back to local clustering diarizer.")
    else:
        print("pyannote not available; using fallback diarizer.")
    return diarize_fallback_clustering(vocals_path, max_speakers=fallback_max_speakers, window_sec=1.0, hop_sec=0.5)

# ------------------------------ utilities ------------------------------
def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    return max(0.0, e - s)

def nearest_distance(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    ov = overlap_seconds(a_start, a_end, b_start, b_end)
    if ov > 0:
        return 0.0
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return float("inf")

# ------------------------------ ASR helpers ------------------------------
def ensure_wav_16k_mono(src_path: str) -> str:
    y, _sr = librosa.load(src_path, sr=16000, mono=True)
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tf.name, y, 16000, subtype="PCM_16")
    return tf.name

def load_asr_model_try(models_preferred: List[str], device: str | None = None):
    device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    last_exc = None

    if HAS_FASTER:
        for mname in models_preferred:
            try:
                compute_type = "float16" if str(device).startswith("cuda") else "int8_float16"
                print(f"Attempting to load faster-whisper model '{mname}' on {device} (compute_type={compute_type})")
                fw = FWWhisperModel(mname, device=device, compute_type=compute_type)
                print(f"Loaded faster-whisper '{mname}'")
                return {"kind": "faster", "model": fw, "name": mname, "device": device}
            except Exception as e:
                last_exc = e
                print(f"faster-whisper load failed for {mname}:", type(e).__name__, e)

    if HAS_WHISPER:
        for mname in models_preferred:
            try:
                print(f"Attempting to load openai-whisper model '{mname}' on {device}")
                w = whisper.load_model(mname, device=device)
                print(f"Loaded openai-whisper '{mname}'")
                return {"kind": "whisper", "model": w, "name": mname, "device": device}
            except Exception as e:
                last_exc = e
                print(f"openai-whisper load failed for {mname}:", type(e).__name__, e)

    raise RuntimeError("No ASR backend/model could be loaded. Last error: " + repr(last_exc))

def trascribe_faster_whisper(wav_path: str,
                             asr_loaded: dict | None,
                             language: str | None = None,
                             prefer_lang: bool = True,
                             context_ms: int = 4000,
                             vocals_full_path: str | None = None,
                             start_ms: int | None = None,
                             end_ms: int | None = None,
                             beam_size: int = 7,
                             max_len_chars: int | None = None) -> str:
    """
    Robust transcription helper with debug logging and aggressive context extension for short segments.
    """
    if asr_loaded is None:
        print("[ASR] asr_loaded is None — no model available")
        return ""

    tmp_input = wav_path
    temp_files: List[str] = []

    # duration
    try:
        info = sf.info(wav_path)
        dur_ms = int(1000.0 * (info.frames / info.samplerate))
    except Exception:
        try:
            ytmp, sr = librosa.load(wav_path, sr=None, mono=True)
            dur_ms = int(1000.0 * (len(ytmp) / (sr or 16000)))
        except Exception:
            dur_ms = 0

    # extend context for very short clips
    if dur_ms < 1200 and context_ms > 0 and vocals_full_path and start_ms is not None and end_ms is not None:
        half = context_ms // 2
        ext_s = max(0, start_ms - half)
        ext_e = end_ms + half
        try:
            full_audio = AudioSegment.from_file(vocals_full_path)
            ext_s_cl = max(0, min(len(full_audio), ext_s))
            ext_e_cl = max(0, min(len(full_audio), ext_e))
            seg = full_audio[ext_s_cl:ext_e_cl]
            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            seg.export(tf.name, format="wav")
            tmp_input = tf.name
            temp_files.append(tf.name)
            print(f"[ASR] created extended context for {wav_path} -> {tf.name} ({ext_s_cl}-{ext_e_cl} ms)")
        except Exception as e:
            print("[ASR] failed to create extended context audio:", e)
            tmp_input = wav_path

    # ensure 16k mono
    try:
        wav16 = ensure_wav_16k_mono(tmp_input)
        temp_files.append(wav16)
    except Exception as e:
        print("[ASR] ensure_wav_16k_mono failed:", e)
        wav16 = tmp_input

    kind = asr_loaded.get("kind")
    model = asr_loaded.get("model")
    out_text = ""

    try:
        if kind == "faster":
            lang_arg = language if (prefer_lang and language) else None
            print(f"[ASR] faster-whisper transcribing {wav16} (lang={lang_arg}, beam={beam_size})")
            segments, info = model.transcribe(wav16, language=lang_arg, task="transcribe", beam_size=beam_size)
            parts: List[str] = []
            for seg in segments:
                t = getattr(seg, "text", None) or (seg.get("text") if isinstance(seg, dict) else None)
                start = getattr(seg, "start", None)
                end = getattr(seg, "end", None)
                print(f"[ASR SEG] {start}->{end} : {repr(t)}")
                if t:
                    parts.append(t.strip())
            out_text = " ".join(parts).strip()
        else:
            opts = {}
            if prefer_lang and language:
                opts["language"] = language
            print(f"[ASR] whisper transcribing {wav16} opts={opts}")
            res = model.transcribe(wav16, **opts)
            if isinstance(res, dict):
                out_text = res.get("text", "").strip()
            else:
                out_text = getattr(res, "text", str(res)).strip()
    except Exception as e:
        print("[ASR] transcription failed:", type(e).__name__, e)
        out_text = ""

    if max_len_chars and len(out_text) > max_len_chars:
        out_text = out_text[:max_len_chars]

    # cleanup temp files unless requested to keep
    if not os.environ.get("ASR_KEEP_TEMPFILES"):
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
    else:
        print("[ASR] ASR_KEEP_TEMPFILES set — keeping temp files:", temp_files)

    out_text = re.sub(r"\s+", " ", out_text).strip()
    return out_text

# ------------------------------ main mapping/export with ASR ------------------------------
def map_subs_to_speakers_and_export(subs, speaker_segments, label_to_dir, vocals_path: Path, out_dir: Path,
                                    do_asr: bool, asr_loaded: dict | None, asr_lang: str | None, asr_context_ms: int, asr_beam: int):
    audio = AudioSegment.from_file(str(vocals_path))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for d in sorted(set(label_to_dir.values())):
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    mapping_out: Dict[str, dict] = {}
    assigned = {}

    for block in subs:
        idx = int(block["index"])
        s_start = float(block["start"]); s_end = float(block["end"])
        best_label = None; best_overlap = 0.0
        for label, segments in speaker_segments.items():
            tot_ov = 0.0
            for (a,b) in segments:
                tot_ov += overlap_seconds(s_start, s_end, a, b)
            if tot_ov > best_overlap:
                best_overlap = tot_ov
                best_label = label
        if best_label is None or best_overlap <= 0.0:
            best_label = None; best_dist = float("inf")
            for label, segments in speaker_segments.items():
                for (a,b) in segments:
                    d = nearest_distance(s_start, s_end, a, b)
                    if d < best_dist:
                        best_dist = d; best_label = label

        speaker_dir_name = label_to_dir[best_label]
        speaker_dir = out_dir / speaker_dir_name
        start_ms = max(0, int(round(s_start * 1000)))
        end_ms = max(0, int(round(s_end * 1000)))
        audio_len = len(audio)
        start_ms_cl = min(max(0, start_ms), audio_len)
        end_ms_cl = min(max(0, end_ms), audio_len)
        if start_ms_cl >= end_ms_cl:
            seg_audio = AudioSegment.silent(duration=50)
        else:
            seg_audio = audio[start_ms_cl:end_ms_cl]
        filename = f"{idx:03d}_{start_ms_cl:07d}_{end_ms_cl:07d}.wav"
        out_path = speaker_dir / filename
        seg_audio.export(out_path, format="wav")
        assigned[idx] = (speaker_dir_name, str(out_path))

        # ASR
        ref_text = ""
        if do_asr and asr_loaded is not None:
            try:
                ref_text = trascribe_faster_whisper(
                    str(out_path),
                    asr_loaded,
                    language=asr_lang,
                    prefer_lang=(asr_lang is not None),
                    context_ms=asr_context_ms,
                    vocals_full_path=str(vocals_path) if vocals_path is not None else None,
                    start_ms=start_ms_cl,
                    end_ms=end_ms_cl,
                    beam_size=asr_beam,
                )
                print(f"[ASR] subtitle {idx}: got {len(ref_text)} chars")
            except Exception as e:
                print("[ASR] unexpected error for", out_path, ":", type(e).__name__, e)
                ref_text = ""
        else:
            if not do_asr:
                print("[ASR] skipping ASR (no-asr flag).")
            else:
                print("[ASR] asr_loaded is None; cannot run ASR.")

        mapping_out[str(idx)] = {
            "index": idx,
            "speaker_dir": speaker_dir_name,
            "file": str(out_path),
            "start_s": s_start,
            "end_s": s_end,
            "text": block.get("text", ""),
            "ref_text": ref_text
        }
        print(f"Exported subtitle {idx:03d} -> {speaker_dir_name} ({out_path.name}), overlap_s={best_overlap:.3f}, asr_len={len(ref_text)}")

    counts = {}
    for idx, (spk, p) in assigned.items():
        counts[spk] = counts.get(spk, 0) + 1
    print("\nSummary (files per speaker):")
    total = 0
    for spk in sorted(counts.keys()):
        print(f"  {spk}: {counts[spk]}")
        total += counts[spk]
    print("Total subtitle files exported:", total)
    return mapping_out

# ------------------------------ CLI / main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Map SRT subtitles to diarized speakers, export per-subtitle WAVs into speaker_N/ and produce mapping JSON with ASR ref_text.")
    ap.add_argument("input", help="Input audio/video path (wav/mp3/mp4 etc).")
    ap.add_argument("--srt", required=True, help="Reference SRT (used only for subtitle times/text).")
    ap.add_argument("--out-dir", default="my_voices", help="Output directory containing speaker_N/ folders.")
    ap.add_argument("--mapping-out", default="mapping.json", help="Output mapping JSON path.")
    ap.add_argument("--tmp", default="/tmp/demucs_out", help="Temporary demucs output dir.")
    ap.add_argument("--device", default="cuda", help="Device for demucs (cuda or cpu).")
    ap.add_argument("--no-demucs", action="store_true", help="Skip Demucs separation and run diarization on input audio directly.")
    ap.add_argument("--hf-token-env", default="HF_TOKEN", help="Env var name containing HF token.")
    ap.add_argument("--no-asr", action="store_true", help="Skip ASR/ref_text generation.")
    ap.add_argument("--asr-models", default="large-v2,medium,small", help="Comma-separated ASR model names to try in order (e.g. large-v2,medium,small).")
    ap.add_argument("--asr-device", default=None, help="Device for ASR: 'cuda' or 'cpu'. Defaults to cuda if available.")
    ap.add_argument("--asr-lang", default=None, help="Force ASR language code (e.g. 'kn' for Kannada). If omitted, ASR will auto-detect.")
    ap.add_argument("--asr-context-ms", type=int, default=4000, help="For very short segments (<1200ms), extend ASR window by this many ms (distributed on both sides).")
    ap.add_argument("--asr-beam", type=int, default=7, help="Beam size for faster-whisper decoding (higher may improve accuracy).")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    if not inp.exists():
        print("Input not found:", inp); sys.exit(1)

    hf_token = os.environ.get(args.hf_token_env) or os.environ.get("hf_token") or os.environ.get("HF_TOKEN".lower())
    if not hf_token:
        print("Set HF token env, e.g.: export HF_TOKEN=hf_xxx"); sys.exit(1)

    # Prepare vocals file (Demucs)
    if args.no_demucs:
        vocals_path = inp
        print("Skipping Demucs; using input for diarization:", vocals_path)
    else:
        tmp_out = Path(args.tmp)
        if tmp_out.exists():
            try:
                shutil.rmtree(tmp_out)
            except Exception:
                pass
        tmp_out.mkdir(parents=True, exist_ok=True)
        print("Running Demucs to extract vocals (device=%s)..." % args.device)
        ok, chosen = demucs_separate(inp, tmp_out, device=args.device)
        if not ok:
            print("Demucs failed; aborting."); sys.exit(1)
        vocals_path = chosen
        try:
            working_vocals = Path.cwd() / f"{inp.stem}_vocals.wav"
            shutil.copy2(vocals_path, working_vocals)
            vocals_path = working_vocals
        except Exception:
            pass
        print("Vocals for diarization:", vocals_path)

    # Run diarization (pyannote preferred, fallback to clustering)
    try:
        speaker_segments, label_to_dir = safe_diarize(Path(vocals_path), hf_token, fallback_max_speakers=4)
    except Exception as e:
        print("Diarization failed:", e); sys.exit(1)

    # Parse SRT
    subs = parse_srt_simple(args.srt)
    if not subs:
        print("No subtitles parsed from SRT"); sys.exit(1)
    print("Parsed", len(subs), "subtitles from SRT")

    # ASR setup
    do_asr = not args.no_asr
    asr_loaded = None
    if do_asr:
        try:
            asr_device = args.asr_device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        except Exception:
            asr_device = args.asr_device or "cpu"
        try:
            models_list = [m.strip() for m in args.asr_models.split(",") if m.strip()]
            asr_loaded = load_asr_model_try(models_list, asr_device)
        except Exception as e:
            print("Failed to load ASR model:", e)
            print("You can re-run with --no-asr to skip ASR.")
            sys.exit(1)

    # Map & export (and ASR)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mapping_out = map_subs_to_speakers_and_export(subs, speaker_segments, label_to_dir, Path(vocals_path), out_dir, do_asr, asr_loaded, args.asr_lang, args.asr_context_ms, args.asr_beam)

    # Write mapping JSON
    mapping_path = Path(args.mapping_out)
    mapping_path.write_text(json.dumps(mapping_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote mapping JSON to:", mapping_path)

    print("Done. Per-subtitle WAVs in:", out_dir)
    sys.exit(0)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clone_from_srt_indicf5.py 

Pipeline that uses only mapping.json to synthesise each segment with IndicF5.
For each mapping entry:
  - uses mapping[file] as reference WAV (fallbacks attempted)
  - uses mapping[text] (or mapping[ref_text]) as the text to synthesise
  - uses mapping[start_s]/mapping[end_s] as target duration
  - synthesises with IndicF5, pads/trims to timestamp duration, stitches outputs

Usage:
    python clone_from_mapping_indicf5_only.py --mapping mapping.json \
        --voices-dir my_voices --extracted-dir extracted_voices --out outputs/final.wav \
        --lang en
"""
from __future__ import annotations

import argparse
import json
import os
import glob
import shlex
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, Union
import threading
import shutil
import sys

import soundfile as sf
import numpy as np
from pydub import AudioSegment
import torch
from transformers import AutoModel

# -------------------- audio helpers --------------------
def audio_duration_s(in_wav: str) -> float:
    seg = AudioSegment.from_file(in_wav)
    return len(seg) / 1000.0

def pad_or_trim_keep_first(in_wav: str, target_sec: float, out_wav: str, end_pad_ms: int = 30, start_pad_ms: int = 30):
    seg = AudioSegment.from_file(in_wav)
    tgt_ms = int(target_sec * 1000)
    start_pad = start_pad_ms
    end_target = max(0, tgt_ms - start_pad)
    if len(seg) > end_target:
        new_seg = seg[:end_target]
    else:
        new_seg = seg + AudioSegment.silent(duration=(end_target - len(seg)))
    new_seg = AudioSegment.silent(duration=start_pad) + new_seg
    new_seg.export(out_wav, format="wav")
    return out_wav

# -------------------- mapping helpers --------------------
def load_mapping_json(mapping_path: Path):
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    # Normalize mapping entries to use integer indices and ensure start/end exist
    entries = []
    for k, v in raw.items():
        entry = dict(v)
        try:
            idx = int(entry.get("index", None) or k)
        except Exception:
            # skip entries without numeric index
            continue
        entry["index"] = idx
        # ensure start_s/end_s available - if missing, try start_s/end or start/end
        for key in ("start_s", "start", "start_sec"):
            if "start_s" not in entry and key in entry:
                entry["start_s"] = float(entry[key])
        for key in ("end_s", "end", "end_sec"):
            if "end_s" not in entry and key in entry:
                entry["end_s"] = float(entry[key])
        # if no times, skip (we rely on mapping timestamps)
        if "start_s" not in entry or "end_s" not in entry:
            # allow small fallback: duration seconds field
            if "duration_s" in entry:
                entry["end_s"] = float(entry.get("start_s", 0.0)) + float(entry["duration_s"])
            else:
                # skip entry if no timestamps
                continue
        entries.append(entry)
    # sort by start time then index
    entries = sorted(entries, key=lambda e: (float(e.get("start_s", 0.0)), int(e["index"])))
    return raw, {e["index"]: e for e in entries}, entries

def pick_longest_in_speaker_dir(voices_dir: Path, speaker_dir_name: str):
    spath = voices_dir / speaker_dir_name
    if not spath.exists() or not spath.is_dir():
        return None
    wavs = list(spath.glob("*.wav"))
    if not wavs:
        return None
    best = None
    best_dur = -1.0
    for w in wavs:
        try:
            d = audio_duration_s(str(w))
            if d > best_dur:
                best_dur = d
                best = w
        except Exception:
            try:
                sz = w.stat().st_size
                if sz > best_dur:
                    best_dur = sz
                    best = w
            except Exception:
                pass
    return str(best) if best else None

# -------------------- IndicF5 helpers --------------------
def save_model_audio(audio_obj, out_path, samplerate=24000):
    # audio_obj may be numpy array or torch tensor or list
    if hasattr(audio_obj, "detach"):  # torch tensor
        arr = audio_obj.detach().cpu().numpy()
    else:
        arr = np.array(audio_obj)
    # handle int16 output
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    arr = arr.astype(np.float32)
    sf.write(out_path, arr, samplerate)

# Reusable helper that accepts either a loaded model or repo string
def synthesize_indicf5(
    model_or_repo: Union[str, AutoModel],
    text: str,
    ref_audio_path: str,
    ref_text: str,
    out_wav: Optional[str] = None,
    device: Optional[str] = None,
    samplerate: int = 24000,
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    extra_kwargs: forwarded to model(...). Use {'language':'en','lang':'en'} to force text language.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None
    model_loaded_here = False
    if isinstance(model_or_repo, str):
        repo = model_or_repo
        model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        model.to(device)
        model_loaded_here = True
    else:
        model = model_or_repo
        try:
            model.to(device)
        except Exception:
            pass

    if extra_kwargs is None:
        extra_kwargs = {}

    # call model forwarding extra kwargs such as language/lang
    out_audio = model(text, ref_audio_path=str(ref_audio_path), ref_text=ref_text, **extra_kwargs)

    if hasattr(out_audio, "detach"):
        arr = out_audio.detach().cpu().numpy()
    else:
        arr = np.array(out_audio)
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    arr = arr.astype(np.float32)

    if out_wav:
        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_wav, arr, samplerate)
        if model_loaded_here:
            try:
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass
        return out_wav
    else:
        if model_loaded_here:
            try:
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass
        return arr

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Clone from mapping.json using IndicF5 only (no SRT).")
    ap.add_argument("--mapping", required=True, help="mapping.json produced by the mapping step.")
    ap.add_argument("--voices-dir", default="my_voices", help="Directory containing speaker_N/ folders.")
    ap.add_argument("--extracted_dir", default="extracted_voices", help="Fallback directory with extracted reference WAVs.")
    ap.add_argument("--out", default="outputs_v2/mapping_cloned_final.wav", help="Final stitched output WAV.")
    ap.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    ap.add_argument("--lead-silence-ms", type=int, default=120, help="Lead silence prepended to output segments (ms).")
    ap.add_argument("--end-pad-ms", type=int, default=30, help="Pad at segment end (ms).")
    ap.add_argument("--gpu-workers", type=int, default=1, help="Number of parallel GPU synthesis workers.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip generation when a segment already exists.")
    ap.add_argument("--no-clean", action="store_true", help="Do not remove temp dir after run (for debugging).")
    ap.add_argument("--lang", default=None, help="Language code for the TEXT to be synthesized (e.g. 'en', 'hi', 'kn'). This is forwarded to the model as language/lang.")
    ap.add_argument("--temp-dir", default="temp", help="Directory to store generated per-segment TTS files before stitching.")
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device chosen:", device)
    repo = Path(".").resolve()
    tmp_dir = repo / args.temp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        raise SystemExit(f"mapping.json not found at: {mapping_path}. Please provide mapping.json")

    raw_mapping, mapping_by_index, ordered_entries = load_mapping_json(mapping_path)
    if not ordered_entries:
        raise SystemExit("No valid entries with timestamps found in mapping.json. Each entry must include start_s and end_s.")
    print(f"Loaded mapping.json with {len(ordered_entries)} entries (using timestamps).")

    voices_dir = Path(args.voices_dir)

    def select_ref_from_entry(entry: Dict[str, Any]):
        # priority: mapping.file -> matched segment path inside mapping -> speaker_dir longest file -> extracted_dir fallback
        f = entry.get("file")
        if f and Path(f).exists():
            return str(Path(f))
        # check matched_segments list
        for m in entry.get("matched_segments", []) or []:
            if m.get("segment_path") and Path(m["segment_path"]).exists():
                return str(Path(m["segment_path"]))
        # speaker_dir fallback
        spk = entry.get("speaker_dir")
        if spk:
            cand = pick_longest_in_speaker_dir(voices_dir, spk)
            if cand:
                return cand
        # final fallback to extracted_dir
        refs = sorted(glob.glob(os.path.join(args.extracted_dir, "*.wav")))
        if refs:
            return refs[0]
        return None

    # Load IndicF5 model once and reuse
    HF_REPO_ID = "ai4bharat/IndicF5"
    print("Loading IndicF5 model from HF (this may download weights; set HUGGINGFACE_TOKEN if gated)...")
    model = AutoModel.from_pretrained(HF_REPO_ID, trust_remote_code=True).to(device)
    print("IndicF5 model loaded to device:", device)

    # build extra kwargs from --lang if provided (forward to model)
    extra_kwargs: Dict[str, Any] = {}
    if args.lang:
        extra_kwargs["language"] = args.lang
        extra_kwargs["lang"] = args.lang

    semaphore = threading.Semaphore(max(1, args.gpu_workers))

    def synth_for_entry(entry: Dict[str, Any]):
        idx = int(entry["index"])
        out_seg = tmp_dir / f"seg_{idx}_out_final.wav"
        if args.skip_existing and out_seg.exists():
            print(f"[skip] idx={idx} exists -> {out_seg}")
            return str(out_seg)

        ref_wav = select_ref_from_entry(entry)
        if not ref_wav:
            print(f"[warn] no reference WAV found for idx={idx}; producing silence fallback.")
            dur = max(0.05, float(entry.get("end_s", entry.get("start_s", 0.0))) - float(entry.get("start_s", 0.0)))
            silent = AudioSegment.silent(duration=int(max(50, args.lead_silence_ms) + dur * 1000))
            silent.export(out_seg, format="wav")
            return str(out_seg)

        # pick text to synthesise: prefer mapping.text -> mapping.ref_text -> filename stem
        text = entry.get("text") or entry.get("ref_text") or Path(ref_wav).stem.replace("_", " ")
        ref_text = entry.get("ref_text") or entry.get("text") or Path(ref_wav).stem.replace("_", " ")

        target_dur = max(0.05, float(entry.get("end_s", 0.0)) - float(entry.get("start_s", 0.0)))
        if target_dur <= 0:
            target_dur = audio_duration_s(ref_wav)

        # synthesize (limit concurrency with semaphore)
        semaphore.acquire()
        try:
            print(f"[synth] idx={idx} dur={target_dur:.2f}s text='{text[:60]}' ref={Path(ref_wav).name} lang={args.lang}")
            synthesize_indicf5(model, text, ref_wav, ref_text, out_wav=str(out_seg), device=device, samplerate=24000, extra_kwargs=extra_kwargs)
        except Exception as e:
            print(f"[error] synthesis failed idx={idx}: {e}")
            # fallback to silence
            silent = AudioSegment.silent(duration=int(max(50, args.lead_silence_ms) + target_dur * 1000))
            silent.export(out_seg, format="wav")
            return str(out_seg)
        finally:
            semaphore.release()

        # pad/trim to mapping timestamp duration
        padded = tmp_dir / f"seg_{idx}_out_final_padded.wav"
        pad_or_trim_keep_first(str(out_seg), target_dur, str(padded), end_pad_ms=args.end_pad_ms, start_pad_ms=args.lead_silence_ms)
        try:
            out_seg.unlink(missing_ok=True)
        except Exception:
            pass
        padded.rename(out_seg)
        return str(out_seg)

    # Submit tasks in timestamp order but threads will perform work; we collect results keyed by start time order
    print("Starting synthesis for mapping entries... (per-segment files saved to:", tmp_dir, ")")
    conv_results = {}
    with ThreadPoolExecutor(max_workers=max(1, args.gpu_workers)) as ex:
        fut_map = {}
        for entry in ordered_entries:
            sidx = int(entry["index"])
            fut = ex.submit(synth_for_entry, entry)
            fut_map[fut] = (sidx, entry)
        for fut in as_completed(fut_map):
            sidx, entry = fut_map[fut]
            try:
                res = fut.result()
                if res:
                    conv_results[sidx] = {"path": res, "start_s": float(entry.get("start_s", 0.0))}
                else:
                    print(f"[warn] synthesis returned no result for idx={sidx}")
            except Exception as e:
                print(f"[error] synthesis failed for idx={sidx}: {e}")

    if not conv_results:
        raise SystemExit("No produced segments; aborting.")

    # Order results by start time to preserve timeline
    ordered = sorted(conv_results.items(), key=lambda kv: conv_results[kv[0]]["start_s"])
    converted_segments = [v["path"] for k, v in ordered]

    if not converted_segments:
        raise SystemExit("No converted segments were produced; aborting.")

    # Create ffmpeg concat list and stitch
    concat_list = tmp_dir / "ff_concat_list_reencode.txt"
    with open(concat_list, "w") as fh:
        for f in converted_segments:
            fh.write(f"file '{Path(f).resolve()}'\n")

    final_out = Path(args.out).resolve()
    cmd = f"ffmpeg -y -f concat -safe 0 -i {shlex.quote(str(concat_list))} -c:a pcm_s16le -ar 16000 -ac 1 {shlex.quote(str(final_out))}"
    print("Running ffmpeg concat to produce final output at:", final_out)
    subprocess.check_call(cmd, shell=True)
    print("Final output written to:", final_out)

    # Cleanup
    if not args.no_clean:
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    print("Done.")

if __name__ == "__main__":
    main()

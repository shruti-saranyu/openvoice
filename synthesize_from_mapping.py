#!/usr/bin/env python3
"""
synthesize_from_mapping.py
Usage:
  python synthesize_from_mapping.py --mapping mapping.json --out output.wav --device cpu

Assumptions:
 - model returns float32 waveform at 24000 Hz
 - ref audio files exist and their transcripts are in mapping.json as `ref_text`
 - Python packages: transformers, soundfile, numpy, pydub, tqdm
"""
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import soundfile as sf

# pip: pydub uses ffmpeg, ensure ffmpeg installed on system
from pydub import AudioSegment

# transformers import
from transformers import AutoModel

SAMPLE_RATE = 24000  # IndicF5 default

def load_model(repo_id="ai4bharat/IndicF5", device="cpu"):
    print(f"Loading model {repo_id} (trust_remote_code=True) on {device} ...")
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    # many custom model wrappers accept .to(device) — try if available
    try:
        model.to(device)
    except Exception:
        pass
    return model

def read_audio_as_float(path, target_sr=SAMPLE_RATE):
    # use pydub to load (handles many formats) -> pydub returns ms-based durations
    audio = AudioSegment.from_file(path)
    # convert to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)
    # set frame rate (resample) if needed
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    # normalize from int to float (-1.0..1.0)
    sample_width = audio.sample_width  # bytes per sample
    max_val = float(2 ** (8 * sample_width - 1))
    samples = samples / max_val
    return samples, target_sr

def ensure_length(wav, sr, target_seconds):
    target_samples = int(round(target_seconds * sr))
    if len(wav) == target_samples:
        return wav
    if len(wav) < target_samples:
        # pad with small silence
        pad = np.zeros(target_samples - len(wav), dtype=wav.dtype)
        return np.concatenate([wav, pad], axis=0)
    else:
        # trim
        return wav[:target_samples]

def synth_segment(model, text, ref_audio_path=None, ref_text=None, device="cpu"):
    """
    Calls model(...) similar to the basic example.
    Returns numpy float32 waveform (mono) at SAMPLE_RATE.
    """
    call_kwargs = {}
    if ref_audio_path:
        call_kwargs["ref_audio_path"] = ref_audio_path
    if ref_text:
        call_kwargs["ref_text"] = ref_text

    # Many HF trust_remote_code models accept direct call: model(text, **kwargs)
    out = model(text, **call_kwargs)

    # `out` may be numpy array or torch tensor; normalize to np float32
    if hasattr(out, "numpy"):
        wav = out.numpy()
    elif isinstance(out, np.ndarray):
        wav = out
    else:
        # try to extract .wav or .audio attribute
        if hasattr(out, "audio"):
            candidate = out.audio
            if hasattr(candidate, "numpy"):
                wav = candidate.numpy()
            elif isinstance(candidate, np.ndarray):
                wav = candidate
            else:
                raise RuntimeError("Unknown model output structure.")
        else:
            raise RuntimeError("Unknown model output type: %s" % type(out))

    # if int16 -> convert
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    wav = wav.astype(np.float32)

    # If multi-channel, convert to mono by averaging
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    return wav

def main(args):
    mapping_path = Path(args.mapping)
    assert mapping_path.exists(), f"{mapping_path} not found"

    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    # load model once
    model = load_model(device=args.device)

    # prepare output buffer length: take max end_s across mapping
    max_end = 0.0
    entries = []
    for k, v in mapping.items():
        start_s = float(v.get("start_s", 0.0))
        end_s = float(v.get("end_s", start_s + 5.0))
        if end_s > max_end:
            max_end = end_s
        entries.append((int(v["index"]), v))
    entries.sort(key=lambda x: x[0])

    total_samples = int(round(max_end * SAMPLE_RATE)) + 1
    final = np.zeros(total_samples, dtype=np.float32)

    for idx, v in tqdm(entries, desc="synth segments"):
        text = v.get("text", "").strip()
        ref_text = v.get("ref_text", None)
        ref_audio = v.get("file", None)
        start_s = float(v.get("start_s", 0.0))
        end_s = float(v.get("end_s", start_s + 1.0))
        seg_len = end_s - start_s
        if not text:
            print(f"[WARN] index {idx} has empty target text — skipping")
            continue

        # check ref audio path (make relative to mapping file)
        if ref_audio and not os.path.isabs(ref_audio):
            ref_audio_path = (mapping_path.parent / ref_audio).resolve()
        else:
            ref_audio_path = Path(ref_audio) if ref_audio else None

        if ref_audio_path and not ref_audio_path.exists():
            print(f"[WARN] ref audio {ref_audio_path} not found for index {idx}, synthesizing without ref")
            ref_audio_path = None

        try:
            wav = synth_segment(model, text, ref_audio_path=str(ref_audio_path) if ref_audio_path else None, ref_text=ref_text, device=args.device)
        except Exception as e:
            print(f"[ERROR] synthesis failed for idx {idx}: {e} -- skipping")
            continue

        # Ensure sr == SAMPLE_RATE, if model returns different sr attempt resample via pydub
        # (convert to AudioSegment and back) — only if needed
        # length adjustments (pad/trim) to match segment length
        wav = ensure_length(wav, SAMPLE_RATE, seg_len)

        s_idx = int(round(start_s * SAMPLE_RATE))
        e_idx = s_idx + len(wav)
        # Mix additive (if overlap) — simple additive mixing with clipping prevention
        final[s_idx:e_idx] = final[s_idx:e_idx] + wav
        # avoid clipping >1 or <-1 by simple normalization later

    # clamp / normalize to -0.99 .. 0.99 if needed
    peak = np.abs(final).max()
    if peak > 0.99:
        print(f"[INFO] peak {peak:.3f} > 0.99 — normalizing output")
        final = final / (peak + 1e-9) * 0.99

    # write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), final.astype(np.float32), SAMPLE_RATE)
    print(f"Saved stitched audio -> {out_path.resolve()} (sr={SAMPLE_RATE})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mapping", type=str, default="mapping.json", help="path to mapping.json")
    p.add_argument("--out", type=str, default="output.wav", help="output WAV path")
    p.add_argument("--device", type=str, default="cpu", help="device for model (cpu or cuda)")
    args = p.parse_args()
    main(args)

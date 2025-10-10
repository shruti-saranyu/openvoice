#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clone_from_srt.py — updated: first-word-safe + timestamp-aware speed adjustment

Behavior:
 - Prepend a short lead silence to each TTS to prevent losing the first phoneme.
 - Pad/trim each segment to the subtitle duration.
 - If TTS > subtitle duration, try to speed it up to fit (subject to caps).
 - For very short segments, keep speed near normal to avoid artifacts.
 - Parallel TTS generation with --tts-workers, optional parallel GPU conversions with --gpu-workers.
 - Sanitizes SRT text so gTTS isn't given punctuation-only strings.
 - Optionally cleans tmp_clone, my_voices and label_map.json after successful run (unless --no-clean supplied).
"""

import argparse
import json
import os
import re
import glob
import shlex
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import threading
import math
import tempfile
import html
import unicodedata
import shutil

from gtts import gTTS
from pydub import AudioSegment
import torch

from openvoice.api import ToneColorConverter
from openvoice import se_extractor


# ---------------------- SRT parsing ----------------------
def parse_srt_simple(srt_path: str):
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
        if "|" in time_line:
            time_part, *meta = [s.strip() for s in time_line.split("|")]
        else:
            time_part, meta = time_line.strip(), []
        m = re.search(r"(\d{2}:\d{2}:\d{2}[,\.]\d+)\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d+)", time_part)
        if not m:
            continue

        def to_sec(t):
            hh, mm, ss_ms = t.split(":")
            ss, ms = re.split(r"[,.]", ss_ms)
            return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

        start, end = to_sec(m.group(1)), to_sec(m.group(2))
        idx = lines.index(time_line)
        text_body = " ".join(lines[idx + 1 :]).strip()
        label = meta[0].strip() if meta else None
        blocks.append({"start": start, "end": end, "text": text_body, "label": label})
    return blocks


# ---------------------- helpers ----------------------
def detect_speaker_label(text: str):
    """
    Conservative inline label detection.

    Returns (label, rest_text) if an inline label is confidently detected,
    otherwise (None, original_text).
    """
    if not text or not isinstance(text, str):
        return None, text

    txt = text.strip()

    # 1) bracketed explicit: [Speaker 1] Some text
    m = re.match(r'^\[\s*(?P<label>[^]\r\n]+?)\s*\]\s*(?P<rest>.*)', txt, flags=re.I)
    if m:
        return m.group('label').strip(), m.group('rest').strip()

    # 2) explicit speaker-like prefixes: "speaker 1", "spkr-2", "spk_3"
    m = re.match(r'^(?P<label>(?:speaker|spkr|spk)\s*[\-_\d\w]+)\s*[:\-\|]?\s*(?P<rest>.*)$',
                 txt, flags=re.I)
    if m:
        return m.group('label').strip(), m.group('rest').strip()

    # 3) general "Label: rest" or "Label - rest" — but only accept if left-hand candidate is short and clean
    m = re.match(r'^(?P<label_candidate>[^:\-\|]{1,40})\s*[:\-\|]\s*(?P<rest>.*)$', txt)
    if m:
        cand = m.group('label_candidate').strip()
        if len(cand) <= 30 and re.search(r'[\.\,\?\!;\/\\\(\)\[\]\{\}]', cand) is None:
            words = [w for w in re.split(r'\s+', cand) if w]
            if 1 <= len(words) <= 4:
                return cand, m.group('rest').strip()

    return None, text


# ---------- text sanitization ----------
def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_punctuation_edges(s: str) -> str:
    # remove leading/trailing punctuation but keep internal punctuation (commas, apostrophes)
    return re.sub(r"^[\W_]+|[\W_]+$", "", s, flags=re.UNICODE)


def sanitize_for_tts(s: str) -> str:
    """
    Clean SRT text to be safe for TTS.
    """
    if s is None:
        s = ""
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"^(?:speaker\s*\d+[:\-\)]\s*)", "", s, flags=re.I)
    s = normalize_whitespace(s)
    s = strip_punctuation_edges(s)
    s = normalize_whitespace(s)
    if s == "" or re.fullmatch(r"[\W_]+", s or ""):
        return ""
    return s


def synthesize_tts_with_lead(text: str, out_wav: str, lang: str = "en", lead_silence_ms: int = 120):
    """
    Synthesize TTS via gTTS, prepend lead silence to avoid missing first phonemes.
    Writes WAV to out_wav.
    """
    text = text.strip()
    if text == "":
        AudioSegment.silent(duration=max(50, lead_silence_ms)).export(out_wav, format="wav")
        return out_wav

    tmp_mp3 = out_wav + ".tmp.mp3"
    gTTS(text=text, lang=lang).save(tmp_mp3)
    seg = AudioSegment.from_file(tmp_mp3)
    lead = AudioSegment.silent(duration=lead_silence_ms)
    seg2 = lead + seg
    seg2.export(out_wav, format="wav")
    try:
        os.remove(tmp_mp3)
    except Exception:
        pass
    return out_wav


def audio_duration_s(in_wav: str) -> float:
    seg = AudioSegment.from_file(in_wav)
    return len(seg) / 1000.0


def pad_or_trim_keep_first(in_wav: str, target_sec: float, out_wav: str, end_pad_ms: int = 30, start_pad_ms: int = 30):
    """
    Ensure the output has duration close to target_sec.
    """
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


def ensure_embedding_for_ref(ref_wav: str, pth_out: str, tcc):
    if Path(pth_out).exists():
        return pth_out
    se, _ = se_extractor.get_se(ref_wav, tcc, vad=False)
    Path(pth_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(se, pth_out)
    return pth_out


def load_embedding(path: str, device: str):
    obj = torch.load(path, map_location="cpu")
    t = None
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                t = v
                break
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, torch.Tensor):
                        t = vv
                        break
            if t is not None:
                break
    if t is None:
        raise RuntimeError(f"No tensor found in embedding: {path}")
    if t.ndim == 1:
        t = t.unsqueeze(0).unsqueeze(-1)
    elif t.ndim == 2:
        t = t.unsqueeze(0)
    return t.to(device).float()


def find_ref_for_label(label: Optional[str], extracted_dir="extracted_voices"):
    refs = sorted(glob.glob(os.path.join(extracted_dir, "*.wav")))
    if not refs:
        return None
    try:
        mapping_path = Path("label_map.json")
        if mapping_path.exists():
            mapping = json.loads(mapping_path.read_text())
            if label and label in mapping and Path(mapping[label]).exists():
                return mapping[label]
    except Exception:
        pass
    if not label:
        return refs[0]
    s = str(label).strip().lower()
    m = re.search(r"(?:speaker|spkr|spk)?\s*[-_]*\s*(\d+)", s)
    if m:
        num = m.group(1)
        cand = Path(extracted_dir) / f"speaker_{num}.wav"
        if cand.exists():
            return str(cand)
    def norm(x): return re.sub(r"[^0-9a-z]", "", x.lower())
    nlabel = norm(s)
    for r in refs:
        stem = Path(r).stem
        if nlabel and (nlabel in norm(stem) or norm(stem) in nlabel):
            return r
    tokens = re.split(r"\W+", s)
    for r in refs:
        stem = Path(r).stem.lower()
        for t in tokens:
            if t and (t in stem or stem in t):
                return r
    return refs[0]


# ---------------------- tempo adjust via ffmpeg ----------------------
def build_atempo_filters(speed: float):
    """
    Decompose desired speed >0 into a list of atempo multipliers each in [0.5,2.0].
    """
    if speed <= 0:
        raise ValueError("speed must be > 0")
    factors = []
    remaining = speed
    while remaining > 2.0 + 1e-9:
        factors.append(2.0)
        remaining /= 2.0
    if remaining < 0.5:
        remaining = 0.5
    factors.append(remaining)
    return ",".join([f"atempo={f:.6f}" for f in factors])


def change_tempo_ffmpeg(in_wav: str, out_wav: str, speed: float):
    """
    Change playback speed (tempo) using ffmpeg atempo filters (preserves pitch).
    """
    if abs(speed - 1.0) < 1e-6:
        subprocess.check_call(
            ["ffmpeg", "-y", "-i", in_wav, "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", out_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return out_wav

    max_speed = 4.0
    min_speed = 0.5
    s = max(min_speed, min(max_speed, speed))
    filter_str = build_atempo_filters(s)
    cmd = ["ffmpeg", "-y", "-i", in_wav, "-filter:a", filter_str, "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", out_wav]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_wav


def adjust_to_target_duration(
    src_wav: str,
    target_sec: float,
    out_wav: str,
    max_speed: float = 2.0,
    short_threshold: float = 1.0,
    short_max_speed: float = 1.15,
    lead_silence_ms: int = 120,
    end_pad_ms: int = 30,
):
    """
    Try to make src_wav fit into target_sec.
    """
    cur = audio_duration_s(src_wav)
    if cur <= target_sec + 1e-6:
        return pad_or_trim_keep_first(src_wav, target_sec, out_wav, end_pad_ms=end_pad_ms, start_pad_ms=lead_silence_ms)

    desired_factor = cur / target_sec
    allowed_max = max_speed
    if target_sec < short_threshold:
        allowed_max = min(allowed_max, short_max_speed)
    apply_speed = min(desired_factor, allowed_max)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
        tmp_path = tmpf.name
    try:
        change_tempo_ffmpeg(src_wav, tmp_path, apply_speed)
        return pad_or_trim_keep_first(tmp_path, target_sec, out_wav, end_pad_ms=end_pad_ms, start_pad_ms=lead_silence_ms)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ---------------------- safety helpers for cleanup ----------------------
def _is_safe_to_remove_dir(path: Path, repo_root: Path) -> bool:
    """
    Only allow removal if path exists, is a directory, and is inside the repo.
    """
    try:
        p_resolved = path.resolve()
        p_resolved.relative_to(repo_root.resolve())
    except Exception:
        return False
    if p_resolved == repo_root.resolve():
        return False
    return p_resolved.exists() and p_resolved.is_dir()


def _is_safe_to_remove_file(path: Path, repo_root: Path) -> bool:
    """
    Only allow file removal if file exists and is inside repo.
    """
    try:
        p_resolved = path.resolve()
        p_resolved.relative_to(repo_root.resolve())
    except Exception:
        return False
    if p_resolved == repo_root.resolve():
        return False
    return p_resolved.exists() and p_resolved.is_file()


# ---------------------- main pipeline ----------------------
def main():
    ap = argparse.ArgumentParser(description="Clone from SRT using OpenVoice v2 (first-word-safe + tempo adjust)")
    ap.add_argument("--srt", required=True)
    ap.add_argument("--extracted_dir", default="extracted_voices")
    ap.add_argument("--out", default="outputs_v2/kan2_cloned_final.wav")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--lead-silence-ms", type=int, default=120)
    ap.add_argument("--end-pad-ms", type=int, default=30)
    ap.add_argument("--tts-workers", type=int, default=4)
    ap.add_argument("--gpu-workers", type=int, default=1)
    ap.add_argument("--max-speed", type=float, default=2.0, help="global max speed multiplier")
    ap.add_argument("--short-threshold", type=float, default=1.0, help="short segments (<seconds) get conservative speed")
    ap.add_argument("--short-max-speed", type=float, default=1.15, help="max speed for short segments")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--reuse-base-embedding", action="store_true")
    ap.add_argument("--no-clean", action="store_true", help="Do not remove tmp_clone, my_voices or label_map.json after successful run (useful for debugging).")
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    repo = Path(".").resolve()
    tmp_dir = repo / "tmp_clone"
    tmp_dir.mkdir(exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    conv_cfg = "checkpoints_v2/converter/config.json"
    conv_ckpt = "checkpoints_v2/converter/checkpoint.pth"
    tcc = ToneColorConverter(conv_cfg, device=device)
    tcc.load_ckpt(conv_ckpt)
    print("Loaded converter")

    subs = parse_srt_simple(args.srt)
    if not subs:
        raise SystemExit("No subtitles parsed")

    ref_wavs = sorted(glob.glob(os.path.join(args.extracted_dir, "*.wav")))
    if not ref_wavs:
        raise SystemExit("No reference wavs found in " + args.extracted_dir)

    # prepare base embedding
    pth_dir = Path("checkpoints_v2/base_speakers/ses")
    pth_dir.mkdir(parents=True, exist_ok=True)
    base_pth = pth_dir / "base.pth"
    if not base_pth.exists() or not args.reuse_base_embedding:
        base_tmp = tmp_dir / "base_src.wav"
        synthesize_tts_with_lead(
            "This is a neutral base voice used for cloning experiments.",
            str(base_tmp),
            lang=args.lang,
            lead_silence_ms=args.lead_silence_ms,
        )
        ensure_embedding_for_ref(str(base_tmp), str(base_pth), tcc)
    else:
        print("Reusing existing base embedding:", base_pth)

    # Step 1: produce TTS for all segments in parallel
    padded_map = {}

    def tts_task(idx, text, dur):
        """
        Produces an adjusted padded WAV for segment `idx`.
        """
        src_tmp = tmp_dir / f"src_{idx}.wav"
        adjusted = tmp_dir / f"src_{idx}_pad_adj.wav"
        if args.skip_existing and Path(adjusted).exists():
            return str(adjusted)

        inline_label, inline_text = detect_speaker_label(text)
        raw_text = inline_text if inline_label else text
        clean_text = sanitize_for_tts(raw_text)

        if clean_text == "":
            silent = AudioSegment.silent(duration=int(max(50, args.lead_silence_ms) + (dur * 1000)))
            out_path = tmp_dir / f"src_{idx}_pad_adj.wav"
            silent.export(out_path, format="wav")
            return pad_or_trim_keep_first(str(out_path), dur, str(out_path), end_pad_ms=args.end_pad_ms, start_pad_ms=args.lead_silence_ms)

        synthesize_tts_with_lead(clean_text, str(src_tmp), lang=args.lang, lead_silence_ms=args.lead_silence_ms)

        adjusted_path = adjust_to_target_duration(
            str(src_tmp),
            dur,
            str(adjusted),
            max_speed=args.max_speed,
            short_threshold=args.short_threshold,
            short_max_speed=args.short_max_speed,
            lead_silence_ms=args.lead_silence_ms,
            end_pad_ms=args.end_pad_ms,
        )
        return adjusted_path

    with ThreadPoolExecutor(max_workers=max(1, args.tts_workers)) as ex:
        fut_map = {}
        for idx, block in enumerate(subs):
            meta_label = block.get("label")
            inline_label, inline_text = detect_speaker_label(block["text"])
            label = meta_label if meta_label else inline_label
            text = inline_text if inline_label else block["text"]
            dur = max(0.05, block["end"] - block["start"])
            fut = ex.submit(tts_task, idx, text, dur)
            fut_map[fut] = idx
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                padded_map[idx] = fut.result()
            except Exception as e:
                print("TTS failed for idx", idx, ":", e)

    # Step 2: convert segments with optional parallelism (cap via semaphore)
    semaphore = threading.Semaphore(max(1, args.gpu_workers))
    base_se_tensor = load_embedding(str(base_pth), device)

    def convert_task(idx, padded_path, label):
        out_seg = tmp_dir / f"seg_{idx}_out_final.wav"
        if args.skip_existing and Path(out_seg).exists():
            print(f"  [skip] segment {idx} exists -> {out_seg}")
            return str(out_seg)
        ref_wav = find_ref_for_label(label, args.extracted_dir)
        if not ref_wav:
            print("  No ref for label", label, "skipping")
            return None
        ref_stem = Path(ref_wav).stem
        tgt_pth = pth_dir / f"{ref_stem}.pth"
        if not tgt_pth.exists():
            print("  extracting target embedding from", ref_wav)
            ensure_embedding_for_ref(ref_wav, str(tgt_pth), tcc)
        else:
            print("  using existing target embedding:", tgt_pth)
        tgt_se = load_embedding(str(tgt_pth), device)
        semaphore.acquire()
        try:
            print(f"  converting idx={idx} -> {out_seg} (tau={args.tau})")
            tcc.convert(
                audio_src_path=str(padded_path),
                src_se=base_se_tensor,
                tgt_se=tgt_se,
                output_path=str(out_seg),
                tau=args.tau,
            )
            return str(out_seg)
        except Exception as e:
            print("  conversion failed for idx", idx, ":", e)
            return None
        finally:
            semaphore.release()

    # submit conversion jobs and collect results while the executor is active
    conv_results = {}
    with ThreadPoolExecutor(max_workers=max(1, args.gpu_workers)) as conv_ex:
        conv_fut_map = {}
        for idx, block in enumerate(subs):
            meta_label = block.get("label")
            inline_label, inline_text = detect_speaker_label(block["text"])
            label = meta_label if meta_label else inline_label
            text = inline_text if inline_label else block["text"]
            dur = max(0.05, block["end"] - block["start"])
            padded = padded_map.get(idx)
            print(f"[{idx}] label={label} ref_guess={find_ref_for_label(label, args.extracted_dir)} dur={dur:.2f}s text='{text[:60]}'")
            if not padded:
                print("  Missing padded/adjusted TTS for idx", idx, " — skipping")
                continue
            fut = conv_ex.submit(convert_task, idx, padded, label)
            conv_fut_map[fut] = idx

        for fut in as_completed(conv_fut_map):
            idx = conv_fut_map[fut]
            try:
                res = fut.result()
                if res:
                    conv_results[idx] = res
                else:
                    print(f"  conversion returned no result for idx={idx}")
            except Exception as e:
                print(f"  conversion failed for idx={idx}: {e}")

    # Assemble converted_segments in chronological order (by subtitle index)
    if not conv_results:
        raise SystemExit("No converted segments produced; aborting.")

    converted_segments = [conv_results[i] for i in sorted(conv_results.keys())]

    if not converted_segments:
        raise SystemExit("No converted segments produced; aborting. ")

    # Step 3: stitch results
    concat_list = tmp_dir / "ff_concat_list_reencode.txt"
    with open(concat_list, "w") as fh:
        for f in converted_segments:
            fh.write(f"file '{Path(f).resolve()}'\n")

    final_out = Path(args.out).resolve()
    cmd = f"ffmpeg -y -f concat -safe 0 -i {shlex.quote(str(concat_list))} -c:a pcm_s16le -ar 16000 -ac 1 {shlex.quote(str(final_out))}"
    print("Running concat ->", final_out)
    subprocess.check_call(cmd, shell=True)
    print("Wrote final output:", final_out)
    print("Done.")

    # ------------------ SAFE POST-RUN CLEANUP ------------------
    # if args.no_clean:
    #     print("[cleanup] --no-clean set; skipping removal of tmp_clone, my_voices, and label_map.json.")
    # else:
    #     # Remove tmp_clone and my_voices directories safely (only if inside repo)
    #     targets_dirs = [repo / "tmp_clone", repo / "my_voices"]
    #     for t in targets_dirs:
    #         if _is_safe_to_remove_dir(t, repo):
    #             try:
    #                 print(f"[cleanup] removing {t} ...")
    #                 shutil.rmtree(t, ignore_errors=True)
    #                 print(f"[cleanup] removed {t}")
    #             except Exception as e:
    #                 print(f"[cleanup] warning removing {t}: {e}")
    #         else:
    #             if t.exists():
    #                 print(f"[cleanup] NOT removing {t} (unsafe or outside repo)")
    #             else:
    #                 print(f"[cleanup] {t} does not exist; nothing to remove")

    #     # Recreate empty tmp_clone and my_voices so the next run starts fresh
    #     for t in targets_dirs:
    #         try:
    #             t.mkdir(parents=True, exist_ok=True)
    #             print(f"[cleanup] created empty directory {t}")
    #         except Exception as e:
    #             print(f"[cleanup] failed creating {t}: {e}")

    #     # Remove label_map.json in repo root and inside extracted_dir (if present and safe)
    #     repo_label = repo / "label_map.json"
    #     extracted_label = Path(args.extracted_dir) / "label_map.json"

    #     for label_file in [repo_label, extracted_label]:
    #         if _is_safe_to_remove_file(label_file, repo):
    #             try:
    #                 print(f"[cleanup] removing label map file {label_file} ...")
    #                 label_file.unlink(missing_ok=True)
    #                 print(f"[cleanup] removed {label_file}")
    #             except Exception as e:
    #                 print(f"[cleanup] warning removing label map file {label_file}: {e}")
    #         else:
    #             if label_file.exists():
    #                 print(f"[cleanup] NOT removing {label_file} (unsafe or outside repo)")
    #             else:
    #                 print(f"[cleanup] {label_file} does not exist; nothing to remove")

    # End cleanup

if __name__ == "__main__":
    main()

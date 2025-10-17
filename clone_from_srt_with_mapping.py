#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clone_from_srt_with_mapping_indian_english_v2.py

Same behaviour as the clone_from_srt_with_mapping.py script but all user-facing
text, docstrings and comments are localised to Indian English and the script
is updated for OpenVoice v2 usage.

Example (hypothetical) equivalent API usage for an Indian-accent TTS call:
# This is a hypothetical API call, but demonstrates the concept.
client.generate_speech(
    text="Hello, how are you?",
    language="English",
    accent="Indian",
    version="v2"
)

Usage:
    python clone_from_srt_with_mapping_indian_english_v2.py \
        --srt subs.srt --mapping mapping.json --voices-dir my_voices \
        --extracted_dir extracted_voices --out outputs_v2/final.wav
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

# -------------------- (SRT parsing and helper functions; unchanged logic) --------------------
def parse_srt_simple(srt_path: str):
    """
    Parse a simple SRT file and return blocks with start/end seconds, text, label and index.
    The parser accepts blocks where the time-line may contain metadata separated by '|'.
    """
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
        idx = None
        if lines and re.fullmatch(r"\d+", lines[0]):
            idx = int(lines[0])
        idx = idx or -1
        idx_timepos = lines.index(time_line)
        text_body = " ".join(lines[idx_timepos + 1 :]).strip()
        label = meta[0].strip() if meta else None
        blocks.append({"start": start, "end": end, "text": text_body, "label": label, "index": idx})
    return blocks

# ---------------------- text helpers (behaviour unchanged) ----------------------
def detect_speaker_label(text: str):
    """
    Conservative inline label detection. Returns (label, rest_text) or (None, text).
    """
    if not text or not isinstance(text, str):
        return None, text
    txt = text.strip()
    m = re.match(r'^\[\s*(?P<label>[^]\r\n]+?)\s*\]\s*(?P<rest>.*)', txt, flags=re.I)
    if m:
        return m.group('label').strip(), m.group('rest').strip()
    m = re.match(r'^(?P<label>(?:speaker|spkr|spk)\s*[\-_\d\w]+)\s*[:\-\|]?\s*(?P<rest>.*)$',
                 txt, flags=re.I)
    if m:
        return m.group('label').strip(), m.group('rest').strip()
    m = re.match(r'^(?P<label_candidate>[^:\-\|]{1,40})\s*[:\-\|]\s*(?P<rest>.*)$', txt)
    if m:
        cand = m.group('label_candidate').strip()
        if len(cand) <= 30 and re.search(r'[\.\,\?\!;\/\\\(\)\[\]\{\}]', cand) is None:
            words = [w for w in re.split(r'\s+', cand) if w]
            if 1 <= len(words) <= 4:
                return cand, m.group('rest').strip()
    return None, text

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def strip_punctuation_edges(s: str) -> str:
    return re.sub(r"^[\W_]+|[\W_]+$", "", s, flags=re.UNICODE)

def sanitize_for_tts(s: str) -> str:
    """
    Clean SRT text to be safe for TTS engines.
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

def synthesize_tts_with_lead(text: str, out_wav: str, lang: str = "en-in", lead_silence_ms: int = 120):
    """
    Use gTTS to produce speech and prepend a short lead silence to avoid cutting
    the very first phoneme. If the text is empty, produce a short silent file.

    For Indian-accent English, use tld='co.in' with gTTS. We accept lang codes
    like 'en' or 'en-in' and map en-in -> en with tld co.in to bias Google TTS
    towards an Indian accent.
    """
    text = text.strip()
    if text == "":
        AudioSegment.silent(duration=max(50, lead_silence_ms)).export(out_wav, format="wav")
        return out_wav

    tmp_mp3 = out_wav + ".tmp.mp3"
    # Map 'en-in' to gTTS parameters (lang='en', tld='co.in') to favour Indian accent.
    tld = None
    gtts_lang = lang
    if isinstance(lang, str) and lang.lower() in ("en-in", "en_indian", "indian-english"):
        gtts_lang = "en"
        tld = "co.in"
    try:
        if tld:
            gTTS(text=text, lang=gtts_lang, tld=tld).save(tmp_mp3)
        else:
            gTTS(text=text, lang=gtts_lang).save(tmp_mp3)
        seg = AudioSegment.from_file(tmp_mp3)
    except Exception as e:
        # Fallback: if gTTS fails for any reason, produce silence and raise a helpful message.
        AudioSegment.silent(duration=max(50, lead_silence_ms)).export(out_wav, format="wav")
        raise RuntimeError(f"gTTS failed to synthesise text: {e}")

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
    Ensure the output has duration close to target_sec while preserving
    the initial phoneme (by adding leading silence).
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

# ---------------------- embedding helpers ----------------------
def ensure_embedding_for_ref(ref_wav: str, pth_out: str, tcc):
    """
    Extract and save a speaker embedding for the given reference WAV if not present.
    """
    if Path(pth_out).exists():
        return pth_out
    se, _ = se_extractor.get_se(ref_wav, tcc, vad=False)
    Path(pth_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(se, pth_out)
    return pth_out

def load_embedding(path: str, device: str):
    """
    Load a saved embedding tensor from disk and normalise its shape for the converter.
    """
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

# ---------------------- tempo adjust via ffmpeg ----------------------
def build_atempo_filters(speed: float):
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
    Change playback speed while preserving pitch using ffmpeg atempo filter.
    """
    if abs(speed - 1.0) < 1e-6:
        subprocess.check_call(
            ["ffmpeg", "-y", "-i", in_wav, "-c:a", "pcm_s16le", "-ar", 16000, "-ac", "1", out_wav],
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
    Attempt to fit src_wav into target_sec by speeding up (within caps) or padding/trimming.
    Conservative for very short fragments to avoid distortion.
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

# ---------------------- safety helpers ----------------------
def _is_safe_to_remove_dir(path: Path, repo_root: Path) -> bool:
    try:
        p_resolved = path.resolve()
        p_resolved.relative_to(repo_root.resolve())
    except Exception:
        return False
    if p_resolved == repo_root.resolve():
        return False
    return p_resolved.exists() and p_resolved.is_dir()

def _is_safe_to_remove_file(path: Path, repo_root: Path) -> bool:
    try:
        p_resolved = path.resolve()
        p_resolved.relative_to(repo_root.resolve())
    except Exception:
        return False
    if p_resolved == repo_root.resolve():
        return False
    return p_resolved.exists() and p_resolved.is_file()

# ---------------------- mapping helpers ----------------------
def load_mapping_json(mapping_path: Path):
    """
    Load mapping.json produced by the mapping step and return both raw and
    a dict keyed by subtitle index for convenient lookup.
    """
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    by_index = {}
    for k, v in raw.items():
        idx = v.get("srt_index", None)
        if idx is None:
            try:
                idx = int(k.split("_")[0])
            except Exception:
                idx = None
        if idx is not None:
            by_index[int(idx)] = v
    return raw, by_index

def pick_best_segment_for_subtitle(mapping_entry: dict):
    """
    Choose the matched segment with the largest overlap for this subtitle.
    Returns the path string or None.
    """
    matched = mapping_entry.get("matched_segments") or []
    if matched:
        def score(m):
            ov = m.get("overlap_ms", 0)
            segdur = abs(m.get("segment_end_ms", 0) - m.get("segment_start_ms", 0))
            return (ov, segdur)
        best = max(matched, key=score)
        return best.get("segment_path")
    return None

def pick_longest_in_speaker_dir(voices_dir: Path, speaker_dir_name: str):
    """
    From voices_dir/speaker_dir_name choose the WAV file with the longest duration.
    """
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

# ---------------------- main pipeline ----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Clone from SRT using OpenVoice v2 (mapping.json-based references). Localised to Indian English.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--srt", required=True, help="Path to the SRT file.")
    ap.add_argument("--mapping", required=True, help="mapping.json produced by the mapping step.")
    ap.add_argument("--voices-dir", default="my_voices", help="Directory containing speaker_N/ folders.")
    ap.add_argument("--extracted_dir", default="extracted_voices", help="Fallback directory with extracted reference WAVs.")
    ap.add_argument("--out", default="outputs_v2/kan2_cloned_final.wav", help="Final stitched output WAV.")
    ap.add_argument("--lang", default="en-in", help="Language code for gTTS (use 'en-in' for Indian English).")
    ap.add_argument("--tau", type=float, default=0.4, help="Conversion tau parameter.")
    ap.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    ap.add_argument("--lead-silence-ms", type=int, default=120, help="Lead silence prepended to TTS (ms).")
    ap.add_argument("--end-pad-ms", type=int, default=30, help="Pad at segment end (ms).")
    ap.add_argument("--tts-workers", type=int, default=4, help="Number of parallel TTS workers.")
    ap.add_argument("--gpu-workers", type=int, default=1, help="Number of parallel GPU conversion workers.")
    ap.add_argument("--max-speed", type=float, default=2.0, help="Global maximum speed multiplier for tempo adjust.")
    ap.add_argument("--short-threshold", type=float, default=1.0, help="Short segment threshold (s) for conservative speed.")
    ap.add_argument("--short-max-speed", type=float, default=1.15, help="Max speed for very short segments.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip regeneration when output files already exist.")
    ap.add_argument("--reuse-base-embedding", action="store_true", help="Reuse an existing base embedding if present.")
    ap.add_argument("--no-clean", action="store_true", help="Do not remove tmp_clone and my_voices after run (for debugging).")
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device chosen:", device)
    repo = Path(".").resolve()
    tmp_dir = repo / "tmp_clone"
    tmp_dir.mkdir(exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Load converter configuration and weights (OpenVoice v2)
    conv_cfg = "checkpoints_v2/converter/config.json"
    conv_ckpt = "checkpoints_v2/converter/checkpoint.pth"
    # If your repo placed converter weights elsewhere, update conv_cfg / conv_ckpt accordingly.
    tcc = ToneColorConverter(conv_cfg, device=device)
    tcc.load_ckpt(conv_ckpt)
    print("Converter (OpenVoice v2) loaded successfully.")

    subs = parse_srt_simple(args.srt)
    if not subs:
        raise SystemExit("No subtitles parsed from the SRT provided; please check the file format.")

    # Load mapping.json
    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        raise SystemExit(f"mapping.json not found at: {mapping_path}. Please generate mapping first.")
    raw_mapping, mapping_by_index = load_mapping_json(mapping_path)
    print(f"Loaded mapping.json with {len(mapping_by_index)} indexed entries.")

    voices_dir = Path(args.voices_dir)

    def select_ref_for_sub(idx, label_hint=None):
        """
        Choose the best reference WAV for a given subtitle index.
        Priority:
         1) mapping entry matched_segments (largest overlap)
         2) longest WAV in the speaker directory from voices_dir
         3) fallback to extracted_dir (old behaviour)
        """
        ent = mapping_by_index.get(int(idx))
        if ent:
            best_seg = pick_best_segment_for_subtitle(ent)
            if best_seg and Path(best_seg).exists():
                return best_seg
            spk = ent.get("speaker_dir")
            if spk:
                cand = pick_longest_in_speaker_dir(voices_dir, spk)
                if cand:
                    return cand
        # fallback using label hint
        if label_hint:
            m = re.search(r"(?:speaker|spkr|spk)?\s*[-_]*\s*(\d+)", str(label_hint), flags=re.I)
            if m:
                cand = voices_dir / f"speaker_{m.group(1)}.wav"
                if cand.exists():
                    return str(cand)
            cand2 = pick_longest_in_speaker_dir(voices_dir, str(label_hint))
            if cand2:
                return cand2
        # final fallback to extracted_dir
        refs = sorted(glob.glob(os.path.join(args.extracted_dir, "*.wav")))
        if refs:
            return refs[0]
        return None

    # Prepare base embedding (neutral voice) if absent
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
        print("Created base embedding for neutral source voice.")
    else:
        print("Reusing existing base embedding at:", base_pth)

    # Step 1: Produce TTS for all subtitles in parallel
    padded_map = {}

    def tts_task(idx, text, dur):
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

    print("Starting TTS generation for each subtitle (this may take some time).")
    with ThreadPoolExecutor(max_workers=max(1, args.tts_workers)) as ex:
        fut_map = {}
        for idx, block in enumerate(subs):
            sidx = block.get("index", idx)
            meta_label = block.get("label")
            inline_label, inline_text = detect_speaker_label(block["text"])
            label = meta_label if meta_label else inline_label
            text = inline_text if inline_label else block["text"]
            dur = max(0.05, block["end"] - block["start"])
            fut = ex.submit(tts_task, sidx, text, dur)
            fut_map[fut] = sidx
        for fut in as_completed(fut_map):
            sidx = fut_map[fut]
            try:
                padded_map[sidx] = fut.result()
            except Exception as e:
                print(f"TTS failed for subtitle index {sidx}:", e)

    # Step 2: Convert via OpenVoice converter, with limited parallel GPU workers
    semaphore = threading.Semaphore(max(1, args.gpu_workers))
    base_se_tensor = load_embedding(str(base_pth), device)

    def convert_task(idx, padded_path):
        out_seg = tmp_dir / f"seg_{idx}_out_final.wav"
        if args.skip_existing and Path(out_seg).exists():
            print(f"[skip] segment {idx} exists -> {out_seg}")
            return str(out_seg)
        ref_wav = select_ref_for_sub(idx)
        if not ref_wav:
            print(f"No reference audio found for subtitle idx {idx}; skipping that subtitle.")
            return None
        ref_stem = Path(ref_wav).stem
        tgt_pth = pth_dir / f"{ref_stem}.pth"
        if not tgt_pth.exists():
            print("Extracting target embedding from", ref_wav)
            ensure_embedding_for_ref(ref_wav, str(tgt_pth), tcc)
        else:
            print("Using existing target embedding:", tgt_pth)
        tgt_se = load_embedding(str(tgt_pth), device)
        semaphore.acquire()
        try:
            print(f"Converting idx={idx} -> {out_seg} (tau={args.tau}) using ref {ref_wav}")
            tcc.convert(
                audio_src_path=str(padded_path),
                src_se=base_se_tensor,
                tgt_se=tgt_se,
                output_path=str(out_seg),
                tau=args.tau,
            )
            return str(out_seg)
        except Exception as e:
            print(f"Conversion failed for idx {idx}:", e)
            return None
        finally:
            semaphore.release()

    print("Starting conversion to cloned voice for each subtitle.")
    conv_results = {}
    with ThreadPoolExecutor(max_workers=max(1, args.gpu_workers)) as conv_ex:
        conv_fut_map = {}
        for idx, block in enumerate(subs):
            sidx = block.get("index", idx)
            padded = padded_map.get(sidx)
            meta_label = block.get("label")
            inline_label, inline_text = detect_speaker_label(block["text"])
            label = meta_label if meta_label else inline_label
            dur = max(0.05, block["end"] - block["start"])
            print(f"[{sidx}] label={label} dur={dur:.2f}s text='{(block['text'] or '')[:60]}'")
            if not padded:
                print(f"Missing padded TTS for idx {sidx} — skipping conversion for this subtitle.")
                continue
            fut = conv_ex.submit(convert_task, sidx, padded)
            conv_fut_map[fut] = sidx

        for fut in as_completed(conv_fut_map):
            sidx = conv_fut_map[fut]
            try:
                res = fut.result()
                if res:
                    conv_results[sidx] = res
                else:
                    print(f"Conversion returned no result for idx={sidx}")
            except Exception as e:
                print(f"Conversion failed for idx={sidx}:", e)

    # Assemble converted segments in chronological order by subtitle index
    if not conv_results:
        raise SystemExit("No converted segments were produced; aborting.")

    converted_segments = [conv_results[i] for i in sorted(conv_results.keys())]

    if not converted_segments:
        raise SystemExit("No converted segments were produced; aborting.")

    # Step 3: Stitch all converted segments to produce final WAV
    concat_list = tmp_dir / "ff_concat_list_reencode.txt"
    with open(concat_list, "w") as fh:
        for f in converted_segments:
            fh.write(f"file '{Path(f).resolve()}'\n")

    final_out = Path(args.out).resolve()
    cmd = f"ffmpeg -y -f concat -safe 0 -i {shlex.quote(str(concat_list))} -c:a pcm_s16le -ar 16000 -ac 1 {shlex.quote(str(final_out))}"
    print("Running ffmpeg concat to produce final output at:", final_out)
    subprocess.check_call(cmd, shell=True)
    print("Final output written to:", final_out)
    print("All done — process completed successfully. Thank you.")

    # Optional safe cleanup could be reinstated here if desired (currently omitted by default).
    if not args.no_clean:
        # The original script had careful safety checks before removing directories.
        # If user prefers automatic cleanup, we may re-enable it here.
        pass

if __name__ == "__main__":
    main()

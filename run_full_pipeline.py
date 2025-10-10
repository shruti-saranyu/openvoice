#!/usr/bin/env python3
"""
run_full_pipeline.py

Wrapper that:
  1) runs automate_extract_and_split.py -> produces extracted_voices/* and label_map.json
  2) normalizes / reconciles label_map.json with the SRT labels and writes canonical label_map.json
  3) (optionally) backs up the canonical label_map.json
  4) runs clone_from_srt.py using the extracted_dir, forwarding --no-clean if requested

Usage:
  export HF_TOKEN=hf_xxx
  python run_full_pipeline.py --input input/source_audio.wav --srt srt/translated.srt

New flags:
  --no-clean           Forwarded to clone_from_srt.py; prevents it from deleting tmp_clone/my_voices/label_map.json.
  --backup-label-map   Create repo-root backup file 'label_map.backup.json' before running clone_from_srt.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Set, Tuple, List
import shutil

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR  # repo root (assumes scripts run from repo root location)

# -------------------- utilities --------------------
def run_cmd(cmd, env=None, check=True):
    print("RUN:", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    print("-> rc:", p.returncode)
    if p.stdout:
        print("--- STDOUT ---")
        print(p.stdout)
    if p.stderr:
        print("--- STDERR ---")
        print(p.stderr)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)} (rc={p.returncode})")
    return p.returncode, p.stdout, p.stderr

def norm_label_key(s: str) -> str:
    """Normalize label for fuzzy matching: lowercase, replace non-alnum with space, collapse."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("_", " ")
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def find_label_map_file(extracted_dir: Path) -> Path:
    # priority: <extracted_dir>/label_map.json, then repo-root label_map.json
    p1 = extracted_dir / "label_map.json"
    p2 = REPO_ROOT / "label_map.json"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None

def read_label_map(pm: Path) -> Dict[str, str]:
    data = {}
    try:
        data = json.loads(pm.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read label_map.json at {pm}: {e}")
    # ensure values are strings
    return {str(k): str(v) for k, v in data.items()}

def write_canonical_label_map(mapping: Dict[str, str], repo_root: Path):
    """
    Writes canonical label_map.json in repo root in format:
      { "Speaker 1": "extracted_voices/speaker_1.wav", ... }
    Paths are relative to repo_root when possible.
    """
    canonical = {}
    for k, v in mapping.items():
        p = Path(v)
        try:
            rel = str(Path(v).resolve().relative_to(repo_root.resolve()))
        except Exception:
            rel = str(v)
        canonical[k] = rel
    outp = repo_root / "label_map.json"
    outp.write_text(json.dumps(canonical, indent=2), encoding="utf-8")
    print("Wrote canonical label_map.json ->", outp)
    return outp

def parse_srt_labels(srt_path: Path) -> List[str]:
    """
    Extract set/list of speaker labels used in SRT (ordered by first appearance).
    """
    text = srt_path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"\n\s*\n", text.strip())
    labels_ordered = []
    seen = set()
    for p in parts:
        lines = [l.rstrip() for l in p.splitlines() if l.strip()]
        if not lines:
            continue
        time_line = next((l for l in lines if "-->" in l), None)
        label = None
        if time_line and "|" in time_line:
            parts_time = [s.strip() for s in time_line.split("|")]
            if len(parts_time) >= 2:
                label = parts_time[1]
        if label is None:
            text_line_candidates = [l for l in lines if "-->" not in l]
            if text_line_candidates:
                first_line = text_line_candidates[0]
                m = re.match(r'^\[\s*(?P<label>[^\]]+?)\s*\]\s*(?P<rest>.*)', first_line)
                if m:
                    label = m.group('label').strip()
                else:
                    m2 = re.match(r'^(?P<label>(?:speaker|spkr|spk)\s*[\-_\d\w]+)\s*[:\-\|]\s*(?P<rest>.*)$', first_line, flags=re.I)
                    if m2:
                        label = m2.group('label').strip()
                    else:
                        m3 = re.match(r'^(?P<label_candidate>[^:\-\|]{1,40})\s*[:\-\|]\s*(?P<rest>.*)$', first_line)
                        if m3:
                            cand = m3.group('label_candidate').strip()
                            if len(cand) <= 40 and re.search(r'[\.\,\?\!;\/\\\(\)\[\]\{\}]', cand) is None:
                                words = [w for w in re.split(r'\s+', cand) if w]
                                if 1 <= len(words) <= 4:
                                    label = cand
        if label:
            if label not in seen:
                labels_ordered.append(label)
                seen.add(label)
    return labels_ordered

def reconcile_labels(label_map: Dict[str, str], srt_labels: List[str], extracted_dir: Path) -> Dict[str, str]:
    """
    Return a mapping where keys are SRT labels (in original SRT form) and values are paths to wav files (relative).
    """
    if not label_map:
        raise RuntimeError("Empty label_map provided to reconcile_labels()")

    label_map_norm = {}
    for k, v in label_map.items():
        label_map_norm[norm_label_key(k)] = v

    refs = list(label_map.values())
    refs_used = set()
    result = {}

    for s_label in srt_labels:
        n = norm_label_key(s_label)
        matched = None
        if n in label_map_norm:
            matched = label_map_norm[n]
        else:
            m = re.search(r"(\d+)$", n)
            if m:
                num = m.group(1)
                candidate = extracted_dir / f"speaker_{num}.wav"
                if candidate.exists():
                    matched = str(candidate)
            if not matched:
                for k_norm, v in label_map_norm.items():
                    if n and (n in k_norm or k_norm in n):
                        matched = v
                        break
        if matched:
            result[s_label] = matched
            refs_used.add(str(matched))
        else:
            result[s_label] = None

    remaining_refs = [r for r in refs if str(r) not in refs_used]
    remaining_iter = iter(remaining_refs)
    for s_label, v in list(result.items()):
        if v is None:
            try:
                picked = next(remaining_iter)
                result[s_label] = picked
                refs_used.add(str(picked))
                print(f"[reconcile] Assigned {picked} -> missing SRT label '{s_label}'")
            except StopIteration:
                result[s_label] = None
                print(f"[reconcile] WARNING: no remaining extracted WAV to assign for SRT label '{s_label}'")

    final = {}
    for k, v in result.items():
        if v is None:
            final[k] = None
        else:
            p = Path(v)
            try:
                rel = str(p.resolve().relative_to(REPO_ROOT.resolve()))
            except Exception:
                rel = str(v)
            final[k] = rel

    return final

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Run full pipeline: extract -> write label_map.json -> clone_from_srt")
    ap.add_argument("--input", "-i", required=True, help="Input audio file (wav/mp3/video audio).")
    ap.add_argument("--srt", "-s", required=True, help="SRT file (translated or original) to use for TTS timing/text.")
    ap.add_argument("--extracted-outdir", default="extracted_voices", help="Where automate_extract_and_split writes speaker_N.wav")
    ap.add_argument("--demucs-tmp", default="/tmp/demucs_out", help="Temp dir for demucs output.")
    ap.add_argument("--device", default="cuda", help="Device passed to demucs (and optionally other tools).")
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--tts-workers", type=int, default=4)
    ap.add_argument("--gpu-workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="Final output wav path. Default -> outputs_v2/<input_stem>_cloned_final.wav")
    ap.add_argument("--skip-extract", action="store_true", help="Skip extraction/diarization step (if you already ran it).")
    ap.add_argument("--skip-clone", action="store_true", help="Skip clone step (test extraction only).")
    ap.add_argument("--no-clean", action="store_true", help="Forward --no-clean to clone_from_srt.py to keep tmp_clone/my_voices/label_map.json.")
    ap.add_argument("--backup-label-map", action="store_true", help="Create label_map.backup.json before running clone_from_srt.py.")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print("Input not found:", inp)
        sys.exit(1)
    srtp = Path(args.srt)
    if not srtp.exists():
        print("SRT file not found:", srtp)
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN".lower()) or os.environ.get("HF_TOKEN".upper())
    if not hf_token:
        print("Warning: HF_TOKEN not set in environment. If pyannote or translation uses HF auth this will fail.")

    extract_dir = Path(args.extracted_outdir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    out_default = Path("outputs_v2") / f"{inp.stem}_cloned_final.wav"
    final_out = Path(args.out) if args.out else out_default
    final_out.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: extraction + diarization (generates extracted_dir and label_map.json)
    if not args.skip_extract:
        print(f"\n=== STEP 1/3: Running extraction + diarization -> {extract_dir} ===")
        cmd = [
            sys.executable,
            str(THIS_DIR / "automate_extract_and_split.py"),
            str(inp),
            "--out-dir", str(extract_dir),
            "--tmp", str(args.demucs_tmp),
            "--device", str(args.device),
        ]
        env = os.environ.copy()
        if hf_token:
            env["HF_TOKEN"] = hf_token
        try:
            run_cmd(cmd, env=env, check=True)
        except Exception as e:
            print("Extraction/diarization failed:", e)
            sys.exit(1)
    else:
        print("Skipping extraction (as requested). Using existing:", extract_dir)

    # locate label_map.json
    lm_file = find_label_map_file(extract_dir)
    if not lm_file:
        print("No label_map.json found in", extract_dir, "or repo root. Will attempt to build one from extracted WAV filenames.")
        wavs = sorted([str(p) for p in extract_dir.glob("*.wav")])
        fallback = {}
        for i, p in enumerate(wavs, start=1):
            fallback[f"Speaker {i}"] = str(Path(p).resolve())
        write_canonical_label_map(fallback, REPO_ROOT)
        label_map = fallback
    else:
        print("Found label_map.json at:", lm_file)
        try:
            label_map = read_label_map(lm_file)
        except Exception as e:
            print("Failed to read label map:", e)
            sys.exit(1)

    # Step 1.5: reconcile label_map with SRT labels and write canonical label_map.json
    print("\n=== STEP 1.5: Reconciling label_map.json with SRT labels ===")
    srt_labels = parse_srt_labels(srtp)
    if not srt_labels:
        print("Warning: no speaker labels detected in SRT. We'll use mapping keys as-is (or fallback to ordered speaker_N).")
        canonical = {}
        for k, v in label_map.items():
            try:
                rel = str(Path(v).resolve().relative_to(REPO_ROOT.resolve()))
            except Exception:
                rel = str(v)
            canonical[k] = rel
        write_canonical_label_map(canonical, REPO_ROOT)
    else:
        print("Detected SRT labels (in order):", srt_labels)
        reconciled = reconcile_labels(label_map, srt_labels, extract_dir)
        missing = [k for k, v in reconciled.items() if v is None]
        if missing:
            print("WARNING: Could not assign WAVs for these SRT labels:", missing)
        final_map = {}
        for k, v in reconciled.items():
            if v is None:
                final_map[k] = None
            else:
                try:
                    rel = str(Path(v).resolve().relative_to(REPO_ROOT.resolve()))
                except Exception:
                    rel = str(v)
                final_map[k] = rel
        final_map_nonnull = {k: p for k, p in final_map.items() if p is not None}
        write_canonical_label_map(final_map_nonnull, REPO_ROOT)

    # Optionally backup label_map before clone (useful because clone_from_srt may delete it)
    backup_path = REPO_ROOT / "label_map.backup.json"
    if args.backup_label_map:
        src = REPO_ROOT / "label_map.json"
        if src.exists():
            shutil.copy2(src, backup_path)
            print("Backed up label_map.json ->", backup_path)
        else:
            print("No canonical label_map.json to back up at", src)

    # Step 2: run clone_from_srt.py using extracted_dir
    if not args.skip_clone:
        print(f"\n=== STEP 2/3: Running clone_from_srt.py -> {final_out} ===")
        cmd2 = [
            sys.executable,
            str(THIS_DIR / "clone_from_srt.py"),
            "--srt", str(srtp),
            "--extracted_dir", str(extract_dir),
            "--out", str(final_out),
            "--lang", str(args.lang),
            "--tau", str(args.tau),
            "--tts-workers", str(args.tts_workers),
            "--gpu-workers", str(args.gpu_workers),
        ]
        if args.device:
            cmd2 += ["--device", args.device]
        if args.no_clean:
            cmd2 += ["--no-clean"]
            print("[note] forwarding --no-clean to clone_from_srt.py (will keep tmp_clone/my_voices/label_map.json).")
        env2 = os.environ.copy()
        if hf_token:
            env2["HF_TOKEN"] = hf_token
        try:
            run_cmd(cmd2, env=env2, check=True)
        except Exception as e:
            print("clone_from_srt.py failed:", e)
            sys.exit(1)
    else:
        print("Skipping clone step (as requested). Final output was not generated.")

    # Final messages: label_map presence / backup info
    repo_map = REPO_ROOT / "label_map.json"
    if args.backup_label_map and backup_path.exists():
        print("\nPipeline finished. Final output:", final_out.resolve())
        print("A backup of the canonical label map was saved at:", backup_path.resolve())
    else:
        if repo_map.exists():
            print("\nPipeline finished. Final output:", final_out.resolve())
            print("Canonical label_map.json remains at:", repo_map.resolve())
        else:
            print("\nPipeline finished. Final output:", final_out.resolve())
            print("Note: canonical label_map.json was not found after the clone step (clone_from_srt may have removed it).")
            if (REPO_ROOT / "label_map.backup.json").exists():
                print("A backup exists at:", (REPO_ROOT / "label_map.backup.json").resolve())

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
automate_extract_and_split.py

Runs Demucs (GPU) to extract vocals, then runs pyannote speaker diarization,
and exports ordered per-speaker WAVs to `my_voices/` as speaker_1.wav, speaker_2.wav, ...

Usage:
    export HF_TOKEN=hf_xxx
    python automate_extract_and_split.py <input_audio_path> [--out-dir my_voices] [--tmp /tmp/demucs_out]

Notes:
 - Requires demucs installed/available in path or importable via `python -m demucs`.
 - Requires pyannote.audio installed and HF_TOKEN set in environment.
"""
import sys
import subprocess
from pathlib import Path
import shutil
import os
import argparse
from typing import Tuple

# audio processing imports
from pydub import AudioSegment

# pyannote imports (import when needed)
from huggingface_hub import login
from pyannote.audio import Pipeline

def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Return code:", p.returncode)
        if p.stdout:
            print("--- STDOUT ---")
            print(p.stdout)
        if p.stderr:
            print("--- STDERR ---")
            print(p.stderr)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError as e:
        print("FileNotFoundError:", e)
        return None, "", str(e)


def demucs_separate(input_path: Path, out_dir: Path, device: str = "cuda", use_module_if_missing=True) -> Tuple[bool,str]:
    """
    Run demucs to extract vocals. Returns (ok, path_or_error).
    """
    base_args = [
        "demucs",
        "--two-stems=vocals",
        "--name", "htdemucs",
        "--device", device,
        "--out", str(out_dir),
        str(input_path)
    ]
    rc, out, err = run_cmd(base_args)
    if rc is None and use_module_if_missing:
        # try python -m demucs (use same interpreter)
        print("demucs CLI not found, trying python -m demucs using current interpreter.")
        py_cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "--name", "htdemucs",
            "--device", device,
            "--out", str(out_dir),
            str(input_path)
        ]
        rc2, out2, err2 = run_cmd(py_cmd)
        if rc2 != 0:
            return False, f"'python -m demucs' failed. stdout:\n{out2}\nstderr:\n{err2}"
    elif rc != 0:
        return False, f"'demucs' returned code {rc}. stdout:\n{out}\nstderr:\n{err}"

    # search for vocals file
    matches = list(Path(out_dir).rglob("vocals.wav")) + list(Path(out_dir).rglob("*_vocals.wav"))
    if matches:
        # prefer file coming from the input stem (best-effort)
        if len(matches) > 1:
            # try find the one closest to input filename
            stem = input_path.stem
            for m in matches:
                if stem in m.name:
                    return True, str(m)
        return True, str(matches[0])
    return False, "Demucs ran but no vocals.wav found in output"


def diarize_and_split(vocals_path: Path, output_dir: Path, hf_token: str):
    """
    Run pyannote diarization on vocals_path and export ordered speaker wavs to output_dir.
    """
    # login
    login(token=hf_token)

    print("Loading pyannote diarization pipeline (this may take a while)...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

    print("Running diarization...")
    diarization = pipeline({"uri": "audio", "audio": str(vocals_path)})

    # load audio
    audio = AudioSegment.from_file(str(vocals_path))

    speaker_segments = {}
    first_appearance = {}

    # diarization.itertracks yields (segment, track, label) depending on pyannote version
    # But your earlier code used: for turn, _, speaker in diarization.itertracks(yield_label=True):
    try:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_s = float(turn.start)
            end_s = float(turn.end)
            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            seg_audio = audio[start_ms:end_ms]

            if speaker not in speaker_segments:
                speaker_segments[speaker] = AudioSegment.silent(duration=0)
                first_appearance[speaker] = start_s
            else:
                if start_s < first_appearance[speaker]:
                    first_appearance[speaker] = start_s

            speaker_segments[speaker] += seg_audio + AudioSegment.silent(duration=200)
    except TypeError:
        # older/newer pyannote API may iterate differently; try fallback
        # diarization.labels() or diarization.itersegments / items
        print("Warning: diarization.itertracks(yield_label=True) failed — trying fallback iteration.")
        # try items() which yields (segment, label)
        for segment, label in diarization.items():
            start_s = float(segment.start)
            end_s = float(segment.end)
            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            seg_audio = audio[start_ms:end_ms]
            speaker = label
            if speaker not in speaker_segments:
                speaker_segments[speaker] = AudioSegment.silent(duration=0)
                first_appearance[speaker] = start_s
            else:
                if start_s < first_appearance[speaker]:
                    first_appearance[speaker] = start_s
            speaker_segments[speaker] += seg_audio + AudioSegment.silent(duration=200)

    if not speaker_segments:
        raise RuntimeError("No speakers found by diarization.")

    # order
    ordered = sorted(first_appearance.items(), key=lambda kv: kv[1])
    mapping = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (spk_label, _) in enumerate(ordered, start=1):
        out_name = f"speaker_{idx}.wav"
        out_path = output_dir / out_name
        segs = speaker_segments[spk_label]
        segs.export(out_path, format="wav")
        mapping[spk_label] = out_name
        print(f"Saved {out_path}")

    print("\nSpeaker mapping (original_label -> ordered_filename):")
    for orig_label, out_file in mapping.items():
        print(f"  {orig_label} -> {out_file}")

    return mapping


def main():
    parser = argparse.ArgumentParser(description="Extract vocals (demucs) + diarize into ordered speaker wavs.")
    parser.add_argument("input", help="Input audio path (wav/mp3/etc).")
    parser.add_argument("--out-dir", default="my_voices", help="Directory to write speaker_N.wav files.")
    parser.add_argument("--tmp", default="/tmp/demucs_out", help="Temporary demucs output directory.")
    parser.add_argument("--device", default="cuda", help="Device to pass to demucs (cuda or cpu).")
    args = parser.parse_args()

    inp = Path(args.input).resolve()
    if not inp.exists():
        print("Input file not found:", inp)
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN".lower())
    if not hf_token:
        print("Set your HF_TOKEN environment variable first: export HF_TOKEN=hf_xxx")
        sys.exit(1)

    tmp_out = Path(args.tmp)
    # clean tmp
    try:
        if tmp_out.exists():
            shutil.rmtree(tmp_out)
    except Exception as e:
        print("Warning: couldn't remove tmp dir:", e)
    tmp_out.mkdir(parents=True, exist_ok=True)

    print("Running Demucs to extract vocals (device=%s)..." % args.device)
    ok, info = demucs_separate(inp, tmp_out, device=args.device)
    if not ok:
        print("Demucs failed:", info)
        sys.exit(1)

    vocals_path = Path(info)
    print("Vocals found at:", vocals_path)

    # copy vocals to cwd-named file to follow your previous convention
    final_vocals = Path.cwd() / f"{inp.stem}_vocals.wav"
    try:
        shutil.copy2(vocals_path, final_vocals)
        print("Copied vocals to:", final_vocals)
    except Exception as e:
        print("Warning: could not copy vocals file:", e)
        final_vocals = vocals_path

    try:
        print("Running diarization and splitting speakers...")
        mapping = diarize_and_split(final_vocals, Path(args.out_dir), hf_token=hf_token)
    except Exception as e:
        print("Diarization failed:", str(e))
        sys.exit(1)
    finally:
        # cleanup temporary demucs output if it exists
        try:
            if tmp_out.exists():
                shutil.rmtree(tmp_out)
        except Exception:
            pass

    print("Done. Output directory:", args.out_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()

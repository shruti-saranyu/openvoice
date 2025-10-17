#!/usr/bin/env python3
"""
map_srt_to_speaker_segments.py

Given:
 - an SRT file where each subtitle line contains speaker metadata (e.g. " | Speaker 1 | ...")
 - a directory with per-speaker folders like speaker_1/, speaker_2/ containing
   segment files named like: 01_002_0005000_0008123.wav  (speakerIdx_segIdx_startms_endms.wav)

This script builds a JSON mapping of each SRT entry -> list of matching segment files
from the corresponding speaker folder (based on overlap of timestamps).

Usage:
    python map_srt_to_speaker_segments.py --srt subs.srt --voices-dir my_voices --out map.json [--stitch]

Options:
    --overlap-ms N   Minimum overlap in milliseconds to consider a segment matching (default 1).
    --stitch         Create per-subtitle WAV files by concatenating (and trimming) matched segments.
                     Requires pydub + ffmpeg.
"""
import argparse
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict
from collections import namedtuple

# Optional audio stitching
try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

Subtitle = namedtuple("Subtitle", ["index", "start_ms", "end_ms", "speaker_label", "text"])

SEG_FILENAME_RE = re.compile(r"(?P<spkidx>\d+)[_-](?P<segidx>\d+)[_-](?P<start_ms>\d+)[_-](?P<end_ms>\d+)\.wav$", re.IGNORECASE)
# fallback: allow names like speaker_1/01_001_0000000_0002543.wav or 01_001_0000000_0002543.wav

SRT_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})")

def parse_time_to_ms(hh: str, mm: str, ss: str, ms: str) -> int:
    return (int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1000 + int(ms)

def parse_srt(srt_path: Path) -> List[Subtitle]:
    content = srt_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", content.strip())
    subtitles = []
    for blk in blocks:
        lines = [l.strip() for l in blk.strip().splitlines() if l.strip()]
        if not lines:
            continue
        # first line may be index
        idx = None
        try:
            if re.fullmatch(r"\d+", lines[0]):
                idx = int(lines[0])
                times_line = lines[1]
                rest = lines[2:]
            else:
                # no numeric index — assume first line is times
                times_line = lines[0]
                rest = lines[1:]
        except IndexError:
            continue

        m = SRT_TIME_RE.search(times_line)
        if not m:
            # skip if no time found
            continue
        start_ms = parse_time_to_ms(m.group(1), m.group(2), m.group(3), m.group(4))
        end_ms   = parse_time_to_ms(m.group(5), m.group(6), m.group(7), m.group(8))

        # Many of your SRT lines have metadata after times in same or next line: e.g.
        # "00:00:00,000 --> 00:00:07,000 | Speaker 1 | Female | Neutral"
        # so extract speaker label from the times_line if present
        speaker_label = None
        # find pipe-separated parts
        if "|" in times_line:
            parts = [p.strip() for p in times_line.split("|")]
            # last parts include times and speaker. times_line usually begins with times.
            # find any part that matches 'Speaker' (case-insensitive)
            for p in parts:
                if re.search(r"(?i)\bspeaker\s*\d+\b", p):
                    speaker_label = p
                    break
        # if not found in times_line, check rest first line of text for pipe metadata
        if not speaker_label and rest:
            maybe = rest[0]
            if "|" in maybe:
                parts = [p.strip() for p in maybe.split("|")]
                for p in parts:
                    if re.search(r"(?i)\bspeaker\s*\d+\b", p):
                        speaker_label = p
                        # remove the metadata line from text
                        rest = rest[1:]
                        break

        # fallback: extract first "Speaker X" found anywhere in block
        if not speaker_label:
            alltxt = " ".join(lines)
            spm = re.search(r"(?i)(Speaker\s*\d+)", alltxt)
            if spm:
                speaker_label = spm.group(1)

        # normalize speaker label: e.g., "Speaker 1" -> "speaker_1"
        if speaker_label:
            m2 = re.search(r"(?i)speaker\s*(\d+)", speaker_label)
            if m2:
                spk_norm = f"speaker_{int(m2.group(1))}"
            else:
                spk_norm = speaker_label.strip().lower().replace(" ", "_")
        else:
            spk_norm = "unknown"

        text = "\n".join(rest).strip()
        subtitles.append(Subtitle(index=idx or -1, start_ms=start_ms, end_ms=end_ms, speaker_label=spk_norm, text=text))
    return subtitles

def scan_segments(voices_dir: Path) -> Dict[str, List[Tuple[Path, int, int]]]:
    """
    Return { speaker_dir_name : [ (path, start_ms, end_ms), ... ] }
    """
    voices_dir = voices_dir.expanduser().resolve()
    if not voices_dir.exists():
        raise FileNotFoundError(f"voices_dir not found: {voices_dir}")

    result = {}
    # List directories like speaker_1, speaker_2; also accept files directly under voices_dir
    for sp_dir in voices_dir.iterdir():
        if sp_dir.is_dir() and re.match(r"(?i)speaker[_\s-]?\d+$", sp_dir.name):
            segs = []
            for f in sp_dir.glob("*.wav"):
                m = SEG_FILENAME_RE.search(f.name)
                if m:
                    start_ms = int(m.group("start_ms"))
                    end_ms   = int(m.group("end_ms"))
                    segs.append((f, start_ms, end_ms))
                else:
                    # try to parse numbers anywhere in name (fallback)
                    nums = re.findall(r"(\d{6,})", f.name)  # large numbers likely ms
                    if len(nums) >= 2:
                        start_ms = int(nums[-2])
                        end_ms   = int(nums[-1])
                        segs.append((f, start_ms, end_ms))
                    else:
                        # cannot parse timestamps from filename; skip
                        print(f"Skipping (no timestamp parse): {f}")
            # sort by start_ms
            segs_sorted = sorted(segs, key=lambda t: t[1])
            result[sp_dir.name] = segs_sorted

    # Also handle case where segments are directly in voices_dir and filenames include speaker index prefix:
    # e.g. 01_001_0000000_0002543.wav
    direct_segs = []
    for f in voices_dir.glob("*.wav"):
        m = SEG_FILENAME_RE.search(f.name)
        if m:
            spkidx = int(m.group("spkidx"))
            sp_dir_name = f"speaker_{spkidx}"
            start_ms = int(m.group("start_ms"))
            end_ms   = int(m.group("end_ms"))
            result.setdefault(sp_dir_name, []).append((f, start_ms, end_ms))
    # sort sublists
    for k in result:
        result[k] = sorted(result[k], key=lambda t: t[1])
    return result

def overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    return max(0, e - s)

def build_mapping(subtitles: List[Subtitle], segments_by_speaker: Dict[str, List[Tuple[Path, int, int]]], min_overlap_ms: int = 1):
    mapping = {}
    for sub in subtitles:
        spk = sub.speaker_label
        segs = segments_by_speaker.get(spk, [])
        matched = []
        for (p, s_ms, e_ms) in segs:
            ov = overlap_ms(sub.start_ms, sub.end_ms, s_ms, e_ms)
            if ov >= min_overlap_ms:
                matched.append({
                    "segment_path": str(p),
                    "segment_start_ms": s_ms,
                    "segment_end_ms": e_ms,
                    "overlap_ms": ov
                })
        mapping_key = f"{sub.index or -1}_{sub.start_ms}_{sub.end_ms}"
        mapping[mapping_key] = {
            "srt_index": sub.index,
            "srt_start_ms": sub.start_ms,
            "srt_end_ms": sub.end_ms,
            "speaker_dir": spk,
            "text": sub.text,
            "matched_segments": matched
        }
    return mapping

def stitch_for_subtitle(sub: Subtitle, matched_segments: List[Dict], out_dir: Path):
    """
    Concatenate matched segment files (in chronological order) and trim so final
    audio exactly matches subtitle start/end. Output file: subtitle_{index}.wav
    """
    if AudioSegment is None:
        raise RuntimeError("pydub not available. Install pydub and ffmpeg to stitch audio.")
    # create a combined track by pasting segments in order
    pieces = []
    for m in sorted(matched_segments, key=lambda x: x["segment_start_ms"]):
        seg = AudioSegment.from_file(m["segment_path"])
        pieces.append((m["segment_start_ms"], seg))
    if not pieces:
        return None
    # create silent timeline covering from first segment start to last segment end
    first_start = pieces[0][0]
    last_end = max(m["segment_end_ms"] for m in matched_segments)
    duration_ms = last_end - first_start
    timeline = AudioSegment.silent(duration=duration_ms+100)  # small cushion
    # paste each segment at (segment_start - first_start) offset
    for seg_start, seg_audio in pieces:
        offset = seg_start - first_start
        timeline = timeline.overlay(seg_audio, position=offset)
    # now crop the timeline to match subtitle exact window
    crop_start = sub.start_ms - first_start
    crop_end   = sub.end_ms - first_start
    crop_start = max(0, crop_start)
    crop_end = min(len(timeline), crop_end)
    final = timeline[crop_start:crop_end]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"subtitle_{sub.index:03d}_{sub.start_ms}_{sub.end_ms}.wav"
    final.export(out_path, format="wav")
    return out_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--srt", required=True, type=Path, help="Path to SRT file")
    p.add_argument("--voices-dir", required=True, type=Path, help="Directory containing speaker_N/ folders")
    p.add_argument("--out", default=Path("srt_segment_map.json"), type=Path, help="Output JSON mapping path")
    p.add_argument("--min-overlap-ms", type=int, default=1, help="Minimum overlap (ms) to consider a segment matching")
    p.add_argument("--stitch", action="store_true", help="Create per-subtitle WAV by stitching & trimming matched segments (needs pydub+ffmpeg)")
    p.add_argument("--stitched-out", type=Path, default=Path("stitched_subtitles"), help="Directory for stitched subtitle wavs")
    args = p.parse_args()

    subtitles = parse_srt(args.srt)
    print(f"Parsed {len(subtitles)} subtitles from {args.srt}")

    segments_by_speaker = scan_segments(args.voices_dir)
    print(f"Found {sum(len(v) for v in segments_by_speaker.values())} segments across {len(segments_by_speaker)} speaker dirs")

    mapping = build_mapping(subtitles, segments_by_speaker, min_overlap_ms=args.min_overlap_ms)

    # write json
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    print(f"Wrote mapping to {args.out}")

    # print summary
    unmatched = 0
    for k, v in mapping.items():
        if not v["matched_segments"]:
            unmatched += 1
    print(f"Subtitles with no matched segments: {unmatched}/{len(mapping)}")

    if args.stitch:
        if AudioSegment is None:
            print("pydub not available — cannot stitch. Install pydub and ensure ffmpeg is on PATH.")
            return
        stitched_out = args.stitched_out
        for key, entry in mapping.items():
            sub_index = entry["srt_index"]
            # reconstruct Subtitle object (quick)
            sub = Subtitle(index=sub_index, start_ms=entry["srt_start_ms"], end_ms=entry["srt_end_ms"], speaker_label=entry["speaker_dir"], text=entry["text"])
            if not entry["matched_segments"]:
                continue
            out_path = stitch_for_subtitle(sub, entry["matched_segments"], stitched_out)
            if out_path:
                print("Created stitched:", out_path)

if __name__ == "__main__":
    main()

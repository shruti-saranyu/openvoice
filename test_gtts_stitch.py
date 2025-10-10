#!/usr/bin/env python3
# test_gtts_stitch.py
# Requires: pip install gTTS pydub

from gtts import gTTS
from pydub import AudioSegment
from pathlib import Path
import tempfile
import argparse
import re

def parse_srt_like_with_meta(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    entries=[]
    i=0
    while i < len(lines):
        if not lines[i].strip():
            i+=1; continue
        if re.fullmatch(r'\d+', lines[i].strip()):
            i+=1
            if i>=len(lines): break
        time_and_meta = lines[i].strip(); i+=1
        text_lines=[]
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i]); i+=1
        text = " ".join(text_lines).strip()
        if '|' in time_and_meta:
            time_part, meta_part = time_and_meta.split('|',1)
            meta_parts=[p.strip() for p in meta_part.split('|')]
        else:
            time_part=time_and_meta; meta_parts=[]
        m = re.search(r'([\d:,\.]+)\s*-->\s*([\d:,\.]+)', time_part)
        if not m: continue
        def to_ms(s):
            s=s.replace(',', '.'); parts=s.split(':'); parts=[float(p) for p in parts]
            if len(parts)==3:
                h,m,s=parts
            else:
                h=0; m,s=parts
            return int((h*3600 + m*60 + s)*1000)
        start_ms = to_ms(m.group(1)); end_ms=to_ms(m.group(2))
        speaker = meta_parts[0] if meta_parts else "Speaker"
        entries.append({'start_ms':start_ms, 'end_ms':end_ms, 'speaker':speaker, 'text':text})
    return entries

def main():
    p = Path('kan2')  # your SRT
    if not p.exists():
        if (p.with_suffix('.srt')).exists(): p=p.with_suffix('.srt')
        else:
            print("kan2 not found"); return
    entries = parse_srt_like_with_meta(p)
    if not entries:
        print("No entries parsed"); return
    tmpdir = Path(tempfile.mkdtemp(prefix='gtts_test_'))
    print("tmpdir:", tmpdir)
    seg_files=[]
    for idx,e in enumerate(entries):
        txt = e['text'][:500]
        tts = gTTS(txt, lang='en')
        seg = tmpdir / f"seg_{idx:04d}.mp3"
        tts.save(str(seg))
        wav_out = tmpdir / f"seg_{idx:04d}.wav"
        audio = AudioSegment.from_file(seg)
        # convert to 16k mono
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(wav_out, format='wav')
        seg_files.append({'start_ms': e['start_ms'], 'end_ms': e['end_ms'], 'file': str(wav_out)})
    final_len = max(s['end_ms'] for s in seg_files)
    base = AudioSegment.silent(duration=final_len)
    for s in seg_files:
        seg_audio = AudioSegment.from_file(s['file'])
        expected = s['end_ms'] - s['start_ms']
        # pad or trim to expected
        if len(seg_audio) < expected:
            seg_audio = seg_audio + AudioSegment.silent(duration=(expected - len(seg_audio)))
        else:
            seg_audio = seg_audio[:expected]
        base = base.overlay(seg_audio, position=s['start_ms'])
    out = Path('output_test_gtts.wav')
    base.export(out, format='wav')
    print("Wrote", out, "tmpdir left at", tmpdir)

if __name__ == '__main__':
    main()

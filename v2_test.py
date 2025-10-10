#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2_test.py — Patched version with --tau support

This script converts one audio file from a source speaker embedding
to a target speaker embedding using the OpenVoice ToneColorConverter.

Usage:
    python v2_test.py \
        --source tmp_clone/src_0_pad.wav \
        --src_se checkpoints_v2/base_speakers/ses/base.pth \
        --tgt_se checkpoints_v2/base_speakers/ses/speaker_1.pth \
        --out tmp_clone/seg_0_out.wav \
        --tau 0.4
"""

import argparse
import torch
import os
from openvoice.api import ToneColorConverter


def main():
    parser = argparse.ArgumentParser(description="OpenVoice v2 voice conversion test script")
    parser.add_argument("--source", required=True, help="Path to the source .wav file")
    parser.add_argument("--src_se", required=False, default=None, help="Path to source embedding (.pth)")
    parser.add_argument("--tgt_se", required=True, help="Path to target embedding (.pth)")
    parser.add_argument("--out", required=True, help="Output .wav file path")
    parser.add_argument("--tau", type=float, default=None, help="Optional tau (conversion strength)")
    parser.add_argument("--device", default=None, help="Device to use (cuda:0 or cpu)")
    args = parser.parse_args()

    # Determine device
    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load converter
    converter_cfg = "checkpoints_v2/converter/config.json"
    converter_ckpt = "checkpoints_v2/converter/checkpoint.pth"

    converter = ToneColorConverter(converter_cfg, device=device)
    converter.load_ckpt(converter_ckpt)
    print(f"Loaded converter checkpoint: {converter_ckpt}")

    # Load embeddings
    src_se = None
    if args.src_se and os.path.exists(args.src_se):
        src_se = torch.load(args.src_se, map_location=device)
        print(f"Loaded src_se: {args.src_se}")
    tgt_se = torch.load(args.tgt_se, map_location=device)
    print(f"Loaded tgt_se: {args.tgt_se}")

    # Perform conversion
    print(f"Converting: {args.source} -> {args.out} (tau={args.tau})")
    converter.convert(
        audio_src_path=args.source,
        src_se=src_se,
        tgt_se=tgt_se,
        output_path=args.out,
        tau=args.tau
    )

    print(f"✅ Conversion complete. Saved to: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
app.py — FastAPI wrapper for OpenVoice full pipeline (run_full_pipeline.py)

Features:
 - Uses config.json for pipeline defaults
 - Reads Hugging Face token automatically from HUGGINGFACE_TOKEN.txt
 - Accepts only:
     - input_audio (file)
     - srt (file)
     - language (optional)
"""

import os
import shutil
import tempfile
import subprocess
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Paths
THIS_DIR = Path(__file__).resolve().parent
RUN_WRAPPER = THIS_DIR / "run_full_pipeline.py"
CONFIG_PATH = THIS_DIR / "config.json"
TOKEN_PATH = THIS_DIR / "HUGGINGFACE_TOKEN.txt"
EXTRACTED_DIR_DEFAULT = THIS_DIR / "extracted_voices"
OUTPUTS_DIR = THIS_DIR / "outputs_v2"

app = FastAPI(title="OpenVoice_v2 Pipeline API", version="1.0")


# ---------------------- Config loading ----------------------
def _load_config():
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing config file: {CONFIG_PATH}")
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Error reading config.json: {e}")

    cfg.setdefault("tau", 0.4)
    cfg.setdefault("tts_workers", 4)
    cfg.setdefault("gpu_workers", 1)
    cfg.setdefault("no_clean", False)
    cfg.setdefault("backup_label_map", False)
    cfg.setdefault("skip_extract", False)
    cfg.setdefault("skip_clone", False)
    cfg.setdefault("device", "cuda")
    cfg.setdefault("extracted_outdir", str(EXTRACTED_DIR_DEFAULT))
    cfg.setdefault("lang", "en")
    return cfg


def _load_hf_token():
    """Read Hugging Face token from HUGGINGFACE_TOKEN.txt"""
    if TOKEN_PATH.exists():
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
        if token:
            return token
    print("⚠️  Warning: HUGGINGFACE_TOKEN.txt missing or empty — diarization may fail.")
    return ""


CONFIG = _load_config()
HF_TOKEN = _load_hf_token()


def _save_upload_to_path_bytes(content: bytes, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest


# ---------------------- API endpoint ----------------------
@app.post("/run_pipeline")
async def run_pipeline(
    input_audio: UploadFile = File(..., description="Input audio file (wav/mp3/video)"),
    srt: UploadFile = File(..., description="SRT file with timings and optional speaker labels"),
    language: Optional[str] = Form(None),
):
    """Run the full OpenVoice pipeline."""
    if not RUN_WRAPPER.exists():
        raise HTTPException(status_code=500, detail=f"Missing runner: {RUN_WRAPPER}")

    cfg = dict(CONFIG)
    if language:
        cfg["lang"] = language

    run_tmpdir = Path(tempfile.mkdtemp(prefix="openvoice_run_", dir=str(THIS_DIR)))
    try:
        # Save uploaded files
        audio_ext = Path(input_audio.filename).suffix or ".wav"
        srt_ext = Path(srt.filename).suffix or ".srt"
        audio_path = run_tmpdir / f"input{audio_ext}"
        srt_path = run_tmpdir / f"subs{srt_ext}"

        _save_upload_to_path_bytes(await input_audio.read(), audio_path)
        _save_upload_to_path_bytes(await srt.read(), srt_path)

        # Build command
        cmd = [
            "python",
            str(RUN_WRAPPER),
            "--input", str(audio_path),
            "--srt", str(srt_path),
            "--extracted-outdir", str(cfg.get("extracted_outdir")),
            "--tau", str(cfg.get("tau")),
            "--lang", str(cfg.get("lang")),
            "--tts-workers", str(cfg.get("tts_workers")),
            "--gpu-workers", str(cfg.get("gpu_workers")),
            "--device", str(cfg.get("device")),
        ]
        if cfg.get("no_clean"):
            cmd.append("--no-clean")
        if cfg.get("backup_label_map"):
            cmd.append("--backup-label-map")
        if cfg.get("skip_extract"):
            cmd.append("--skip-extract")
        if cfg.get("skip_clone"):
            cmd.append("--skip-clone")

        # Environment
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if HF_TOKEN:
            env["HF_TOKEN"] = HF_TOKEN

        print(f"\n=== Running pipeline with config: {cfg} ===")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)

        resp = {
            "returncode": proc.returncode,
            "cmd": " ".join(cmd),
            "stdout": proc.stdout[-20000:],  # last 20k chars
            "stderr": proc.stderr[-20000:],
            "final_output": None,
            "label_map_backed_up": False,
        }

        expected = Path(cfg.get("out") or (OUTPUTS_DIR / f"{audio_path.stem}_cloned_final.wav"))
        if proc.returncode == 0 and expected.exists():
            resp["final_output"] = str(expected.resolve())
        else:
            matches = list(OUTPUTS_DIR.glob(f"{audio_path.stem}*"))
            if matches:
                resp["final_output"] = str(matches[0].resolve())

        backup_file = THIS_DIR / "label_map.backup.json"
        if cfg.get("backup_label_map") and backup_file.exists():
            resp["label_map_backed_up"] = True
            resp["label_map_backup_path"] = str(backup_file.resolve())

        if proc.returncode != 0:
            resp["message"] = "Pipeline failed. See stdout/stderr for details."
            return JSONResponse(status_code=500, content=resp)

        resp["message"] = "Pipeline completed successfully."
        return resp

    finally:
        shutil.rmtree(run_tmpdir, ignore_errors=True)


# ---------------------- Health check ----------------------
@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "openvoice-pipeline"}


# ---------------------- Server entrypoint ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8899, reload=True)

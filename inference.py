#!/usr/bin/env python3
# inference.py  (fallback-capable, ToneColorConverter in-process)
"""
synthesize_openvoice wrapper.

Workflow:
1. Try to load ToneColorConverter in-process (best-effort).
2. If that fails, run an external CLI command (configurable) to produce the wav.

Configure fallback CLI:
- Set environment variable OPENVOICE_CLI_TEMPLATE, or
- Edit FALLBACK_CLI_TEMPLATE below.

Template fields:
  {text}    -> text to synthesize (will be shell-escaped)
  {ref}     -> path to reference wav
  {out}     -> desired output wav path
  {device}  -> device string (cpu|cuda)
  {source}  -> optional source wav (if CLI expects it)
"""
import os
import shlex
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import numpy as np
import soundfile as sf
import torchaudio

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger("inference")

# --- Fallback CLI template (edit to match your repo's CLI) ---
FALLBACK_CLI_TEMPLATE = (
    'python v2_test.py --mode full --source "{source}" --ref "{ref}" --out "{out}"'
)

# Converter default target SR fallback
DEFAULT_TARGET_SR = 24000

# ----------------- simple audio helpers -----------------
def _resample_tensor(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """wav: (channels, samples)"""
    if orig_sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=target_sr)

def _load_audio_tensor(path: str, ensure_mono: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Returns (waveform_tensor, sample_rate) with waveform shape (channels, samples) or (1, N).
    """
    if not Path(path).exists():
        raise FileNotFoundError(path)
    wav, sr = torchaudio.load(path)  # shape: (channels, samples)
    if ensure_mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr

def _write_wav(path: str, audio: np.ndarray, sr: int):
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sr, subtype="PCM_16")

# ----------------- ToneColorConverter in-process loader & cache -----------------
_CONVERTER = None  # cached ToneColorConverter instance
_SE_CACHE: Dict[str, torch.Tensor] = {}  # ref_path -> speaker embedding tensor

def _load_tonecolor_converter(ckpt_path: str, device: str = "cpu"):
    """
    Robust loader for ToneColorConverter.

    Accepts:
      - path to a config.json file
      - path to a checkpoint file (checkpoint.pth or *.pt)
      - path to a directory containing config.json and checkpoint.pth

    Returns cached converter instance.
    """
    global _CONVERTER
    if _CONVERTER is not None:
        return _CONVERTER

    # normalize path
    p = Path(ckpt_path) if ckpt_path is not None else None

    # If directory, try to find config.json and checkpoint.pth
    if p and p.is_dir():
        config_file = p / "config.json"
        ckpt_candidates = [p / "checkpoint.pth", p / "tone_color_converter.pt", p / "model.pt"]
        if not any(c.exists() for c in ckpt_candidates):
            for f in sorted(p.glob("*.pth")):
                ckpt_candidates.insert(0, f)
            for f in sorted(p.glob("*.pt")):
                ckpt_candidates.insert(0, f)
        chosen_ckpt = next((str(c) for c in ckpt_candidates if Path(c).exists()), None)
        if config_file.exists():
            chosen_config = str(config_file)
        else:
            chosen_config = chosen_ckpt or str(p)
        to_pass = chosen_config
    else:
        to_pass = str(p) if p else None

    try:
        from openvoice.api import ToneColorConverter  # type: ignore
    except Exception as e:
        _LOG.debug("openvoice.api.ToneColorConverter import failed: %s", e)
        raise

    try:
        converter = ToneColorConverter(to_pass, device=device)
        _CONVERTER = converter
        _LOG.info("Loaded ToneColorConverter from %s (device=%s)", to_pass, device)
        return converter
    except Exception as e:
        _LOG.exception("Failed to instantiate ToneColorConverter with path=%s.", to_pass)
        raise

def _get_target_sr_from_converter(converter) -> int:
    return getattr(converter, "target_sample_rate", DEFAULT_TARGET_SR)

# ----------------- speaker embedding extraction (repo-specific) -----------------
def _get_speaker_embedding_for_ref(converter, ref_path: str, device: str = "cpu") -> torch.Tensor:
    """
    For repos where se_extractor.get_se(audio_path, vc_model, target_dir='processed', vad=True)
    (or similar) is expected. Uses converter.vc_model if available, otherwise passes converter.
    Caches per-ref embeddings in _SE_CACHE.
    """
    global _SE_CACHE
    if ref_path in _SE_CACHE:
        return _SE_CACHE[ref_path]

    try:
        from openvoice import se_extractor  # type: ignore
    except Exception as e:
        _LOG.exception("Failed to import se_extractor: %s", e)
        raise

    # prefer vc_model attribute if present
    vc_model = getattr(converter, "vc_model", None) or getattr(converter, "model", None) or converter

    # Resolve torch.device
    try:
        torch_dev = torch.device(device)
    except Exception:
        torch_dev = torch.device("cpu")

    # Try to move model to device (non-fatal)
    try:
        if hasattr(vc_model, "to") and callable(getattr(vc_model, "to")):
            try:
                vc_model.to(torch_dev)
            except Exception:
                pass
    except Exception:
        pass

    # Ensure vc_model.device exists
    try:
        if not hasattr(vc_model, "device"):
            try:
                setattr(vc_model, "device", torch_dev)
            except Exception:
                try:
                    setattr(vc_model, "device", str(torch_dev))
                except Exception:
                    pass
    except Exception:
        pass

    # Ensure vc_model.version exists (default to "v1" matching common forks)
    try:
        if not hasattr(vc_model, "version"):
            setattr(vc_model, "version", "v1")
    except Exception:
        try:
            setattr(vc_model, "version", "v1")
        except Exception:
            pass

    # --- shim: attach best-effort extract_se if missing ---
    if not hasattr(vc_model, "extract_se"):
        _LOG.warning("vc_model lacks 'extract_se'; attaching best-effort shim to vc_model (%s).", type(vc_model).__name__)

        def _best_effort_extract_se(audio_segs, se_save_path=None, **kwargs):
            """
            audio_segs: could be path or list of paths.
            Try model-provided alternatives; if none, compute a mel-mean fallback embedding.
            Returns embedding (ndarray or tensor) or (embedding, meta).
            """
            # try a set of plausible method names on the model
            alt_names = [
                "extract_speaker_embedding",
                "get_spk_emb",
                "get_spk",
                "get_embedding",
                "extract_embedding",
                "inference_embedding",
                "encode_speaker",
                "speaker_encoder",
                "speaker_embedding",
            ]

            # pick first path-like item if list
            first = None
            try:
                if isinstance(audio_segs, (list, tuple)) and len(audio_segs) > 0:
                    first = audio_segs[0]
                else:
                    first = audio_segs
            except Exception:
                first = audio_segs

            # Attempt model methods
            for name in alt_names:
                fn = getattr(vc_model, name, None)
                if callable(fn):
                    _LOG.info("vc_model shim: using alternative method '%s' for extract_se", name)
                    try:
                        # try same signature as original
                        return fn(audio_segs, se_save_path=se_save_path)
                    except TypeError:
                        try:
                            return fn(first)
                        except Exception:
                            try:
                                return fn(audio_segs)
                            except Exception:
                                pass
                    except Exception as e:
                        _LOG.debug("vc_model alt method '%s' failed: %s", name, e)
                        continue

            # Fallback: compute a naive mel-mean embedding using torchaudio
            try:
                # load audio (first must be path)
                if isinstance(first, str) and Path(first).exists():
                    wav, sr = torchaudio.load(first)  # (channels, samples)
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    wav = wav.squeeze(0)  # (samples,)
                else:
                    # if array-like or tensor
                    if isinstance(first, torch.Tensor):
                        wav = first.detach().cpu()
                        if wav.ndim > 1:
                            wav = wav.mean(dim=0)
                    else:
                        arr = np.asarray(first, dtype="float32")
                        if arr.ndim > 1:
                            arr = arr.mean(axis=0)
                        wav = torch.from_numpy(arr)
                    sr = 22050

                # ensure float tensor
                wav = wav.to(torch.float32)

                # Resample if needed (assume we want 22050)
                target_sr = 22050
                # if torchaudio.load returned sr and it's different, resample
                # we don't have original sr reliably; try catching attribute
                try:
                    if 'sr' in locals() and sr != target_sr:
                        wav = torchaudio.functional.resample(wav.unsqueeze(0), orig_freq=sr, new_freq=target_sr).squeeze(0)
                except Exception:
                    pass

                # compute mel spectrogram (80 mel bins)
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=target_sr, n_fft=1024, win_length=1024, hop_length=256, n_mels=80
                )
                m = mel_transform(wav.unsqueeze(0))  # (1, n_mels, frames)
                emb = m.mean(dim=2).squeeze(0)  # (n_mels,)
                emb_np = emb.cpu().numpy()
                _LOG.warning("vc_model shim: using naive mel-mean fallback embedding (low-quality).")
                return emb_np
            except Exception as e:
                _LOG.exception("vc_model shim fallback failed: %s", e)
                raise RuntimeError("vc_model missing extract_se and no alternative worked.") from e

        # Attach shim to instance if possible; otherwise attach to class
        try:
            setattr(vc_model, "extract_se", _best_effort_extract_se)
        except Exception:
            try:
                setattr(vc_model.__class__, "extract_se", _best_effort_extract_se)
            except Exception:
                _LOG.warning("Could not attach extract_se shim to vc_model object or class; in-process SE extraction may fail.")

    # Create a small temporary directory for any processed outputs get_se might write
    tmpdir = Path(tempfile.mkdtemp(prefix="seproc_"))
    try:
        _LOG.info("Calling se_extractor.get_se on %s with vc_model=%s (device=%s, version=%s) target_dir=%s",
                  ref_path, type(vc_model).__name__, getattr(vc_model, "device", None), getattr(vc_model, "version", None), str(tmpdir))

        # Call the repo's expected signature (many forks: get_se(audio_path, vc_model, target_dir='processed', vad=True))
        ret = se_extractor.get_se(ref_path, vc_model, target_dir=str(tmpdir), vad=False)

        if isinstance(ret, (tuple, list)):
            se = ret[0]
        else:
            se = ret

        # Ensure torch.Tensor if possible
        if not isinstance(se, torch.Tensor):
            try:
                se = torch.tensor(se)
            except Exception:
                pass

        _SE_CACHE[ref_path] = se
        _LOG.info("Cached speaker embedding for %s", ref_path)
        return se
    except Exception as e:
        _LOG.exception("se_extractor.get_se call failed: %s", e)
        raise
    finally:
        # leave tmpdir for debugging; remove if desired
        # import shutil; shutil.rmtree(tmpdir, ignore_errors=True)
        pass

# ----------------- CLI fallback runner -----------------
def run_cli_fallback(text: str, ref: str, out: str, device: str):
    """
    Runs an external script to generate the audio.
    Template fields supported: {text}, {ref}, {out}, {device}, {source} (optional).
    If template expects {source} and none provided, create a short temp WAV and pass it.
    """
    template = os.environ.get("OPENVOICE_CLI_TEMPLATE", FALLBACK_CLI_TEMPLATE)

    mapping = {
        "text": text.replace('"', r'\"'),
        "ref": ref,
        "out": out,
        "device": device,
    }

    tmp_source_path = None
    # If CLI template requires {source} but mapping doesn't have it, create a temp source wav
    if "{source}" in template and "source" not in mapping:
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_source_path = tf.name
        tf.close()
        # create short 1s low-amplitude tone or silent wav
        try:
            sr = 22050
            t = np.linspace(0, 1, int(sr * 1), endpoint=False)
            w = (0.02 * np.sin(2 * np.pi * 220 * t)).astype("float32")
            sf.write(tmp_source_path, w, sr)
        except Exception:
            import wave, struct
            sr = 22050
            n_samples = sr * 1
            with wave.open(tmp_source_path, "wb") as wfh:
                wfh.setnchannels(1)
                wfh.setsampwidth(2)
                wfh.setframerate(sr)
                for _ in range(n_samples):
                    wfh.writeframes(struct.pack("<h", 0))
        mapping["source"] = tmp_source_path

    try:
        cmd_str = template.format(**mapping)
    except KeyError as ke:
        raise RuntimeError(f"CLI template requires unknown placeholder: {ke}. Template: {template}")

    _LOG.info("Attempting fallback CLI: %s", cmd_str)
    args = shlex.split(cmd_str)

    try:
        res = subprocess.run(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _LOG.info("Fallback stdout:\n%s", res.stdout)
        if res.returncode != 0:
            _LOG.error("Fallback CLI failed (exit %d). stderr:\n%s", res.returncode, res.stderr)
            raise RuntimeError(f"Fallback CLI failed: exit {res.returncode}")
        if not Path(out).exists():
            _LOG.error("Fallback CLI reported success but output file not found: %s", out)
            raise RuntimeError("Fallback CLI did not produce output file.")
        _LOG.info("Fallback CLI produced: %s", out)
        return out
    finally:
        if tmp_source_path and os.path.exists(tmp_source_path):
            try:
                os.remove(tmp_source_path)
            except Exception:
                pass

# ----------------- main synthesis function -----------------
def synthesize_openvoice(text: str, ref_wav: str, out_path: str, device: Optional[str] = "cuda", ckpt_path: Optional[str] = None, allow_cli_fallback: bool = True):
    """
    Convert TTS text to audio and apply tone-color conversion using ToneColorConverter in-process.
    If in-process fails and allow_cli_fallback==True, will attempt the CLI fallback.

    Args:
        text: subtitle text to synthesize (string)
        ref_wav: path to reference wav for speaker timbre
        out_path: desired output wav path
        device: 'cuda' or 'cpu' (auto-detected if cuda available)
        ckpt_path: path to ToneColorConverter checkpoint (required for in-process)
        allow_cli_fallback: whether to attempt CLI fallback when in-process fails
    Returns:
        path to output wav on success (string) or raises on failure
    """
    # determine device
    device = "cuda" if (torch.cuda.is_available() and device and device.startswith("cuda")) else "cpu"
    _LOG.info("synthesize_openvoice: device=%s ref=%s out=%s", device, ref_wav, out_path)

    # Try in-process ToneColorConverter if checkpoint provided
    if ckpt_path:
        try:
            converter = _load_tonecolor_converter(ckpt_path, device=device)
            target_sr = _get_target_sr_from_converter(converter)

            # 1) Synthesize base speech audio from text using model-provided TTS if available,
            #    otherwise use CLI fallback for the base TTS step.
            base_src_wav_path = None
            try:
                if hasattr(converter, "synthesize_text"):
                    _LOG.info("Using converter.synthesize_text to create source audio in-process.")
                    src_tensor = converter.synthesize_text(text, device=device)
                    if isinstance(src_tensor, torch.Tensor):
                        sr = getattr(converter, "target_sample_rate", target_sr)
                        tmp_src = Path(tempfile.mkdtemp()) / "src_from_converter.wav"
                        torchaudio.save(str(tmp_src), src_tensor.cpu(), sr)
                        base_src_wav_path = str(tmp_src)
                elif hasattr(converter, "synthesize"):
                    _LOG.info("Using converter.synthesize to create source audio in-process.")
                    src_tensor = converter.synthesize(text, device=device)
                    if isinstance(src_tensor, torch.Tensor):
                        sr = getattr(converter, "target_sample_rate", target_sr)
                        tmp_src = Path(tempfile.mkdtemp()) / "src_from_converter.wav"
                        torchaudio.save(str(tmp_src), src_tensor.cpu(), sr)
                        base_src_wav_path = str(tmp_src)
            except Exception:
                _LOG.debug("In-process TTS via converter failed or not present; will fallback to CLI if needed.")

            # If we don't have a source wav produced in-process, attempt to synthesize via CLI fallback
            if base_src_wav_path is None:
                cli_template = os.environ.get("OPENVOICE_CLI_TEMPLATE", FALLBACK_CLI_TEMPLATE)
                # If CLI template supports {source}, run run_cli_fallback to produce tmp source (it will create a tmp source if needed)
                try:
                    if "{source}" in cli_template or "{text}" in cli_template:
                        _LOG.info("Generating temporary source WAV via CLI for text.")
                        tmp_src_file = Path(tempfile.mkdtemp()) / "tmp_source_for_conversion.wav"
                        run_cli_fallback(text=text, ref=ref_wav, out=str(tmp_src_file), device=device)
                        base_src_wav_path = str(tmp_src_file)
                except Exception:
                    _LOG.debug("CLI fallback for temporary source failed; will attempt direct convert if converter accepts text.")
                    base_src_wav_path = None

            # If still no source wav, attempt converter.convert_text(path) or converter.convert_text(text,se)
            if base_src_wav_path is None:
                if hasattr(converter, "convert_text"):
                    _LOG.info("Attempting converter.convert_text(text, se) flow.")
                    se = _get_speaker_embedding_for_ref(converter, ref_wav, device=device)
                    try:
                        out_tensor = converter.convert_text(text, se, device=device)
                        out_sr = getattr(converter, "target_sample_rate", target_sr)
                        if isinstance(out_tensor, torch.Tensor):
                            if out_tensor.dim() == 1:
                                out_tensor = out_tensor.unsqueeze(0)
                            torchaudio.save(out_path, out_tensor.cpu(), out_sr)
                            _LOG.info("Saved converted audio to: %s", out_path)
                            return out_path
                        else:
                            raise RuntimeError("converter.convert_text returned non-tensor.")
                    except Exception:
                        _LOG.exception("converter.convert_text failed.")
                else:
                    _LOG.debug("No converter TTS/convert_text API available; attempting waveform-based convert with CLI-provided source.")

            # At this point we hope to have a base_src_wav_path
            if base_src_wav_path:
                # load and resample the source waveform to target_sr
                src_wav_tensor, src_sr = _load_audio_tensor(base_src_wav_path, ensure_mono=True)
                if src_sr != target_sr:
                    src_wav_tensor = _resample_tensor(src_wav_tensor, src_sr, target_sr)
                src_wav_tensor = src_wav_tensor.to(torch.float32).to(device)
                # converter preprocess if available
                try:
                    if hasattr(converter, "preprocess_wav"):
                        src_proc = converter.preprocess_wav(src_wav_tensor, target_sr, device=device)
                    else:
                        src_proc = src_wav_tensor
                except Exception:
                    src_proc = src_wav_tensor

                # get speaker embedding
                se = _get_speaker_embedding_for_ref(converter, ref_wav, device=device)

                # perform conversion
                _LOG.info("Performing in-process conversion (waveform-based).")
                with torch.no_grad():
                    if hasattr(converter, "convert"):
                        converted = converter.convert(src_proc, se, device=device)
                    elif hasattr(converter, "infer"):
                        converted = converter.infer(src_proc, se, device=device)
                    else:
                        raise RuntimeError("Converter object has no 'convert' or 'infer' method.")

                # normalized output tensor -> save
                if isinstance(converted, torch.Tensor):
                    out_tensor = converted.cpu()
                else:
                    out_tensor = torch.tensor(converted, dtype=torch.float32).cpu()

                # ensure shape (channels, samples)
                if out_tensor.dim() == 1:
                    out_tensor = out_tensor.unsqueeze(0)
                elif out_tensor.dim() == 2 and out_tensor.shape[0] > 2:
                    out_tensor = out_tensor.mean(dim=0, keepdim=True)

                out_sr = getattr(converter, "target_sample_rate", target_sr)
                out_dir = Path(out_path).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                torchaudio.save(out_path, out_tensor, out_sr)
                _LOG.info("Saved converted audio to: %s (sr=%d)", out_path, out_sr)
                return out_path

            # If reached here, in-process attempt could not produce a source and no direct convert_text API
            _LOG.warning("In-process converter could not synthesize a source or convert text directly.")
            raise RuntimeError("In-process ToneColorConverter path incomplete (no source).")

        except Exception as e:
            _LOG.exception("In-process ToneColorConverter failed: %s", e)
            if not allow_cli_fallback:
                raise

    # 2) CLI fallback (configurable)
    _LOG.info("Falling back to CLI.")
    return run_cli_fallback(text=text, ref=ref_wav, out=out_path, device=device)

# ----------------- CLI test script -----------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt", default=None, help="Path to ToneColorConverter checkpoint for in-process attempt")
    args = p.parse_args()
    try:
        res = synthesize_openvoice(args.text, args.ref, args.out, device=args.device, ckpt_path=args.ckpt, allow_cli_fallback=True)
        print("Wrote:", res)
    except Exception as e:
        print("Synthesis failed:", e)
        raise

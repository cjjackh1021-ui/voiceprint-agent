"""Standalone Windows GPU / isolated Zero-Shot voice cloning worker for SubprocessNeuralSynthesizer.

Intended to run inside a separate virtual environment (e.g. D:/voice confirm/.venv_tts)
where optional heavy TTS/VC dependencies such as F5-TTS or CosyVoice2 are installed,
keeping the core voiceprint-agent environment dependency-free.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf


def _infer_f5_tts(req: dict) -> tuple[np.ndarray, int, dict]:
    if req.get('mode') != 'text':
        raise ValueError(f"unsupported_mode: f5_tts engine only supports 'text' mode, got {req.get('mode')!r}")

    try:
        from f5_tts.api import F5TTS
    except ImportError as exc:
        raise RuntimeError(
            '未安装 f5-tts。请在独立 Windows 虚拟环境中执行：pip install f5-tts'
        ) from exc

    device = 'cpu'
    try:
        import torch

        if torch.cuda.is_available():
            device = 'cuda:0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
    except ImportError:
        torch = None  # type: ignore[assignment]

    ref_file = str(req['reference_audio_path'])
    ref_text = str(req.get('reference_text') or '')
    gen_text = str(req.get('text') or '').strip()
    out_wav = str(req['output_wav_path'])
    seed = int(req.get('seed') or 0)

    # Redirect noisy third-party init/tqdm prints away from stdout JSON protocol.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            tts = F5TTS(device=device)
        except TypeError:
            tts = F5TTS()
        try:
            infer_res = tts.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=gen_text,
                file_wave=out_wav,
                seed=seed,
            )
        except TypeError:
            infer_res = tts.infer(ref_file=ref_file, ref_text=ref_text, gen_text=gen_text, file_wave=out_wav)

    if Path(out_wav).is_file():
        wav, sr = sf.read(out_wav, dtype='float32', always_2d=False)
    elif isinstance(infer_res, tuple) and len(infer_res) >= 2:
        wav, sr = np.asarray(infer_res[0], dtype=np.float32), int(infer_res[1])
    else:
        raise RuntimeError('F5TTS 未生成音频返回值或输出文件。')

    if torch is not None and device.startswith('cuda'):
        torch.cuda.empty_cache()

    return np.asarray(wav, dtype=np.float32), int(sr), {'engine': 'f5_tts', 'device': device}


def _infer_cosyvoice2(req: dict, model_dir: str | None) -> tuple[np.ndarray, int, dict]:
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore[import-not-found]
        from cosyvoice.utils.file_utils import load_wav as cosy_load_wav  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            '未安装 cosyvoice 运行环境。请先安装 CosyVoice2 依赖并通过 --model-dir 指定模型目录。'
        ) from exc

    if not model_dir:
        raise RuntimeError('使用 --engine cosyvoice2 时必须通过 --model-dir 指定本地权重目录。')

    mode = req.get('mode')
    ref_file = str(req['reference_audio_path'])
    ref_text = str(req.get('reference_text') or '')
    with contextlib.redirect_stdout(sys.stderr):
        model = CosyVoice2(model_dir)
        prompt_speech_16k = cosy_load_wav(ref_file, 16000)
        if mode == 'text':
            chunks = [
                out['tts_speech'].squeeze(0).cpu().numpy()
                for out in model.inference_zero_shot(str(req.get('text') or ''), ref_text, prompt_speech_16k)
            ]
        elif mode == 'audio':
            src_speech_16k = cosy_load_wav(str(req['source_audio_path']), 16000)
            chunks = [
                out['tts_speech'].squeeze(0).cpu().numpy()
                for out in model.inference_vc(src_speech_16k, prompt_speech_16k)
            ]
        else:
            raise ValueError(f'unsupported_mode: {mode!r}')

    if not chunks:
        raise RuntimeError('CosyVoice2 未返回有效音频切片。')
    wav = np.concatenate(chunks, axis=-1).astype(np.float32)
    return wav, int(getattr(model, 'sample_rate', 22050)), {'engine': 'cosyvoice2', 'device': 'auto'}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Isolated neural voice cloning worker for voiceprint-agent')
    parser.add_argument('--engine', default='f5_tts', choices=['f5_tts', 'cosyvoice2'])
    parser.add_argument('--model-dir', default=None, help='Optional local model weight directory (required for cosyvoice2)')
    parser.add_argument('--request', required=True, type=Path, help='Path to UTF-8 SimulationRequest JSON file')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    try:
        req = json.loads(args.request.read_text(encoding='utf-8'))
        if args.engine == 'f5_tts':
            wav, sr, meta = _infer_f5_tts(req)
        else:
            wav, sr, meta = _infer_cosyvoice2(req, args.model_dir)

        gain_scale = float(req.get('gain_scale') or 0.85)
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim == 2:
            # Support both (T, C) and channel-first (C, T) outputs from neural backends
            axis = 1 if arr.shape[1] <= arr.shape[0] else 0
            arr = arr.mean(axis=axis)
        elif arr.ndim > 2:
            raise ValueError(f'Unsupported audio tensor shape: {tuple(arr.shape)}')
        arr = arr.reshape(-1)
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > 0.98 or gain_scale < 0.99:
            scale = min(1.0, (0.92 * min(1.0, max(0.1, gain_scale))) / max(peak, 1e-6))
            arr = np.clip(arr * scale, -1.0, 1.0).astype(np.float32)

        out_path = Path(req['output_wav_path'])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), arr, sr)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(json.dumps({'ok': True, **meta, 'elapsed_ms': round(elapsed_ms, 2)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        err_payload = {'ok': False, 'error': type(exc).__name__, 'message': str(exc)}
        sys.stderr.write(json.dumps(err_payload, ensure_ascii=False) + '\n')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

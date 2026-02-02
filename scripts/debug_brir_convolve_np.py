#!/usr/bin/env python3
"""Convolve an input wav with a stereo BRIR using numpy FFT.

Usage:
  python scripts/debug_brir_convolve_np.py --input in.wav --brir brir_stereo.wav --out out.wav
  python scripts/debug_brir_convolve_np.py --input in.wav --brir brir_stereo.wav --out out.wav --no_rms_match
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np

from sound_foundry.pipeline.data_generator import _read_wav_info


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    info = _read_wav_info(path)
    if info.audio_format not in (1, 3):
        raise ValueError(f"Unsupported WAV format {info.audio_format} in {path}")
    if info.bits_per_sample % 8 != 0:
        raise ValueError(
            f"Unsupported bits_per_sample {info.bits_per_sample} in {path}"
        )

    dtype: np.dtype
    if info.audio_format == 1:
        if info.bits_per_sample == 16:
            dtype = np.dtype("<i2")
        elif info.bits_per_sample == 32:
            dtype = np.dtype("<i4")
        elif info.bits_per_sample == 8:
            dtype = np.dtype("<u1")
        else:
            raise ValueError(
                f"Unsupported PCM bits_per_sample {info.bits_per_sample} in {path}"
            )
    else:
        if info.bits_per_sample == 32:
            dtype = np.dtype("<f4")
        elif info.bits_per_sample == 64:
            dtype = np.dtype("<f8")
        else:
            raise ValueError(
                f"Unsupported float bits_per_sample {info.bits_per_sample} in {path}"
            )

    with path.open("rb") as f:
        f.seek(info.data_offset)
        raw = f.read(info.data_size)

    data = np.frombuffer(raw, dtype=dtype)
    if info.channels > 1:
        data = data.reshape(-1, info.channels)
    else:
        data = data.reshape(-1, 1)

    if info.audio_format == 1:
        if info.bits_per_sample == 8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            peak = float(2 ** (info.bits_per_sample - 1))
            data = data.astype(np.float32) / peak
    else:
        data = data.astype(np.float32)

    return data, info.sample_rate


def _resample(signal: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return signal
    ratio = dst_rate / src_rate
    n_out = int(round(signal.shape[0] * ratio))
    if n_out <= 1:
        return signal[:1].copy()
    x_old = np.linspace(0.0, 1.0, signal.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    out = np.empty((n_out, signal.shape[1]), dtype=np.float32)
    for ch in range(signal.shape[1]):
        out[:, ch] = np.interp(x_new, x_old, signal[:, ch])
    return out


def _fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = x.shape[0] + h.shape[0] - 1
    n_fft = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, n_fft)
    H = np.fft.rfft(h, n_fft)
    y = np.fft.irfft(X * H, n_fft)[:n]
    return y.astype(np.float32)


def _write_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    data = np.clip(data, -1.0, 1.0)
    int_data = (data * 32767.0).astype(np.int16)
    with path.open("wb") as f:
        import wave

        with wave.open(f, "wb") as wf:
            wf.setnchannels(int_data.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int_data.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--brir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--keep_full", action="store_true")
    parser.add_argument("--no_rms_match", action="store_true")
    args = parser.parse_args()

    x, sr = _read_wav(args.input)
    h, h_sr = _read_wav(args.brir)

    if h.shape[1] != 2:
        raise ValueError(f"BRIR must be stereo, got {h.shape[1]} channels")

    h = _resample(h, h_sr, sr)

    # mimic prior behavior: use first channel as mono source
    x_mono = x[:, 0]

    y_l = _fft_convolve(x_mono, h[:, 0])
    y_r = _fft_convolve(x_mono, h[:, 1])

    if not args.keep_full:
        y_l = y_l[: x_mono.shape[0]]
        y_r = y_r[: x_mono.shape[0]]

    if not args.no_rms_match:
        rms_in = math.sqrt(float(np.mean(x_mono**2))) if x_mono.size else 0.0
        rms_out = (
            math.sqrt(float(np.mean((y_l**2 + y_r**2) * 0.5))) if y_l.size else 0.0
        )
        if rms_in > 0 and rms_out > 0:
            scale = rms_in / rms_out
            peak_out = (
                float(np.max(np.abs(np.stack([y_l, y_r], axis=1)))) if y_l.size else 0.0
            )
            if peak_out > 0:
                scale = min(scale, 0.99 / peak_out)
            y_l *= scale
            y_r *= scale

    y = np.stack([y_l, y_r], axis=1)

    rms = math.sqrt(float(np.mean(y**2))) if y.size else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    print(f"Output RMS={rms:.6f}, peak={peak:.6f}")

    _write_wav(args.out, y, sr)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

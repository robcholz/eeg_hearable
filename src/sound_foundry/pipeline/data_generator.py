from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Iterable
import subprocess
import tempfile
import shutil
import hashlib
import math
import logging
import struct
import wave

import numpy as np

from sound_foundry.version_control.version_control import get_current_data_folder

from sound_foundry.pipeline.dynamic_audio_decorator import DynamicEffectDecorationResult
from sound_foundry.pipeline.source_selector import SourceSelectionResult
from sound_foundry.pipeline.transient_effect_builder import (
    TransientEffectBuildingResult,
)
from sound_foundry.pipeline.util import get_build_cache_dir
from sound_foundry.data_accessor.clip import Clip

LOG = logging.getLogger("sound_foundry")


@dataclass(frozen=True, slots=True)
class AudioManifest:
    dynamic_effect: DynamicEffectDecorationResult
    file_id: int


def _probe_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ffprobe duration failed: "
            f"exit={result.returncode}, stderr={result.stderr.strip()}"
        )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned invalid duration: {result.stdout!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class _WavInfo:
    audio_format: int
    channels: int
    sample_rate: int
    bits_per_sample: int
    block_align: int
    data_offset: int
    data_size: int


def _read_wav_info(path: Path) -> _WavInfo:
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) < 12:
            raise ValueError(f"Invalid WAV header: {path}")
        riff = header[:4]
        wave = header[8:12]
        if riff not in (b"RIFF", b"RF64") or wave != b"WAVE":
            raise ValueError(f"Not a RIFF/WAVE file: {path}")

        data_size_override = None
        fmt: tuple[int, int, int, int, int] | None = None
        fmt_ext: bytes | None = None
        data_offset = None
        data_size = None

        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            chunk_size = struct.unpack("<I", chunk_header[4:])[0]

            if chunk_id == b"ds64":
                data = f.read(chunk_size)
                if len(data) >= 16:
                    data_size_override = struct.unpack("<Q", data[8:16])[0]
            elif chunk_id == b"fmt ":
                data = f.read(chunk_size)
                if len(data) < 16:
                    raise ValueError(f"Invalid fmt chunk in WAV: {path}")
                (
                    audio_format,
                    channels,
                    sample_rate,
                    _byte_rate,
                    block_align,
                    bits_per_sample,
                ) = struct.unpack("<HHIIHH", data[:16])
                fmt = (
                    audio_format,
                    channels,
                    sample_rate,
                    bits_per_sample,
                    block_align,
                )
                if chunk_size > 16:
                    fmt_ext = data[16:chunk_size]
            elif chunk_id == b"data":
                data_offset = f.tell()
                if chunk_size == 0xFFFFFFFF and data_size_override is not None:
                    data_size = data_size_override
                else:
                    data_size = chunk_size
                break
            else:
                f.seek(chunk_size, 1)

            if chunk_size % 2 == 1:
                f.seek(1, 1)

        if fmt is None or data_offset is None or data_size is None:
            raise ValueError(f"Missing fmt/data chunk in WAV: {path}")

        audio_format, channels, sample_rate, bits_per_sample, block_align = fmt
        if audio_format == 0xFFFE:
            if not fmt_ext or len(fmt_ext) < 24:
                raise ValueError(f"Invalid WAVE_FORMAT_EXTENSIBLE chunk: {path}")
            subformat = fmt_ext[8:24]
            if (
                subformat
                == b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
            ):
                audio_format = 1
            elif (
                subformat
                == b"\x03\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
            ):
                audio_format = 3
            else:
                raise ValueError(
                    f"Unsupported WAVE_FORMAT_EXTENSIBLE subformat: {path}"
                )
        return _WavInfo(
            audio_format=audio_format,
            channels=channels,
            sample_rate=sample_rate,
            bits_per_sample=bits_per_sample,
            block_align=block_align,
            data_offset=data_offset,
            data_size=data_size,
        )


def _read_wav_samples(path: Path) -> tuple[np.ndarray, int]:
    info = _read_wav_info(path)
    if info.audio_format not in (1, 3):
        raise ValueError(f"Unsupported WAV format {info.audio_format} in {path}")
    if info.bits_per_sample % 8 != 0:
        raise ValueError(
            f"Unsupported bits_per_sample {info.bits_per_sample} in {path}"
        )

    with path.open("rb") as f:
        f.seek(info.data_offset)
        raw = f.read(info.data_size)

    if info.audio_format == 1:
        if info.bits_per_sample == 8:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0
        elif info.bits_per_sample == 16:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif info.bits_per_sample == 24:
            bytes_arr = np.frombuffer(raw, dtype=np.uint8)
            if bytes_arr.size % 3 != 0:
                raise ValueError(f"Invalid 24-bit PCM data size in {path}")
            triplets = bytes_arr.reshape(-1, 3)
            values = (
                triplets[:, 0].astype(np.int32)
                | (triplets[:, 1].astype(np.int32) << 8)
                | (triplets[:, 2].astype(np.int32) << 16)
            )
            sign_bit = 1 << 23
            values = np.where(values & sign_bit, values - (1 << 24), values)
            data = values.astype(np.float32) / float(1 << 23)
        elif info.bits_per_sample == 32:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
        else:
            raise ValueError(
                f"Unsupported PCM bits_per_sample {info.bits_per_sample} in {path}"
            )
    else:
        if info.bits_per_sample == 32:
            data = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        elif info.bits_per_sample == 64:
            data = np.frombuffer(raw, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported float bits_per_sample {info.bits_per_sample} in {path}"
            )

    if info.channels > 1:
        data = data.reshape(-1, info.channels)
    else:
        data = data.reshape(-1, 1)

    return data, info.sample_rate


def _resample_linear(signal: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
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


def _write_wav_samples(path: Path, data: np.ndarray, sample_rate: int) -> None:
    data = np.clip(data, -1.0, 1.0)
    int_data = (data * 32767.0).astype(np.int16)
    with path.open("wb") as f:
        with wave.open(f, "wb") as wf:
            wf.setnchannels(int_data.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int_data.tobytes())


def _normalize_clip_duration(clip: Clip, duration_seconds: float) -> Clip:
    if not duration_seconds > 0:
        raise ValueError(f"duration_seconds must be > 0, got {duration_seconds}")

    cache_dir = get_build_cache_dir()
    digest = hashlib.sha1(
        f"{clip.path}:{duration_seconds}".encode("utf-8")
    ).hexdigest()[:10]
    out_path = cache_dir / f"{clip.path.stem}-dur{digest}.wav"
    if out_path.exists():
        return Clip(
            unified_label=clip.unified_label,
            underlying_label=clip.underlying_label,
            path=out_path,
        )

    original_duration = _probe_duration_seconds(clip.path)
    if original_duration <= 0:
        raise RuntimeError(f"Invalid source duration for {clip.path}")

    if original_duration < duration_seconds:
        repeat_count = math.ceil(duration_seconds / original_duration)
        loop_count = max(repeat_count - 1, 0)
        LOG.debug(
            "Normalize clip (action=repeat, id=%s, label=%s, underlying=%s, original=%.3f, target=%.3f, loops=%d)",
            clip.key,
            clip.unified_label,
            clip.underlying_label,
            original_duration,
            duration_seconds,
            loop_count,
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stream_loop",
            str(loop_count),
            "-i",
            str(clip.path),
            "-t",
            str(duration_seconds),
            "-acodec",
            "pcm_s16le",
            str(out_path),
        ]
    else:
        LOG.debug(
            "Normalize clip (action=trim, id=%s, label=%s, underlying=%s, original=%.3f, target=%.3f)",
            clip.key,
            clip.unified_label,
            clip.underlying_label,
            original_duration,
            duration_seconds,
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(clip.path),
            "-t",
            str(duration_seconds),
            "-acodec",
            "pcm_s16le",
            str(out_path),
        ]

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg normalize failed: "
            f"exit={result.returncode}, stderr={result.stderr.strip()}"
        )

    return Clip(
        unified_label=clip.unified_label,
        underlying_label=clip.underlying_label,
        path=out_path,
    )


def _normalize_source_selection(
    source_selection: SourceSelectionResult, duration_seconds: float
) -> SourceSelectionResult:
    outputs = [
        [_normalize_clip_duration(clip, duration_seconds) for clip in output_set]
        for output_set in source_selection.outputs
    ]
    return SourceSelectionResult(
        allocation_result=source_selection.allocation_result,
        outputs=outputs,
    )


def _normalize_transient_effect(
    transient_effect: TransientEffectBuildingResult, duration_seconds: float
) -> TransientEffectBuildingResult:
    outputs = [
        [_normalize_clip_duration(clip, duration_seconds) for clip in output_set]
        for output_set in transient_effect.outputs
    ]
    return TransientEffectBuildingResult(
        source_selection=transient_effect.source_selection,
        labels=transient_effect.labels,
        outputs=outputs,
    )


def _normalize_dynamic_effects(
    dynamic_effects: Sequence[DynamicEffectDecorationResult],
    source_duration_seconds: float,
    transient_duration_seconds: float,
) -> Sequence[DynamicEffectDecorationResult]:
    normalized: list[DynamicEffectDecorationResult] = []
    for dynamic_effect in dynamic_effects:
        transient_effect = dynamic_effect.transient_effect
        normalized_source = _normalize_source_selection(
            transient_effect.source_selection, source_duration_seconds
        )
        normalized_transient = _normalize_transient_effect(
            transient_effect, transient_duration_seconds
        )
        normalized_transient = TransientEffectBuildingResult(
            source_selection=normalized_source,
            labels=normalized_transient.labels,
            outputs=normalized_transient.outputs,
        )
        normalized.append(
            DynamicEffectDecorationResult(
                transient_effect=normalized_transient,
                labels=dynamic_effect.labels,
                outputs=dynamic_effect.outputs,
            )
        )
    return normalized


def _render_transient_effects(
    outputs: Sequence[Sequence[Clip]],
    temp_dir: Path,
    volume: float = 0.2,
) -> list[Path]:
    paths: list[Path] = []
    index = 0
    for output_set in outputs:
        for clip in output_set:
            out_path = temp_dir / f"transient_{index:06d}.wav"
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(clip.path),
                    "-filter:a",
                    f"volume={volume}",
                    str(out_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "ffmpeg transient render failed: "
                    f"exit={result.returncode}, stderr={result.stderr.strip()}"
                )
            paths.append(out_path)
            index += 1
    return paths


def _mix_transients(
    base_path: Path, transient_paths: Sequence[Path], output_path: Path
) -> None:
    if not transient_paths:
        shutil.copy2(base_path, output_path)
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(base_path),
    ]
    for path in transient_paths:
        cmd.extend(["-i", str(path)])

    filter_parts = []
    mix_inputs = ["[0:a]"]
    for index in range(1, len(transient_paths) + 1):
        filter_parts.append(f"[{index}:a]asetpts=PTS-STARTPTS[a{index}]")
        mix_inputs.append(f"[a{index}]")

    filter_parts.append(
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0[aout]"
    )

    cmd.extend(
        ["-filter_complex", ";".join(filter_parts), "-map", "[aout]", str(output_path)]
    )

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg mix failed: "
            f"exit={result.returncode}, stderr={result.stderr.strip()}"
        )


def _concat_wav_files(sources: Iterable[Path], output_path: Path) -> None:
    sources = list(sources)

    def _escape_concat_path(path: Path) -> str:
        return str(path).replace("'", r"'\''")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        for src in sources:
            tmp.write(f"file '{_escape_concat_path(src)}'\n")
        list_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        list_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg concat failed: "
            f"exit={result.returncode}, stderr={result.stderr.strip()}"
        )


def _extract_brir_stereo(brir_path: Path, yaw: int, output_path: Path) -> None:
    if brir_path.suffix.lower() != ".wav":
        raise ValueError(f"Unsupported BRIR format (expected WAV): {brir_path}")

    info = _read_wav_info(brir_path)
    channel_count = info.channels
    if channel_count < 2:
        raise ValueError(f"BRIR must have at least 2 channels: {brir_path}")
    if channel_count == 2:
        shutil.copy2(brir_path, output_path)
        return
    if channel_count % 2 != 0:
        raise ValueError(
            f"BRIR channel count must be even, got {channel_count}: {brir_path}"
        )
    max_yaw = channel_count // 2
    if not (0 <= yaw < max_yaw):
        raise ValueError(
            f"BRIR yaw out of range (yaw={yaw}, max={max_yaw - 1}): {brir_path}"
        )

    if info.bits_per_sample % 8 != 0:
        raise ValueError(
            f"Unsupported bits per sample ({info.bits_per_sample}) in {brir_path}"
        )

    sample_width = info.bits_per_sample // 8
    frame_size = info.block_align
    if frame_size <= 0:
        raise ValueError(f"Invalid block_align in {brir_path}: {info.block_align}")

    left_ch = yaw * 2
    right_ch = yaw * 2 + 1
    left_offset = left_ch * sample_width
    right_offset = right_ch * sample_width
    out_frame_size = sample_width * 2

    total_frames = info.data_size // frame_size
    out_data_size = total_frames * out_frame_size
    byte_rate_out = info.sample_rate * out_frame_size

    with brir_path.open("rb") as src, output_path.open("wb") as dst:
        src.seek(info.data_offset)
        dst.write(b"RIFF")
        dst.write(struct.pack("<I", 36 + out_data_size))
        dst.write(b"WAVE")
        dst.write(b"fmt ")
        dst.write(struct.pack("<I", 16))
        dst.write(
            struct.pack(
                "<HHIIHH",
                info.audio_format,
                2,
                info.sample_rate,
                byte_rate_out,
                out_frame_size,
                info.bits_per_sample,
            )
        )
        dst.write(b"data")
        dst.write(struct.pack("<I", out_data_size))

        buffer_frames = 4096
        remaining = total_frames
        while remaining > 0:
            frames_to_read = min(remaining, buffer_frames)
            chunk = src.read(frames_to_read * frame_size)
            if not chunk:
                break
            n_frames = len(chunk) // frame_size
            if n_frames == 0:
                break
            out = bytearray(n_frames * out_frame_size)
            mv = memoryview(chunk)
            for i in range(n_frames):
                frame_start = i * frame_size
                out_start = i * out_frame_size
                out[out_start : out_start + sample_width] = mv[
                    frame_start + left_offset : frame_start + left_offset + sample_width
                ]
                out[out_start + sample_width : out_start + out_frame_size] = mv[
                    frame_start
                    + right_offset : frame_start
                    + right_offset
                    + sample_width
                ]
            dst.write(out)
            remaining -= n_frames


def _apply_brir(
    input_path: Path,
    brir_clip: Clip,
    yaw: int,
    output_left: Path,
    output_right: Path,
    temp_dir: Path,
) -> None:
    brir_stereo_path = temp_dir / f"brir_yaw_{yaw:03d}.wav"
    _extract_brir_stereo(brir_clip.path, yaw, brir_stereo_path)

    x, sr = _read_wav_samples(input_path)
    h, h_sr = _read_wav_samples(brir_stereo_path)
    if h.shape[1] != 2:
        raise ValueError(f"BRIR must be stereo, got {h.shape[1]} channels")

    h = _resample_linear(h, h_sr, sr)

    # Match the previous behavior: use the first channel as mono source.
    x_mono = x[:, 0]
    y_left = _fft_convolve(x_mono, h[:, 0])
    y_right = _fft_convolve(x_mono, h[:, 1])

    # Keep output length aligned with input duration.
    target_len = x_mono.shape[0]
    y_left = y_left[:target_len]
    y_right = y_right[:target_len]

    rms_in = float(np.sqrt(np.mean(x_mono**2))) if x_mono.size else 0.0
    if rms_in > 0:
        rms_out = (
            float(np.sqrt(np.mean((y_left**2 + y_right**2) * 0.5)))
            if y_left.size
            else 0.0
        )
        if rms_out > 0:
            scale = rms_in / rms_out
            peak_out = (
                float(np.max(np.abs(np.stack([y_left, y_right], axis=1))))
                if y_left.size
                else 0.0
            )
            if peak_out > 0:
                scale = min(scale, 0.99 / peak_out)
            y_left *= scale
            y_right *= scale
        else:
            LOG.warning(
                "BRIR output RMS is zero for %s (yaw=%d); skipping RMS match.",
                brir_clip.path,
                yaw,
            )

    _write_wav_samples(output_left, y_left.reshape(-1, 1), sr)
    _write_wav_samples(output_right, y_right.reshape(-1, 1), sr)


def generate_audio_data(
    dynamic_effects: Sequence[DynamicEffectDecorationResult],
    total_duration_ms: int,
    preserve_non_dynamic_effect_output: bool,
) -> Sequence[AudioManifest]:
    manifests: list[AudioManifest] = []
    output_dir = get_current_data_folder()
    output_index = 0
    total_duration_seconds = total_duration_ms / 1000.0

    for dynamic_effect in dynamic_effects:
        source_selection = dynamic_effect.transient_effect.source_selection
        source_count = (
            len(source_selection.outputs[0]) if source_selection.outputs else 0
        )
        source_duration_seconds = (
            total_duration_seconds / source_count if source_count else 0
        )
        transient_outputs = dynamic_effect.transient_effect.outputs
        transient_count = len(transient_outputs[0]) if transient_outputs else 0
        transient_duration_seconds = (
            total_duration_seconds / transient_count if transient_count else 0
        )
        LOG.debug(
            "Per-clip durations (sources=%.3f, transients=%.3f)",
            source_duration_seconds,
            transient_duration_seconds,
        )
        normalized_dynamic_effect = _normalize_dynamic_effects(
            [dynamic_effect], source_duration_seconds, transient_duration_seconds
        )[0]
        source_selection = normalized_dynamic_effect.transient_effect.source_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_path = tmp_path / "base.wav"
            normalized_transient_outputs = (
                normalized_dynamic_effect.transient_effect.outputs
            )
            for output_local_index, output_set in enumerate(source_selection.outputs):
                output_path = output_dir / f"{output_index:}.wav"
                transient_output_set = (
                    normalized_transient_outputs[output_local_index]
                    if output_local_index < len(normalized_transient_outputs)
                    else []
                )
                dynamic_output_set = (
                    normalized_dynamic_effect.outputs[output_local_index]
                    if output_local_index < len(normalized_dynamic_effect.outputs)
                    else []
                )
                transient_paths = _render_transient_effects(
                    [transient_output_set], tmp_path
                )
                LOG.debug(
                    "Generate file %d (sources=%d, transients=%d, dynamic=%d, source_dur=%.3f, transient_dur=%.3f)",
                    output_index,
                    len(output_set),
                    len(transient_output_set),
                    len(dynamic_output_set),
                    source_duration_seconds,
                    transient_duration_seconds,
                )
                _concat_wav_files((clip.path for clip in output_set), base_path)
                _mix_transients(base_path, transient_paths, output_path)
                if dynamic_output_set:
                    if len(dynamic_output_set) > 1:
                        LOG.warning(
                            "Multiple BRIR clips provided; using the first one (count=%d).",
                            len(dynamic_output_set),
                        )
                    brir_clip, brir_yaw = dynamic_output_set[0]
                    output_left = output_dir / f"{output_index}-l.wav"
                    output_right = output_dir / f"{output_index}-r.wav"
                    _apply_brir(
                        output_path,
                        brir_clip,
                        brir_yaw,
                        output_left,
                        output_right,
                        tmp_path,
                    )
                    if not preserve_non_dynamic_effect_output:
                        output_path.unlink(missing_ok=True)
                manifests.append(
                    AudioManifest(
                        dynamic_effect=normalized_dynamic_effect, file_id=output_index
                    )
                )
                output_index += 1

    return manifests

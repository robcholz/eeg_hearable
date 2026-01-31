from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Iterable
import subprocess
import tempfile
import shutil
import hashlib
import math

from sound_foundry.version_control.version_control import get_current_data_folder

from sound_foundry.pipeline.dynamic_audio_decorator import DynamicEffectDecorationResult
from sound_foundry.pipeline.source_selector import SourceSelectionResult
from sound_foundry.pipeline.transient_effect_builder import (
    TransientEffectBuildingResult,
)
from sound_foundry.pipeline.util import get_build_cache_dir
from sound_foundry.data_accessor.clip import Clip


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
            DynamicEffectDecorationResult(transient_effect=normalized_transient)
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


def generate_audio_data(
    dynamic_effects: Sequence[DynamicEffectDecorationResult],
    total_duration_ms: int,
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
        normalized_dynamic_effect = _normalize_dynamic_effects(
            [dynamic_effect], source_duration_seconds, transient_duration_seconds
        )[0]
        source_selection = normalized_dynamic_effect.transient_effect.source_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_path = tmp_path / "base.wav"
            transient_paths = _render_transient_effects(
                normalized_dynamic_effect.transient_effect.outputs, tmp_path
            )
            for output_set in source_selection.outputs:
                output_path = output_dir / f"{output_index:}.wav"
                _concat_wav_files((clip.path for clip in output_set), base_path)
                _mix_transients(base_path, transient_paths, output_path)
                manifests.append(
                    AudioManifest(
                        dynamic_effect=normalized_dynamic_effect, file_id=output_index
                    )
                )
                output_index += 1

    return manifests

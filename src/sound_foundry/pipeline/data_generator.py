from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Iterable
import subprocess
import tempfile
import shutil

from sound_foundry.version_control.version_control import get_current_data_folder

from sound_foundry.pipeline.dynamic_audio_decorator import DynamicEffectDecorationResult
from sound_foundry.data_accessor.clip import Clip


@dataclass(frozen=True, slots=True)
class AudioManifest:
    dynamic_effect: DynamicEffectDecorationResult
    file_id: int


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
) -> Sequence[AudioManifest]:
    manifests: list[AudioManifest] = []
    output_dir = get_current_data_folder()
    output_index = 0

    for dynamic_effect in dynamic_effects:
        source_selection = dynamic_effect.transient_effect.source_selection
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_path = tmp_path / "base.wav"
            transient_paths = _render_transient_effects(
                dynamic_effect.transient_effect.outputs, tmp_path
            )
            for output_set in source_selection.outputs:
                output_path = output_dir / f"{output_index:}.wav"
                _concat_wav_files((clip.path for clip in output_set), base_path)
                _mix_transients(base_path, transient_paths, output_path)
                manifests.append(
                    AudioManifest(dynamic_effect=dynamic_effect, file_id=output_index)
                )
                output_index += 1

    return manifests

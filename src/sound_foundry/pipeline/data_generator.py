from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Iterable
import subprocess
import tempfile

from sound_foundry.version_control.version_control import get_current_data_folder

from sound_foundry.pipeline.dynamic_audio_decorator import DynamicEffectDecorationResult


@dataclass(frozen=True, slots=True)
class AudioManifest:
    dynamic_effect: DynamicEffectDecorationResult
    file_id: int


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
        for output_set in source_selection.outputs:
            output_path = output_dir / f"{output_index:}.wav"
            _concat_wav_files((clip.path for clip in output_set), output_path)
            manifests.append(
                AudioManifest(dynamic_effect=dynamic_effect, file_id=output_index)
            )
            output_index += 1

    return manifests

import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, Any

from sound_foundry.data_accessor.clip import Clip
from sound_foundry.pipeline.data_generator import AudioManifest
from sound_foundry.pipeline.dynamic_audio_decorator import BRIRYaw
from sound_foundry.config import get_raw_dataset_path
from sound_foundry.synthesis_parameter.synthesis_parameter import SynthesisParameter
from sound_foundry.version_control.version_control import (
    get_metadata_file_path,
    get_labels_file_path,
    get_git_ref,
    get_checksum,
    get_datetime,
    get_version_name,
    get_original_data_map,
    get_data_dep_folder,
    get_git_all_commits_since_last_merge,
)


def generate_metadata(
    build_parameter: SynthesisParameter, manifests: Sequence[AudioManifest]
) -> None:
    metadata_file_path = get_metadata_file_path()
    metadata_file_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "version": get_version_name(),
        "git_ref": get_git_ref(),
        "checksum": get_checksum(),
        "datetime_utc": get_datetime(),
        "build_params": asdict(build_parameter),
        "changes": [
            asdict(commit) for commit in get_git_all_commits_since_last_merge()
        ],
    }

    with metadata_file_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # generate a csv: column:
    csv_file_path = get_labels_file_path()  # may not exist
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    # columns: id(filename stem), source_labels, transient_labels, source_count, transient_count, dynamic_labels
    header = ["id", "source_labels", "transient_labels", "dynamic_labels"]
    with csv_file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for manifest in manifests:
            data_id = manifest.file_id
            source_labels = (
                manifest.dynamic_effect.transient_effect.source_selection.allocation_result.labels
            )
            transient_labels = manifest.dynamic_effect.transient_effect.labels
            if manifest.dynamic_effect is not None:
                dynamic_labels = manifest.dynamic_effect.labels
            else:
                dynamic_labels = []
            writer.writerow(
                [
                    data_id,
                    ",".join(source_labels),
                    ",".join(transient_labels),
                    ",".join(dynamic_labels),
                ]
            )

    data_map: dict[str, Any] = {"data": []}
    copy_original_files = build_parameter.export_options.copy_original_files
    dep_root = get_data_dep_folder() if copy_original_files else None
    raw_root = get_raw_dataset_path()
    copied_paths: set[Path] = set()

    def _flat_name(rel_path: Path) -> str:
        posix_path = rel_path.as_posix()
        base = posix_path.replace("/", "-")
        digest = hashlib.sha1(posix_path.encode("utf-8")).hexdigest()[:8]
        suffix = Path(base).suffix
        stem = base[: -len(suffix)] if suffix else base
        return f"{stem}-{digest}{suffix}"

    def _copy_clip_path(src_path: Path) -> Path:
        if dep_root is None:
            return src_path
        try:
            rel_path = src_path.relative_to(raw_root)
        except ValueError:
            rel_path = Path(src_path.name)
        flat_name = _flat_name(rel_path)
        dst_path = dep_root / flat_name
        if dst_path not in copied_paths:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied_paths.add(dst_path)
        return dst_path

    def _clip_to_dict(clip: Clip) -> dict[str, object]:
        data = asdict(clip)
        src_path = data["path"]
        dst_path = _copy_clip_path(src_path)
        if dep_root is not None:
            data["path"] = str(dst_path.relative_to(dep_root))
        else:
            data["path"] = str(dst_path)
        return data

    def _dynamic_clip_to_dict(
        dynamic_clip: tuple[Clip, BRIRYaw],
    ) -> dict[str, object]:
        clip, angle = dynamic_clip
        data = _clip_to_dict(clip)
        data["brir_yaw"] = angle
        return data

    effect_offsets: dict[int, int] = {}
    for manifest in manifests:
        data_id = manifest.file_id
        transient_effect = manifest.dynamic_effect.transient_effect
        source_outputs = transient_effect.source_selection.outputs
        transient_outputs = transient_effect.outputs
        dynamic_outputs = manifest.dynamic_effect.outputs

        effect_id = id(manifest.dynamic_effect)
        output_index = effect_offsets.get(effect_id, 0)
        effect_offsets[effect_id] = output_index + 1
        if output_index >= len(source_outputs):
            raise ValueError(
                "metadata generation mismatch: "
                f"output_index={output_index} for data_id={data_id} "
                f"but only {len(source_outputs)} source outputs available"
            )

        source_clips = source_outputs[output_index] if source_outputs else []
        transient_clips = transient_outputs[output_index] if transient_outputs else []
        dynamic_clips = dynamic_outputs[output_index] if dynamic_outputs else []

        data_map["stereo"] = build_parameter.dynamic_effect is not None
        data_map["data"].append(
            {
                "id": data_id,
                "source_clips": [_clip_to_dict(clip) for clip in source_clips],
                "transient_clips": [_clip_to_dict(clip) for clip in transient_clips],
                "dynamic_clips": [
                    _dynamic_clip_to_dict(dynamic_clip)
                    for dynamic_clip in dynamic_clips
                ],
            }
        )

    original_map_path = get_original_data_map()
    original_map_path.parent.mkdir(parents=True, exist_ok=True)
    with original_map_path.open("w", encoding="utf-8") as f:
        json.dump(data_map, f, indent=2, sort_keys=True)

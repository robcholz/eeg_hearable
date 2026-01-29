import csv
import json
from dataclasses import asdict
from typing import Sequence

from sound_foundry.pipeline.data_generator import AudioManifest
from sound_foundry.synthesis_parameter.synthesis_parameter import SynthesisParameter
from sound_foundry.version_control.version_control import (
    get_metadata_file_path,
    get_labels_file_path,
    get_git_ref,
    get_checksum,
    get_datetime,
    get_version_name,
    get_original_data_map,
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
    }

    with metadata_file_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # generate a csv: column:
    csv_file_path = get_labels_file_path()  # may not exist
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    # columns: id(filename stem), source_labels, transient_labels, source_count, transient_count
    header = [
        "id",
        "source_labels",
        "transient_labels",
        "source_count",
        "transient_count",
    ]
    with csv_file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for manifest in manifests:
            data_id = manifest.file_id
            source_labels = (
                manifest.dynamic_effect.transient_effect.source_selection.allocation_result.labels
            )
            transient_labels = manifest.dynamic_effect.transient_effect.labels
            source_count = len(source_labels)
            transient_count = len(transient_labels)
            writer.writerow(
                [
                    data_id,
                    ",".join(source_labels),
                    ",".join(transient_labels),
                    source_count,
                    transient_count,
                ]
            )

    data_map: dict[str, list[dict[str, object]]] = {"data": []}
    effect_offsets: dict[int, int] = {}
    for manifest in manifests:
        data_id = manifest.file_id
        transient_effect = manifest.dynamic_effect.transient_effect
        source_outputs = transient_effect.source_selection.outputs
        transient_outputs = transient_effect.outputs

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
        transient_clips = [
            clip for output_set in transient_outputs for clip in output_set
        ]

        data_map["data"].append(
            {
                "id": data_id,
                "source_clips": [
                    {**asdict(clip), "path": str(clip.path)} for clip in source_clips
                ],
                "transient_clips": [
                    {**asdict(clip), "path": str(clip.path)} for clip in transient_clips
                ],
            }
        )

    original_map_path = get_original_data_map()
    original_map_path.parent.mkdir(parents=True, exist_ok=True)
    with original_map_path.open("w", encoding="utf-8") as f:
        json.dump(data_map, f, indent=2, sort_keys=True)

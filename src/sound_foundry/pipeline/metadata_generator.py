from dataclasses import asdict
import json
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
    pass

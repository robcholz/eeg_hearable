from pathlib import Path

from sound_foundry.utils import get_project_root


def get_raw_dataset_path() -> Path:
    data_path = get_project_root() / "raw_dataset"
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path

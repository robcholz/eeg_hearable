from pathlib import Path

from sound_foundry.config import get_output_dataset_path
from sound_foundry.utils import get_project_root


def _get_snapshot_folder():
    return get_project_root().joinpath("snapshots")


_VERSION_NAME = ""


def set_current_snapshot(path: Path) -> None:
    global _VERSION_NAME
    _VERSION_NAME = path.stem


def get_current_data_folder() -> Path:
    if _VERSION_NAME == "":
        raise Exception("no snapshot name")
    path = get_output_dataset_path().joinpath(_VERSION_NAME)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_validity():
    # 1. check if git is clean, no local changes, everything is commited to cloud.
    # 2. check if the all the files in snapshots are json, and has format vXX.XX.XX
    import subprocess
    import re

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(get_project_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip()}")
    if result.stdout.strip():
        raise RuntimeError("git working tree is not clean")

    snapshot_dir = _get_snapshot_folder()
    if not snapshot_dir.exists():
        return

    version_re = re.compile(r"^v\d+\.\d+\.\d+$")
    for entry in snapshot_dir.iterdir():
        if not entry.is_file():
            raise ValueError(f"snapshot entry is not a file: {entry}")
        if entry.suffix != ".json":
            raise ValueError(f"snapshot must be a .json file: {entry}")
        if not version_re.match(entry.stem):
            raise ValueError(f"snapshot name must be vXX.XX.XX: {entry.name}")

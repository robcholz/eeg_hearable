from pathlib import Path
import csv
import shutil
import tempfile
import urllib.request
import zipfile
import unittest
from typing import Optional

from huggingface_hub import snapshot_download, hf_hub_download

from utils import download_with_progress


def download_fsd50k(dataset: Path):
    print("Start downloading fsd50k")
    repo_id = "Fhrozen/FSD50k"
    dev_path = "clips/dev"
    eval_path = "clips/eval"
    output_dir = Path("cache")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_root = dataset / "fsd50k"
    target_root.mkdir(parents=True, exist_ok=True)

    dev_labels_path = hf_hub_download(
        repo_id=repo_id, filename="labels/dev.csv", repo_type="dataset"
    )
    eval_labels_path = hf_hub_download(
        repo_id=repo_id, filename="labels/eval.csv", repo_type="dataset"
    )
    shutil.copy2(dev_labels_path, target_root / "dev.csv")
    shutil.copy2(eval_labels_path, target_root / "eval.csv")

    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=output_dir,
                      allow_patterns=[f"{dev_path}/**", f"{eval_path}/**"], )
    target_dev = target_root / "dev"
    target_eval = target_root / "eval"

    if target_dev.exists():
        raise FileExistsError(f"Target already exists: {target_dev}")
    if target_eval.exists():
        raise FileExistsError(f"Target already exists: {target_eval}")

    src_dev = output_dir / dev_path
    src_eval = output_dir / eval_path
    if src_dev.exists():
        shutil.move(str(src_dev), str(target_root))
    if src_eval.exists():
        shutil.move(str(src_eval), str(target_root))
    clips_dir = output_dir / "clips"
    if clips_dir.exists() and not any(clips_dir.iterdir()):
        clips_dir.rmdir()


def download_esc50(dataset: Path):
    print("Start downloading esc50")
    repo_zip_url = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
    output_dir = Path("cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "esc50.zip"
    if not zip_path.exists():
        download_with_progress(repo_zip_url, zip_path)

    target_root = dataset / "esc50"
    target_root.mkdir(parents=True, exist_ok=True)
    target_dev = target_root / "dev"
    if target_dev.exists():
        print(f"Target already exists: {target_dev}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)
        src_root = Path(tmpdir) / "ESC-50-master"
        src_audio = src_root / "audio"
        src_csv = src_root / "meta" / "esc50.csv"
        if not src_audio.exists():
            raise FileNotFoundError(f"Missing audio directory: {src_audio}")
        if not src_csv.exists():
            raise FileNotFoundError(f"Missing metadata file: {src_csv}")
        shutil.move(str(src_audio), str(target_dev))
        shutil.copy2(src_csv, target_root / "esc50.csv")


def download_musdb18(dataset: Path):
    print("Start downloading musdb18")
    # download from https://zenodo.org/records/1117372/files/musdb18.zip?download=1
    target_root = dataset / "musdb18"
    repo_zip_url = "https://zenodo.org/records/1117372/files/musdb18.zip?download=1"
    output_dir = Path("cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "musdb18.zip"

    def download_zip(force: bool = False):
        if force and zip_path.exists():
            zip_path.unlink()
        if not zip_path.exists():
            download_with_progress(repo_zip_url, zip_path)

    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        zip_path.unlink()
    download_zip()

    target_root.mkdir(parents=True, exist_ok=True)
    if any(target_root.iterdir()):
        print(f"Target already exists: {target_root}")
        return

    tmpdir_obj = None
    for attempt in range(2):
        tmpdir_obj = tempfile.TemporaryDirectory()
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmpdir_obj.name)
            break
        except zipfile.BadZipFile:
            tmpdir_obj.cleanup()
            tmpdir_obj = None
            download_zip(force=True)
            if attempt == 1:
                raise

    if tmpdir_obj is None:
        raise RuntimeError("Failed to extract musdb18 archive")

    try:
        extracted_root = Path(tmpdir_obj.name)
        src_root = extracted_root / "musdb18"
        if not src_root.exists():
            entries = [p for p in extracted_root.iterdir()]
            if len(entries) == 1 and entries[0].is_dir():
                candidate = entries[0] / "musdb18"
                src_root = candidate if candidate.exists() else entries[0]
            else:
                src_root = extracted_root

        if not src_root.exists():
            raise FileNotFoundError(f"Missing extracted data in: {extracted_root}")

        for item in src_root.iterdir():
            shutil.move(str(item), str(target_root))
    finally:
        tmpdir_obj.cleanup()


def get_audio_list_by_category(dataset_path: Path, dataset: Optional[str], category: str) -> list[str]:
    """
    Args:
        dataset_path: the path to the dataset dir
        category: sound class name, e.g. 'dog', 'speech'
        dataset: optional dataset name, e.g. 'FSD50K';
                 if None, search across all datasets
    Returns:
        list of audio file paths (strings)
    """

    # todo support dataset=fsd50k
    # todo support dataset=musdb18
    # todo support dataset=disco
    if not category or not category.strip():
        raise ValueError("category must be specified")

    dataset_name = (dataset or "esc50").strip().lower()
    if dataset_name != "esc50":
        raise ValueError("Only esc50 is supported by get_audio_list_by_category right now")

    metadata_path = dataset_path / "esc50" / "esc50.csv"
    data_dir = dataset_path / "esc50" / "dev"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing ESC-50 metadata: {metadata_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"ESC-50 audio directory missing: {data_dir}")

    normalized_target = category.strip().lower()
    matched_files: list[str] = []
    with metadata_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("category", "").strip().lower() == normalized_target:
                file_path = data_dir / row["filename"]
                matched_files.append(str(file_path))

    return matched_files


class _DownloadDataTests(unittest.TestCase):
    def test_get_audio_list_by_category_esc50(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            esc50_root = dataset_path / "esc50"
            dev_dir = esc50_root / "dev"
            dev_dir.mkdir(parents=True, exist_ok=True)

            metadata_path = esc50_root / "esc50.csv"
            fieldnames = ["filename", "fold", "target", "category", "esc10", "src_file", "take"]
            rows = [
                ("1-100032-A-0.wav", "1", "0", "dog", "True", "100032", "A"),
                ("1-100038-A-14.wav", "1", "14", "chirping_birds", "False", "100038", "A"),
            ]
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerows(rows)

            for row in rows:
                (dev_dir / row[0]).write_text("audio")

            result = get_audio_list_by_category(dataset_path, "esc50", "Dog")
            self.assertEqual([str(dev_dir / rows[0][0])], result)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import csv
import shutil
import subprocess
import tempfile
import zipfile
import unittest
from typing import Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

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
    target_dev_csv = target_root / "dev.csv"
    target_eval_csv = target_root / "eval.csv"
    shutil.copy2(dev_labels_path, target_dev_csv)
    shutil.copy2(eval_labels_path, target_eval_csv)

    target_dev = target_root / "dev"
    target_eval = target_root / "eval"

    def _expected_paths(base: Path, fname: str) -> list[Path]:
        return [base / fname, base / f"{fname}.wav"]

    def _missing_files(audio_dir: Path, csv_path: Path) -> list[str]:
        missing = []
        if not csv_path.exists():
            return missing
        with csv_path.open(newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                fname = row["fname"]
                if not any(p.exists() for p in _expected_paths(audio_dir, fname)):
                    missing.append(fname)
        return missing

    dev_missing = _missing_files(target_dev, target_dev_csv)
    eval_missing = _missing_files(target_eval, target_eval_csv)
    if not dev_missing and not eval_missing:
        print(f"FSD50K already complete at {target_root}")
        return
    if dev_missing:
        print(f"FSD50K dev missing {len(dev_missing)} files, e.g. {dev_missing[:5]}")
    if eval_missing:
        print(f"FSD50K eval missing {len(eval_missing)} files, e.g. {eval_missing[:5]}")

    def _download_missing_files(missing: list[str], split_path: str, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in missing:
            filename = f"{split_path}/{fname}.wav"
            try:
                src_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=output_dir,
                )
            except HfHubHTTPError as exc:
                status = getattr(exc, "status_code", None)
                if status == 429:
                    print("Rate limited downloading FSD50K; please rerun later.")
                raise
            dest_path = target_dir / f"{fname}.wav"
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)

    if dev_missing:
        _download_missing_files(dev_missing, dev_path, target_dev)
    if eval_missing:
        _download_missing_files(eval_missing, eval_path, target_eval)

    dev_missing = _missing_files(target_dev, target_dev_csv)
    eval_missing = _missing_files(target_eval, target_eval_csv)
    if dev_missing or eval_missing:
        raise RuntimeError(
            f"FSD50K download completed but not all files found (dev missing {len(dev_missing)}, eval missing {len(eval_missing)})."
        )


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


def _download_from_zenodo(dataset: Path, dataset_name: str, zip_filename: str, link: str):
    target_root = dataset / dataset_name
    repo_zip_url = link
    output_dir = Path("cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / zip_filename

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
        src_root = extracted_root / dataset_name
        if not src_root.exists():
            entries = [p for p in extracted_root.iterdir()]
            if len(entries) == 1 and entries[0].is_dir():
                candidate = entries[0] / dataset_name
                src_root = candidate if candidate.exists() else entries[0]
            else:
                src_root = extracted_root

        if not src_root.exists():
            raise FileNotFoundError(f"Missing extracted data in: {extracted_root}")

        for item in src_root.iterdir():
            shutil.move(str(item), str(target_root))
    finally:
        tmpdir_obj.cleanup()


def _postprocess_musdb18(dataset: Path):
    """
    Split each stem file into its five constituent audio tracks and place them under
    `musdb18/audio/<track>/<song_name>.wav`.

    Args:
        dataset: Root path containing the already downloaded `musdb18` folder.
    """
    print("Start postprocessing musdb18")

    target_root = dataset / "musdb18"
    train_dataset = target_root / "train"
    test_dataset = target_root / "test"

    audio_base = target_root / "audio"
    track_names = ["mixture", "drum", "bass", "rest", "vocal"]
    track_dirs = {name: audio_base / name for name in track_names}
    for track_dir in track_dirs.values():
        track_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg binary is required to postprocess musdb18 but was not found in PATH")

    for split_dir in (train_dataset, test_dataset):
        if not split_dir.exists():
            continue
        for stem_path in sorted(split_dir.glob("*.stem.mp4")):
            stem_name = stem_path.name
            song_name = stem_name[: -len(".stem.mp4")] if stem_name.endswith(".stem.mp4") else stem_path.stem
            for idx, track in enumerate(track_names):
                output_path = track_dirs[track] / f"{song_name}.wav"
                if output_path.exists():
                    continue
                subprocess.run(
                    [
                        ffmpeg_bin,
                        "-y",
                        "-i",
                        str(stem_path),
                        "-map",
                        f"0:a:{idx}",
                        "-c:a",
                        "pcm_s16le",
                        "-ar",
                        "44100",
                        str(output_path),
                    ],
                    check=True,
                )


def download_disco(dataset: Path):
    print("Start downloading DISCO")
    link = "https://zenodo.org/records/4019030/files/disco_noises.zip?download=1"
    _download_from_zenodo(dataset, "disco", "disco_noises.zip", link)


def download_musdb18(dataset: Path):
    print("Start downloading musdb18")
    link = "https://zenodo.org/records/1117372/files/musdb18.zip?download=1"
    _download_from_zenodo(dataset, "musdb18", "musdb18.zip", link)
    _postprocess_musdb18(dataset)


def get_audio_categories(dataset_path: Path, dataset: Optional[str]) -> list[str]:
    """
    Return the available audio categories for supported datasets.

    Args:
        dataset_path: The directory containing the downloaded datasets.
        dataset: Optional dataset name. When None, all supported datasets are aggregated.

    Returns:
        A sorted list of categories available for the requested dataset(s).
    """

    # todo support dataset=fsd50k
    # todo support dataset=disco

    def _esc50_categories() -> list[str]:
        metadata_path = dataset_path / "esc50" / "esc50.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing ESC-50 metadata: {metadata_path}")
        categories = set()
        with metadata_path.open(newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                value = row.get("category", "").strip()
                if value:
                    categories.add(value)
        return sorted(categories)

    def _musdb_categories() -> list[str]:
        audio_root = dataset_path / "musdb18" / "audio"
        if not audio_root.exists():
            raise FileNotFoundError(f"Missing MUSDB18 audio directory: {audio_root}")
        return sorted(f"audio-{p.name}" for p in audio_root.iterdir() if p.is_dir())

    dataset_name = dataset.strip().lower() if dataset else None
    if dataset_name is None:
        combined = set(_esc50_categories())
        combined.update(_musdb_categories())
        return sorted(combined)

    if dataset_name == "esc50":
        return _esc50_categories()

    if dataset_name == "musdb18":
        return _musdb_categories()

    raise ValueError("Only esc50 and musdb18 are supported by get_audio_categories right now")


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
    # todo support dataset=disco
    if not category or not category.strip():
        raise ValueError("category must be specified")

    normalized_category = category.strip().lower()
    dataset_name = dataset.strip().lower() if dataset else None
    if dataset_name is None:
        dataset_name = "musdb18" if normalized_category.startswith("audio-") else "esc50"

    if dataset_name == "esc50":
        metadata_path = dataset_path / "esc50" / "esc50.csv"
        data_dir = dataset_path / "esc50" / "dev"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing ESC-50 metadata: {metadata_path}")
        if not data_dir.exists():
            raise FileNotFoundError(f"ESC-50 audio directory missing: {data_dir}")

        matched_files: list[str] = []
        with metadata_path.open(newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("category", "").strip().lower() == normalized_category:
                    file_path = data_dir / row["filename"]
                    matched_files.append(str(file_path))

        return matched_files

    if dataset_name == "musdb18":
        normalized_category = category.strip().lower()
        if not normalized_category.startswith("audio-"):
            raise ValueError("musdb18 categories must start with 'audio-'")
        track_name = normalized_category.split("audio-", 1)[1]
        audio_dir = dataset_path / "musdb18" / "audio" / track_name
        if not audio_dir.exists():
            raise FileNotFoundError(f"Missing MUSDB18 track folder: {audio_dir}")
        return sorted(str(p) for p in audio_dir.iterdir() if p.is_file())

    raise ValueError("Only esc50 and musdb18 are supported by get_audio_list_by_category right now")


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

    def test_get_audio_categories_esc50(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            esc50_root = dataset_path / "esc50"
            esc50_root.mkdir(parents=True, exist_ok=True)
            metadata_path = esc50_root / "esc50.csv"
            fieldnames = ["filename", "fold", "target", "category", "esc10", "src_file", "take"]
            rows = [
                ("1-100032-A-0.wav", "1", "0", "dog", "True", "100032", "A"),
                ("1-100038-A-14.wav", "1", "14", "chirping_birds", "False", "100038", "A"),
            ]
            with metadata_path.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerows(rows)

            categories = get_audio_categories(dataset_path, "esc50")
            self.assertEqual(["chirping_birds", "dog"], categories)

    def test_get_audio_categories_combined_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            esc50_root = dataset_path / "esc50"
            esc50_root.mkdir(parents=True, exist_ok=True)
            metadata_path = esc50_root / "esc50.csv"
            fieldnames = ["filename", "fold", "target", "category", "esc10", "src_file", "take"]
            esc_rows = [
                ("1-100032-A-0.wav", "1", "0", "dog", "True", "100032", "A"),
            ]
            with metadata_path.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerows(esc_rows)

            for track in ("drum", "bass"):
                (dataset_path / "musdb18" / "audio" / track).mkdir(parents=True, exist_ok=True)

            categories = get_audio_categories(dataset_path, None)
            self.assertEqual(["audio-bass", "audio-drum", "dog"], categories)

    def test_get_audio_categories_musdb18(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            tracks = ["mixture", "drum", "bass"]
            for track in tracks:
                (dataset_path / "musdb18" / "audio" / track).mkdir(parents=True, exist_ok=True)

            result = get_audio_categories(dataset_path, "musdb18")
            self.assertEqual(["audio-bass", "audio-drum", "audio-mixture"], result)

    def test_get_audio_list_by_category_musdb18(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            track_dir = dataset_path / "musdb18" / "audio" / "drum"
            track_dir.mkdir(parents=True, exist_ok=True)
            (track_dir / "song_a.wav").write_text("data")
            (track_dir / "song_b.wav").write_text("data")

            result = get_audio_list_by_category(dataset_path, "musdb18", "audio-drum")
            expected = sorted(str(track_dir / f) for f in ["song_a.wav", "song_b.wav"])
            self.assertEqual(expected, result)


if __name__ == "__main__":
    unittest.main()

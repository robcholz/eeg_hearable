from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile
from huggingface_hub import snapshot_download, hf_hub_download

DATASET_PATH = Path("dataset")
DATASET_PATH.mkdir(parents=True, exist_ok=True)


def download_fsd50k(dataset: Path):
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
    repo_zip_url = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
    output_dir = Path("cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "esc50.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(repo_zip_url, zip_path)

    target_root = dataset / "esc50"
    target_root.mkdir(parents=True, exist_ok=True)
    target_dev = target_root / "dev"
    if target_dev.exists():
        raise FileExistsError(f"Target already exists: {target_dev}")

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


download_esc50(DATASET_PATH)
download_fsd50k(DATASET_PATH)

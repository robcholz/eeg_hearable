from pathlib import Path

import download_data

DATASET_PATH = Path("raw_dataset")
DATASET_PATH.mkdir(parents=True, exist_ok=True)

download_data.download_esc50(DATASET_PATH)
download_data.download_fsd50k(DATASET_PATH)

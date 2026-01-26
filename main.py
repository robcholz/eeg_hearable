from pathlib import Path

import download_data
import synthesize_data

RAW_DATASET_PATH = Path("raw_dataset")
RAW_DATASET_PATH.mkdir(parents=True, exist_ok=True)

download_data.download_esc50(RAW_DATASET_PATH)
download_data.download_musdb18(RAW_DATASET_PATH)
download_data.download_disco(RAW_DATASET_PATH)
download_data.download_fsd50k(RAW_DATASET_PATH)

DATASET_PATH = Path("dataset")
DATASET_PATH.mkdir(parents=True, exist_ok=True)

# todo

# download_data.get_audio_list_by_category(RAW_DATASET_PATH, dataset="esc50", category="dog")
# synthesize_data.synthesize_soundscape()

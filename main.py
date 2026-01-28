from pathlib import Path
from typing import Mapping

import download_data
import rawdata_accessor
import synthesize_data

RAW_DATASET_PATH = Path("raw_dataset")
RAW_DATASET_PATH.mkdir(parents=True, exist_ok=True)

download_data.download_esc50(RAW_DATASET_PATH)
download_data.download_musdb18(RAW_DATASET_PATH)
download_data.download_disco(RAW_DATASET_PATH)
download_data.download_fsd50k(RAW_DATASET_PATH)


def print_all_dataset_info(dataset_path: Path):
    total = 0
    print('Dataset info:')
    for dataset_name in rawdata_accessor.get_all_dataset_name():
        dataset_size = 0
        for label in rawdata_accessor.get_audio_labels(dataset_path, dataset_name):
            dataset_size += len(
                rawdata_accessor.get_audio_list_by_label(dataset_path, dataset=dataset_name, label=label))
        print(f'{dataset_name}: {dataset_size} files')
        total += dataset_size
    print('Total files: ', total)


def print_all_label_info(dataset_path: Path):
    total: Mapping[str, int] = {}
    print('Label info:')
    for dataset_name in rawdata_accessor.get_all_dataset_name():
        for label in rawdata_accessor.get_audio_labels(dataset_path, dataset_name):
            files = rawdata_accessor.get_audio_list_by_label(
                dataset_path,
                dataset=dataset_name,
                label=label
            )
            count = len(files)
            total[label] = total.get(label, 0) + count
    total_count = 0
    for label in total:
        total_count += total[label]
        print(f'{label}: {total[label]} files')
    print('Total files:', total_count)


def main():
    print('\n')
    print_all_dataset_info(RAW_DATASET_PATH)
    print('\n')
    print_all_label_info(RAW_DATASET_PATH)

    dataset_path = Path("output_dataset")
    dataset_path.mkdir(parents=True, exist_ok=True)
    # synthesize_data.synthesize_soundscape()


if __name__ == '__main__':
    main()

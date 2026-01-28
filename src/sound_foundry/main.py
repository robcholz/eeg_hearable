from pathlib import Path

from .data_accessor import download_data, print_all_dataset_info, print_all_label_info, get_audio_labels
from .utils import get_project_root


def main():
    raw_dataset_path = get_project_root() / "raw_dataset"
    raw_dataset_path.mkdir(parents=True, exist_ok=True)

    download_data(raw_dataset_path)
    print("\n")
    print_all_dataset_info(raw_dataset_path)
    print("\n")
    print_all_label_info(raw_dataset_path)

    dataset_path = Path("output_dataset")
    dataset_path.mkdir(parents=True, exist_ok=True)
    print(get_audio_labels(dataset_path,None))
    # synthesize_data.synthesize_soundscape()


if __name__ == "__main__":
    main()

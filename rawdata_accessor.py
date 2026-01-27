from pathlib import Path
from typing import Optional


def get_audio_categories(dataset_path: Path, dataset: Optional[str]) -> list[str]:
    """
    Return the available audio categories for supported datasets.

    Args:
        dataset_path: The directory containing the downloaded datasets.
        dataset: Optional dataset name. When None, all supported datasets are aggregated.

    Returns:
        A sorted list of categories available for the requested dataset(s).
    """
    pass


def get_audio_list_by_category(dataset_path: Path, dataset: Optional[str], category: str) -> list[str]:
    """
    Args:
        dataset_path: the path to the dataset dir
        category: sound class name
        dataset: optional dataset name, if None, search across all datasets
    Returns:
        list of audio file paths (strings)
    """
    # todo
    pass

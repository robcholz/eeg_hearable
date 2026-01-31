import argparse
import json
import logging
from pathlib import Path

from dacite import from_dict
from rich.logging import RichHandler

from sound_foundry.config import get_raw_dataset_path
from sound_foundry.data_accessor import (
    download_data,
    print_all_dataset_info,
    print_all_label_info,
)
from sound_foundry.pipeline.data_generator import generate_audio_data
from sound_foundry.pipeline.dynamic_audio_decorator import decorate_dynamic_effect
from sound_foundry.pipeline.metadata_generator import generate_metadata
from sound_foundry.pipeline.percentage_allocator import allocate_percentage
from sound_foundry.pipeline.source_selector import SourceSelector
from sound_foundry.pipeline.transient_effect_builder import build_transient_effect
from sound_foundry.pipeline.util import cleanup_buildcache
from sound_foundry.synthesis_parameter.synthesis_parameter import (
    SynthesisParameter,
)
from sound_foundry.version_control import version_control

LOG = logging.getLogger("sound_foundry")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    parser = argparse.ArgumentParser(description="Read and print a JSON file.")
    parser.add_argument("path", type=Path, help="Path to a JSON file.")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode.",
    )
    args = parser.parse_args()
    if args.dev:
        LOG.info("Enabling development mode.")

    with args.path.open("r", encoding="utf-8") as f:
        parameters_json = json.load(f)
    version_control.check_validity(args.dev)
    version_control.set_current_snapshot(args.path)

    synthesis_parameters = from_dict(SynthesisParameter, parameters_json, config=None)

    LOG.info("Parameter: %s", synthesis_parameters)

    raw_dataset_path = get_raw_dataset_path()
    download_data(raw_dataset_path)
    print("\n")
    print_all_dataset_info(raw_dataset_path)
    print("\n")
    print_all_label_info(raw_dataset_path)

    percentage = allocate_percentage(synthesis_parameters)
    source_selector = SourceSelector()
    sources = source_selector.select_source(percentage)
    transients = build_transient_effect(synthesis_parameters, sources)
    results = decorate_dynamic_effect(transients)
    try:
        manifests = generate_audio_data(results, synthesis_parameters.duration)
        generate_metadata(synthesis_parameters, manifests)
    finally:
        cleanup_buildcache()


if __name__ == "__main__":
    main()

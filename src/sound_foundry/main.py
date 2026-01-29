from pathlib import Path

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
from sound_foundry.synthesis_parameter.synthesis_parameter import (
    SynthesisParameter,
    Partition,
    Sources,
)


def main():
    raw_dataset_path = get_raw_dataset_path()

    download_data(raw_dataset_path)
    print("\n")
    print_all_dataset_info(raw_dataset_path)
    print("\n")
    print_all_label_info(raw_dataset_path)

    dataset_path = Path("output_dataset")
    dataset_path.mkdir(parents=True, exist_ok=True)

    synthesis_parameters = SynthesisParameter(
        total_number=50,
        partitions=[
            Partition(percentage=0.5, n_sources=2),
            Partition(percentage=0.5, n_sources=3),
        ],
        sources=Sources(labels=["animal", "impacts", "music", "weather", "alerts"]),
        transient_effect=None,
        dynamic_effect=None,
    )

    percentage = allocate_percentage(synthesis_parameters)
    source_selector = SourceSelector()
    sources = source_selector.select_source(percentage)
    transients = build_transient_effect(synthesis_parameters, sources)
    results = decorate_dynamic_effect(transients)
    manifests = generate_audio_data(results)
    generate_metadata(synthesis_parameters, manifests)


if __name__ == "__main__":
    main()

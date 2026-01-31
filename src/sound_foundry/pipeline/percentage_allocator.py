from dataclasses import dataclass
import logging
from math import floor
from typing import Sequence

from sound_foundry.synthesis_parameter.synthesis_parameter import (
    SynthesisParameter,
    Partition,
    Label,
)

LOG = logging.getLogger("sound_foundry")


@dataclass(frozen=True, slots=True)
class SourceAllocationResult:
    partition: Partition
    labels: Sequence[Label]
    actual_size: int


def _allocate_labels(
    partition: Partition, synthesis_parameter: SynthesisParameter
) -> Sequence[Label]:
    # 1. check if there is overlap
    # 2. if there is no overlap, just allocate them in series
    available_labels = synthesis_parameter.sources.labels
    available_labels = sorted(available_labels)

    number_of_labels_required_in_partition = sum(
        p.n_sources for p in synthesis_parameter.partitions
    )

    overlap = len(available_labels) < number_of_labels_required_in_partition

    # make it reproducible
    partitions = sorted(synthesis_parameter.partitions)

    # no overlap
    if not overlap:
        cursor = 0
        for current_partition in partitions:
            # todo, currently, if two partitions have the same settings, the program will identify them as same
            if current_partition == partition:
                lower_retrieve = cursor
                upper_retrieve = cursor + current_partition.n_sources

                # get labels in [lower_retrieve,upper_retrieve)
                return available_labels[lower_retrieve:upper_retrieve]

            cursor += current_partition.n_sources

        raise ValueError("unknown partition")

    # overlap
    cursor = 0
    for current_partition in partitions:
        if current_partition == partition:
            labels: list[Label] = []
            for _ in range(current_partition.n_sources):
                labels.append(available_labels[cursor % len(available_labels)])
                cursor += 1
            return labels

        cursor += current_partition.n_sources

    raise ValueError("unknown partition")


def allocate_percentage(
    synthesis_parameter: SynthesisParameter,
) -> list[SourceAllocationResult]:
    # 1. calculate the number of each partition by multiplying each percentage by the required number
    # 2. you know the exact number of sources for each partition, now try to allocate them with specific labels from the Sources,
    # try to avoid overlap, but if there are not enough sources, you can
    # for step 2, _allocate_labels, there is way more things you can do about the overlap strategy!
    # 3. you now have specific labels for each partition. that is it! return them.

    results = []
    for partition in synthesis_parameter.partitions:
        LOG.info(
            "Allocate partition (percentage=%.3f, n_sources=%d, n_transients=%d)",
            partition.percentage,
            partition.n_sources,
            partition.n_transients,
        )
        results.append(
            SourceAllocationResult(
                partition=partition,
                actual_size=floor(
                    synthesis_parameter.total_number * partition.percentage
                ),
                labels=_allocate_labels(
                    partition=partition, synthesis_parameter=synthesis_parameter
                ),
            )
        )

    return results

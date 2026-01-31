import pytest

from sound_foundry.pipeline.percentage_allocator import (
    _allocate_labels,
    allocate_percentage,
)
from sound_foundry.synthesis_parameter.synthesis_parameter import (
    Partition,
    Sources,
    SynthesisParameter,
    ExportOption,
)


def _make_params(partitions, total_number=10, labels=None):
    return SynthesisParameter(
        total_number=total_number,
        duration=1000,
        partitions=partitions,
        sources=Sources(labels=tuple(labels or ())),
        export_options=ExportOption(copy_original_files=False),
    )


def test_allocate_labels_no_overlap_single_partition_returns_sorted_subset():
    partition = Partition(percentage=1.0, n_sources=2, n_transients=0)
    params = _make_params([partition], labels=["zeta", "alpha", "beta"])

    labels = _allocate_labels(partition=partition, synthesis_parameter=params)

    assert labels == ["alpha", "beta"]


def test_allocate_labels_unknown_partition_raises_value_error():
    partition = Partition(percentage=1.0, n_sources=1, n_transients=0)
    params = _make_params([partition], labels=["alpha"])
    other_partition = Partition(percentage=0.5, n_sources=1, n_transients=0)

    with pytest.raises(ValueError, match="unknown partition"):
        _allocate_labels(partition=other_partition, synthesis_parameter=params)


def test_allocate_labels_overlap_raises_not_implemented():
    partition = Partition(percentage=1.0, n_sources=3, n_transients=0)
    params = _make_params([partition], labels=["alpha", "beta"])

    labels = _allocate_labels(partition=partition, synthesis_parameter=params)

    assert labels == ["alpha", "beta", "alpha"]


def test_allocate_percentage_single_partition_uses_floor_and_labels():
    partition = Partition(percentage=0.35, n_sources=2, n_transients=0)
    params = _make_params(
        [partition], total_number=10, labels=["gamma", "alpha", "beta"]
    )

    results = allocate_percentage(params)

    assert len(results) == 1
    assert results[0].partition is partition
    assert results[0].actual_size == 3
    assert results[0].labels == ["alpha", "beta"]

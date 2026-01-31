import pytest

from sound_foundry.synthesis_parameter.synthesis_parameter import (
    Partition,
    Sources,
    TransientEffect,
    _verify_partitions,
    _verify_sources,
    SynthesisParameter,
    verify_synthesis_parameter,
    ExportOption,
)


def _make_params(partitions, total_number=1, sources=None):
    sources_labels = tuple(sources or ("animal",))
    return SynthesisParameter(
        total_number=total_number,
        duration=1000,
        partitions=partitions,
        sources=Sources(labels=sources_labels),
        export_options=ExportOption(copy_original_files=False),
    )


def _patch_available_labels(monkeypatch, labels):
    monkeypatch.setattr(
        "sound_foundry.synthesis_parameter.synthesis_parameter.get_audio_labels",
        lambda *_: list(labels),
    )


@pytest.fixture(autouse=True)
def _default_dataset(monkeypatch):
    _patch_available_labels(monkeypatch, ["animal", "music", "vehicle"])


def test_verify_synthesis_parameter_rejects_empty_partitions():
    with pytest.raises(ValueError, match="partitions cannot be empty"):
        verify_synthesis_parameter(_make_params([]))


def test_verify_synthesis_parameter_rejects_non_positive_duration():
    with pytest.raises(ValueError, match="duration must be positive"):
        verify_synthesis_parameter(
            SynthesisParameter(
                total_number=1,
                duration=0,
                partitions=[Partition(percentage=1.0, n_sources=1)],
                sources=Sources(labels=("animal",)),
                export_options=ExportOption(copy_original_files=False),
            )
        )


def test_verify_synthesis_parameter_rejects_sum_greater_than_one():
    partitions = [
        Partition(percentage=0.6, n_sources=1),
        Partition(percentage=0.5, n_sources=1),
    ]
    with pytest.raises(ValueError, match="cannot exceed"):
        verify_synthesis_parameter(_make_params(partitions))


def test_verify_synthesis_parameter_allows_sum_less_than_one():
    partitions = [
        Partition(percentage=0.4, n_sources=1),
        Partition(percentage=0.5, n_sources=1),
    ]
    verify_synthesis_parameter(_make_params(partitions))


@pytest.mark.parametrize(
    "percentage",
    [-0.1, 0.0, 1.1],
)
def test_verify_synthesis_parameter_rejects_invalid_percentage(percentage):
    partitions = [Partition(percentage=percentage, n_sources=1)]
    with pytest.raises(ValueError, match="invalid percentage"):
        verify_synthesis_parameter(_make_params(partitions))


@pytest.mark.parametrize("n_sources", [0, -1])
def test_verify_synthesis_parameter_rejects_invalid_n_sources(n_sources):
    partitions = [Partition(percentage=1.0, n_sources=n_sources)]
    with pytest.raises(ValueError, match="n_sources must be > 0"):
        verify_synthesis_parameter(_make_params(partitions))


def test_verify_synthesis_parameter_accepts_valid_partitions():
    partitions = [
        Partition(percentage=0.4, n_sources=1),
        Partition(percentage=0.6, n_sources=2),
    ]
    verify_synthesis_parameter(_make_params(partitions))


def test_verify_partitions_rejects_duplicate_dataset_labels(monkeypatch):
    _patch_available_labels(monkeypatch, ["animal", "animal"])
    partitions = [
        Partition(percentage=0.5, n_sources=1),
        Partition(percentage=0.5, n_sources=1),
    ]
    with pytest.raises(ValueError, match="dataset labels must be unique"):
        _verify_partitions(partitions)


def test_verify_partitions_requires_enough_dataset_labels(monkeypatch):
    _patch_available_labels(monkeypatch, ["animal", "music"])
    partitions = [Partition(percentage=1.0, n_sources=3)]
    with pytest.raises(ValueError, match="exceeds available labels"):
        _verify_partitions(partitions)


def test_verify_sources_rejects_duplicate_labels(monkeypatch):
    _patch_available_labels(monkeypatch, ["animal"])
    with pytest.raises(ValueError, match="sources.labels must be unique"):
        _verify_sources(Sources(labels=["animal", "animal"]), None)


def test_verify_sources_rejects_unknown_labels(monkeypatch):
    _patch_available_labels(monkeypatch, ["animal"])
    with pytest.raises(ValueError, match="unknown source labels"):
        _verify_sources(Sources(labels=["music"]), None)


def test_verify_sources_allows_valid_labels(monkeypatch):
    _patch_available_labels(monkeypatch, ["animal", "music"])
    _verify_sources(
        Sources(labels=["animal"]),
        TransientEffect(labels=["music"]),
    )


def test_verify_sources_detects_overlap_with_transient(monkeypatch):
    _patch_available_labels(monkeypatch, ["animal"])
    with pytest.raises(ValueError, match="overlap"):
        _verify_sources(
            Sources(labels=["animal"]),
            TransientEffect(labels=["animal"]),
        )

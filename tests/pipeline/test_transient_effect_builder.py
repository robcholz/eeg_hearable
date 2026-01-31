from pathlib import Path

import pytest

import sound_foundry.pipeline.source_selector as source_selector_module
from sound_foundry.data_accessor.clip import Clip
from sound_foundry.pipeline.percentage_allocator import SourceAllocationResult
from sound_foundry.pipeline.source_selector import SourceSelectionResult
from sound_foundry.pipeline.transient_effect_builder import build_transient_effect
from sound_foundry.synthesis_parameter.synthesis_parameter import (
    Partition,
    SynthesisParameter,
    Sources,
    TransientEffect,
    ExportOption,
)


def _make_clip(label: str, idx: int) -> Clip:
    return Clip(
        unified_label=label,
        underlying_label=label,
        path=Path(f"{label}_{idx}.wav"),
    )


def _make_params(transient_labels: tuple[str, ...] | None) -> SynthesisParameter:
    return SynthesisParameter(
        total_number=1,
        duration=1000,
        partitions=[Partition(percentage=1.0, n_sources=2)],
        sources=Sources(labels=("a", "b")),
        export_options=ExportOption(copy_original_files=False),
        transient_effect=(
            TransientEffect(labels=transient_labels)
            if transient_labels is not None
            else None
        ),
    )


def test_build_transient_effect_selects_single_audio(monkeypatch: pytest.MonkeyPatch):
    transient_labels = ("t1", "t2")
    clip_map = {
        "t1": [_make_clip("t1", 1), _make_clip("t1", 2)],
        "t2": [_make_clip("t2", 1), _make_clip("t2", 2)],
    }

    def fake_get_audio_list_by_label(raw_path, target_datasets, label):
        return clip_map[label]

    monkeypatch.setattr(
        source_selector_module, "get_audio_list_by_label", fake_get_audio_list_by_label
    )
    monkeypatch.setattr(
        source_selector_module, "get_raw_dataset_path", lambda: Path("/tmp/raw")
    )

    allocation = SourceAllocationResult(
        partition=Partition(percentage=1.0, n_sources=2),
        labels=("a", "b"),
        actual_size=2,
    )
    outputs = [
        [_make_clip("a", 1), _make_clip("b", 1)],
        [_make_clip("a", 2), _make_clip("b", 2)],
    ]
    selection = SourceSelectionResult(allocation_result=allocation, outputs=outputs)

    results = build_transient_effect(_make_params(transient_labels), [selection])

    assert len(results) == 1
    assert results[0].source_selection.allocation_result is allocation
    assert results[0].labels == transient_labels
    assert len(results[0].outputs) == 1
    assert [clip.unified_label for clip in results[0].outputs[0]] == list(
        transient_labels
    )


def test_build_transient_effect_handles_missing_transient_effect():
    allocation = SourceAllocationResult(
        partition=Partition(percentage=1.0, n_sources=2),
        labels=("a", "b"),
        actual_size=0,
    )
    selection = SourceSelectionResult(allocation_result=allocation, outputs=[])

    results = build_transient_effect(_make_params(None), [selection])

    assert len(results) == 1
    assert results[0].labels == ()
    assert results[0].outputs == []

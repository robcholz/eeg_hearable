from pathlib import Path

import pytest

import sound_foundry.pipeline.source_selector as source_selector_module
from sound_foundry.data_accessor.clip import Clip
from sound_foundry.pipeline.percentage_allocator import SourceAllocationResult
from sound_foundry.pipeline.source_selector import SourceSelector
from sound_foundry.synthesis_parameter.synthesis_parameter import Partition


def _make_clip(label: str, idx: int) -> Clip:
    return Clip(
        unified_label=label,
        underlying_label=label,
        path=Path(f"{label}_{idx}.wav"),
    )


def test_get_source_by_label_prefers_unused_then_reuse(monkeypatch: pytest.MonkeyPatch):
    label = "alpha"
    clips = [_make_clip(label, 1), _make_clip(label, 2)]
    clip_map = {label: clips}

    def fake_get_audio_list_by_label(raw_path, target_datasets, label):
        return clip_map[label]

    monkeypatch.setattr(
        source_selector_module, "get_audio_list_by_label", fake_get_audio_list_by_label
    )
    monkeypatch.setattr(
        source_selector_module, "get_raw_dataset_path", lambda: Path("/tmp/raw")
    )

    selector = SourceSelector()

    first = selector._get_source_by_label(label)
    second = selector._get_source_by_label(label)
    third = selector._get_source_by_label(label)
    fourth = selector._get_source_by_label(label)

    assert {first, second} == set(clips)
    assert third in set(clips)
    assert fourth in set(clips)


def test_select_source_outputs_shape_and_label_order(monkeypatch: pytest.MonkeyPatch):
    labels = ["x", "y"]
    clip_map = {
        "x": [_make_clip("x", 1), _make_clip("x", 2)],
        "y": [_make_clip("y", 1), _make_clip("y", 2)],
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
        labels=labels,
        actual_size=2,
    )

    selector = SourceSelector()
    results = selector.select_source([allocation])

    assert len(results) == 1
    assert results[0].allocation_result is allocation
    assert len(results[0].outputs) == 2

    first_audio, second_audio = results[0].outputs
    assert [clip.unified_label for clip in first_audio] == labels
    assert [clip.unified_label for clip in second_audio] == labels
    assert first_audio[0] != second_audio[0]
    assert first_audio[1] != second_audio[1]


def test_get_source_by_label_single_clip_reuses_same_clip(
    monkeypatch: pytest.MonkeyPatch,
):
    label = "solo"
    clips = [_make_clip(label, 1)]
    clip_map = {label: clips}

    def fake_get_audio_list_by_label(raw_path, target_datasets, label):
        return clip_map[label]

    monkeypatch.setattr(
        source_selector_module, "get_audio_list_by_label", fake_get_audio_list_by_label
    )
    monkeypatch.setattr(
        source_selector_module, "get_raw_dataset_path", lambda: Path("/tmp/raw")
    )

    selector = SourceSelector()
    first = selector._get_source_by_label(label)
    second = selector._get_source_by_label(label)
    third = selector._get_source_by_label(label)

    assert first is clips[0]
    assert second is clips[0]
    assert third is clips[0]

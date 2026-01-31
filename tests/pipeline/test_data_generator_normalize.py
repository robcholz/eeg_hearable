from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from sound_foundry.data_accessor.clip import Clip
from sound_foundry.pipeline import data_generator
from sound_foundry.pipeline.dynamic_audio_decorator import DynamicEffectDecorationResult
from sound_foundry.pipeline.percentage_allocator import SourceAllocationResult
from sound_foundry.pipeline.source_selector import SourceSelectionResult
from sound_foundry.pipeline.transient_effect_builder import (
    TransientEffectBuildingResult,
)
from sound_foundry.synthesis_parameter.synthesis_parameter import Partition
from sound_foundry.pipeline.util import cleanup_buildcache


@dataclass
class _RunResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def test_normalize_duration_rejects_zero() -> None:
    clip = Clip(unified_label="a", underlying_label="b", path=Path("in.wav"))
    with pytest.raises(ValueError, match="duration_seconds must be > 0"):
        data_generator._normalize_clip_duration(clip, 0)


def test_normalize_duration_rejects_negative() -> None:
    clip = Clip(unified_label="a", underlying_label="b", path=Path("in.wav"))
    with pytest.raises(ValueError, match="duration_seconds must be > 0, got -1.0"):
        data_generator._normalize_clip_duration(clip, -1.0)


def test_normalize_duration_repeats_short_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        if cmd[0] == "ffprobe":
            return _RunResult(stdout="1.0")
        calls.append([str(x) for x in cmd])
        return _RunResult()

    monkeypatch.setattr(data_generator.subprocess, "run", _fake_run)
    clip = Clip(unified_label="u", underlying_label="l", path=Path("short.wav"))
    out = data_generator._normalize_clip_duration(clip, 2.5)
    assert out.path.name.endswith(".wav")
    assert any(part == "-stream_loop" for part in calls[0])
    loop_index = calls[0].index("-stream_loop")
    assert calls[0][loop_index + 1] == "2"
    assert "-t" in calls[0]
    t_index = calls[0].index("-t")
    assert calls[0][t_index + 1] == "2.5"
    cleanup_buildcache()


def test_normalize_duration_truncates_long_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        if cmd[0] == "ffprobe":
            return _RunResult(stdout="3.0")
        calls.append([str(x) for x in cmd])
        return _RunResult()

    monkeypatch.setattr(data_generator.subprocess, "run", _fake_run)
    clip = Clip(unified_label="u", underlying_label="l", path=Path("long.wav"))
    out = data_generator._normalize_clip_duration(clip, 2.0)
    assert out.unified_label == clip.unified_label
    assert out.underlying_label == clip.underlying_label
    assert "-stream_loop" not in calls[0]
    assert "-t" in calls[0]
    t_index = calls[0].index("-t")
    assert calls[0][t_index + 1] == "2.0"
    cleanup_buildcache()


def test_generate_audio_data_computes_per_clip_durations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    durations: list[tuple[float, float]] = []

    def _capture_normalize(effects, source_dur, transient_dur):
        durations.append((source_dur, transient_dur))
        return effects

    monkeypatch.setattr(
        data_generator, "_normalize_dynamic_effects", _capture_normalize
    )
    monkeypatch.setattr(data_generator, "_render_transient_effects", lambda *_: [])
    monkeypatch.setattr(data_generator, "_concat_wav_files", lambda *_: None)
    monkeypatch.setattr(data_generator, "_mix_transients", lambda *_: None)
    monkeypatch.setattr(data_generator, "get_current_data_folder", lambda: tmp_path)

    source_outputs = [
        [
            Clip("s1", "s1", Path("s1.wav")),
            Clip("s2", "s2", Path("s2.wav")),
            Clip("s3", "s3", Path("s3.wav")),
            Clip("s4", "s4", Path("s4.wav")),
        ]
    ]
    transient_outputs = [
        [
            Clip("t1", "t1", Path("t1.wav")),
            Clip("t2", "t2", Path("t2.wav")),
        ]
    ]
    allocation = SourceAllocationResult(
        partition=Partition(percentage=1.0, n_sources=4),
        labels=("s1", "s2", "s3", "s4"),
        actual_size=1,
    )
    selection = SourceSelectionResult(
        allocation_result=allocation, outputs=source_outputs
    )
    transient_effect = TransientEffectBuildingResult(
        source_selection=selection, labels=("t1", "t2"), outputs=transient_outputs
    )
    dynamic_effect = DynamicEffectDecorationResult(transient_effect=transient_effect)

    data_generator.generate_audio_data([dynamic_effect], total_duration_ms=1000)

    assert durations == [(0.25, 0.5)]

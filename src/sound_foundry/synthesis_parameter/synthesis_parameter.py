from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence, Optional

from sound_foundry.config import get_raw_dataset_path
from sound_foundry.data_accessor import (
    get_audio_labels,
    get_all_dataset_name,
    get_audio_list_by_label,
)

Label = str


@functools.total_ordering
@dataclass(frozen=True, slots=True)
class Partition:
    """One partition spec: how much of the final audio + how many sources to sample."""

    percentage: float  # e.g. 0.25 means 25%
    n_sources: int  # number of source clips in this partition

    @property
    def key(self) -> str:
        return f"{self.percentage}-{self.n_sources}"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Partition):
            return NotImplemented
        return self.key < other.key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partition):
            return NotImplemented
        return self.key == other.key


@dataclass(frozen=True, slots=True)
class Sources:
    """Labels that are allowed as sources for partitions."""

    labels: Sequence[Label] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class TransientEffect:
    """Labels that are allowed as transient effects (short events)."""

    labels: Sequence[Label] = field(default_factory=tuple)


class Effect(Enum):
    REVERBERATION = "reverberation"
    MULTI_PATH = "multi_path"


@dataclass(frozen=True, slots=True)
class DynamicEffect:
    """Dynamic effects applied to a partition or synthesis stage."""

    effects: Sequence[Effect] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SynthesisParameter:
    """
    Configuration for a complete audio synthesis job.

    This class describes **how many samples to generate** and **how each sample
    should be synthesized**. A single `SynthesisParameter` instance fully defines
    one reproducible synthesis run.

    Attributes:
        total_number:
            Total number of synthesized audio samples to generate in this run.

        partitions:
            A sequence of `Partition` objects describing how each output audio
            sample is composed. Each partition specifies the proportion of the
            final audio and how many source clips are sampled for that segment.

        sources:
            A collection of dataset labels that are allowed to be used as
            background or base sources during synthesis.

        transient_effect:
            Optional transient (short-duration) effects, such as impacts or
            brief events, that may be injected into partitions. Must not overlap
            with `sources`.

        dynamic_effect:
            Optional dynamic or global effects (e.g., reverberation, multipath)
            applied during synthesis. Kept minimal to allow future extension.
    """

    total_number: int
    partitions: Sequence[Partition]
    sources: Sources
    transient_effect: Optional[TransientEffect] = None
    dynamic_effect: Optional[DynamicEffect] = None


def verify_synthesis_parameter(
    synthesis_parameter: SynthesisParameter,
):
    _verify_total_number(synthesis_parameter.total_number)
    _verify_partitions(synthesis_parameter.partitions)
    _verify_sources(synthesis_parameter.sources, synthesis_parameter.transient_effect)
    _verify_transient_effect(
        synthesis_parameter.transient_effect, synthesis_parameter.sources
    )
    _verify_dynamic_effect(synthesis_parameter.dynamic_effect)
    pass


def _verify_total_number(total_number: int):
    # todo
    pass


def _verify_partitions(partitions: Sequence[Partition]) -> None:
    """
    Validate partition definition integrity and ensure there are enough dataset labels for sampling.

    Partition percentages must be positive, total more than zero, and cannot exceed 1.0 beyond
    the defined tolerance.
    """
    if not partitions:
        raise ValueError("partitions cannot be empty")

    available_labels = _get_available_labels()

    for p in partitions:
        if not (0.0 < p.percentage <= 1.0):
            raise ValueError(f"invalid percentage: {p.percentage}")
        if p.n_sources <= 0:
            raise ValueError(f"n_sources must be > 0, got {p.n_sources}")

    total = sum(p.percentage for p in partitions)
    if total <= 0:
        raise ValueError(f"partition percentages must sum to more than 0, got {total}")
    if total > 1.0 + 1e-6:
        raise ValueError(
            f"partition percentages cannot exceed 1.0 + {1e-6}, got {total}"
        )

    if not available_labels:
        raise ValueError("no dataset labels available to satisfy partition requests")

    max_sources = max(p.n_sources for p in partitions)
    if max_sources > len(available_labels):
        raise ValueError(
            f"n_sources request ({max_sources}) exceeds available labels ({len(available_labels)})"
        )


def _verify_sources(
    sources: Sources, transient_effect: Optional[TransientEffect]
) -> None:
    """
    Ensure source labels are unique, available in the dataset, and do not conflict with transient effects.
    """
    available_labels = set(_get_available_labels())
    source_labels = tuple(sources.labels)

    if len(source_labels) != len(set(source_labels)):
        raise ValueError("sources.labels must be unique")

    missing_labels = set(source_labels) - available_labels
    if missing_labels:
        raise ValueError(f"unknown source labels: {sorted(missing_labels)}")

    if transient_effect is None:
        return

    overlap = set(source_labels) & set(transient_effect.labels)
    if overlap:
        raise ValueError(f"sources and transient_effect overlap: {sorted(overlap)}")


def _verify_transient_effect(
    transient_effect: Optional[TransientEffect], sources: Sources
) -> None:
    """
    Verify transient effect labels are available in the dataset and separate from background sources.
    """
    if transient_effect is None:
        return

    available_labels = set(_get_available_labels())
    missing_labels = set(transient_effect.labels) - available_labels
    if missing_labels:
        raise ValueError(f"unknown transient labels: {sorted(missing_labels)}")

    overlap = set(transient_effect.labels) & set(sources.labels)
    if overlap:
        raise ValueError(f"transient_effect and sources overlap: {sorted(overlap)}")


def _verify_dynamic_effect(dynamic_effect: Optional[DynamicEffect]):
    # todo, right now left empty
    pass


def _get_available_labels() -> tuple[Label, ...]:
    labels = get_audio_labels(get_raw_dataset_path(), None)
    seen: set[Label] = set()
    duplicates: list[Label] = []
    for label in labels:
        if label in seen:
            duplicates.append(label)
        else:
            seen.add(label)
    if duplicates:
        raise ValueError(
            f"dataset labels must be unique, duplicates: {sorted(set(duplicates))}"
        )
    return tuple(labels)

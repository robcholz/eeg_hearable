import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Tuple

from sound_foundry.data_accessor.clip import Label, Clip
from sound_foundry.pipeline.source_selector import AudioSelector
from sound_foundry.pipeline.transient_effect_builder import (
    TransientEffectBuildingResult,
)
from sound_foundry.synthesis_parameter.synthesis_parameter import SynthesisParameter

LOG = logging.getLogger("sound_foundry")

BRIRYaw = int


@dataclass(frozen=True, slots=True)
class DynamicEffectDecorationResult:
    transient_effect: TransientEffectBuildingResult
    labels: Sequence[Label]
    # first sequence is the number of audio files
    # second sequence is all the sources in that audio
    outputs: Sequence[Sequence[Tuple[Clip, BRIRYaw]]]


class _DynamicEffectSelector(AudioSelector):
    def __init__(self):
        """Initialize caches used to track available and reused clips per label."""
        super().__init__()

    def select_source(
        self,
        synthesis_parameter: SynthesisParameter,
        transient_effects: Sequence[TransientEffectBuildingResult],
    ) -> Sequence[DynamicEffectDecorationResult]:
        """Expand results into concrete clip selections per audio set.

        Args:
            synthesis_parameter: SynthesisParameter
            transient_effects: Sequence[TransientEffectBuildingResult].

        Returns:
            A list of results pairing allocations with chosen clips.
        """
        results: list[DynamicEffectDecorationResult] = []
        for transient_effect in transient_effects:
            partition = transient_effect.source_selection.allocation_result.partition
            LOG.info(
                "Select dynamic effects (percentage=%.3f, n_sources=%d, n_transients=%d)",
                partition.percentage,
                partition.n_sources,
                partition.n_transients,
            )
            if synthesis_parameter.dynamic_effect is None:
                labels: Sequence[Label] = ()
                outputs: Sequence[Sequence[Tuple[Clip, BRIRYaw]]] = []
            else:
                labels = synthesis_parameter.dynamic_effect.labels
                raw_outputs = super().select(
                    transient_effect.source_selection.allocation_result.actual_size,
                    labels,
                )
                outputs = [
                    [(clip, random.randint(0, 359)) for clip in output_set]
                    for output_set in raw_outputs
                ]

            results.append(
                DynamicEffectDecorationResult(
                    transient_effect, labels=labels, outputs=outputs
                )
            )

        return results


def decorate_dynamic_effect(
    synthesis_parameter: SynthesisParameter,
    transient_results: Sequence[TransientEffectBuildingResult],
) -> Sequence[DynamicEffectDecorationResult]:
    return _DynamicEffectSelector().select_source(
        synthesis_parameter, transient_results
    )

from dataclasses import dataclass
from typing import Sequence

from sound_foundry.data_accessor.clip import Clip, Label
from sound_foundry.pipeline.source_selector import SourceSelectionResult, AudioSelector
from sound_foundry.synthesis_parameter.synthesis_parameter import SynthesisParameter


@dataclass(frozen=True, slots=True)
class TransientEffectBuildingResult:
    source_selection: SourceSelectionResult
    labels: Sequence[Label]
    # first sequence is the number of audio files
    # second sequence is all the sources in that audio
    outputs: Sequence[Sequence[Clip]]


class _TransientSelector(AudioSelector):
    def __init__(self):
        """Initialize caches used to track available and reused clips per label."""
        super().__init__()

    def select_source(
        self,
        synthesis_parameter: SynthesisParameter,
        source_selections: Sequence[SourceSelectionResult],
    ) -> Sequence[TransientEffectBuildingResult]:
        """Expand results into concrete clip selections per audio set.

        Args:
            synthesis_parameter: SynthesisParameter
            source_selections: Sequence[SourceSelectionResult].

        Returns:
            A list of results pairing allocations with chosen clips.
        """
        results: list[TransientEffectBuildingResult] = []
        for source_selection in source_selections:
            if synthesis_parameter.transient_effect is None:
                labels: Sequence[Label] = ()
                outputs: Sequence[Sequence[Clip]] = []
            else:
                labels = synthesis_parameter.transient_effect.labels
                outputs = super().select(1, labels)

            results.append(
                TransientEffectBuildingResult(
                    source_selection, labels=labels, outputs=outputs
                )
            )

        return results


def build_transient_effect(
    synthesis_parameter: SynthesisParameter,
    source_selections: Sequence[SourceSelectionResult],
) -> Sequence[TransientEffectBuildingResult]:
    return _TransientSelector().select_source(synthesis_parameter, source_selections)

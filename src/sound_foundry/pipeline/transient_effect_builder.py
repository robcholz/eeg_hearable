from dataclasses import dataclass
from typing import Sequence

from sound_foundry.pipeline.source_selector import SourceSelectionResult
from sound_foundry.synthesis_parameter.synthesis_parameter import SynthesisParameter


@dataclass(frozen=True, slots=True)
class TransientEffectBuildingResult:
    source_selection: SourceSelectionResult


def build_transient_effect(
    synthesis_parameter: SynthesisParameter,
    source_selections: Sequence[SourceSelectionResult],
) -> Sequence[TransientEffectBuildingResult]:
    # todo
    return [
        TransientEffectBuildingResult(source_selection=e) for e in source_selections
    ]

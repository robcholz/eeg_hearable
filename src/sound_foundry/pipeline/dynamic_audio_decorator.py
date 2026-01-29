from dataclasses import dataclass
from typing import Sequence

from sound_foundry.pipeline.transient_effect_builder import (
    TransientEffectBuildingResult,
)


@dataclass(frozen=True, slots=True)
class DynamicEffectDecorationResult:
    transient_effect: TransientEffectBuildingResult


def decorate_dynamic_effect(
    transient_results: Sequence[TransientEffectBuildingResult],
) -> Sequence[DynamicEffectDecorationResult]:
    # todo
    return [
        DynamicEffectDecorationResult(transient_effect=e) for e in transient_results
    ]

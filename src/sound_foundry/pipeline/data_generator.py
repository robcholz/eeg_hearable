from dataclasses import dataclass

from black.ranges import Sequence

from sound_foundry.pipeline.dynamic_audio_decorator import DynamicEffectDecorationResult


@dataclass(frozen=True, slots=True)
class AudioManifest:
    dynamic_effect: DynamicEffectDecorationResult


def generate_audio_data(
    dynamic_effects: Sequence[DynamicEffectDecorationResult],
) -> Sequence[AudioManifest]:
    for dynamic_effect in dynamic_effects:
        transient_effect = dynamic_effect.transient_effect
        source_selection = transient_effect.source_selection
        # allocation_result = source_selection.allocation_result

        print("=====================Start=====================")
        print(source_selection)
    # todo
    return []

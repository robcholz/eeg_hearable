from dataclasses import dataclass
from typing import Sequence

from sound_foundry.data_accessor.clip import Clip
from sound_foundry.pipeline.percentage_allocator import SourceAllocationResult


@dataclass(frozen=True, slots=True)
class SourceSelectionResult:
    allocation_result: SourceAllocationResult
    outputs: Sequence[Clip]


def select_source(
    source_allocations: Sequence[SourceAllocationResult],
) -> Sequence[SourceSelectionResult]:
    # todo
    pass

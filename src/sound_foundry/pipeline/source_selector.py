from dataclasses import dataclass
from typing import Sequence, Mapping

from sound_foundry.data_accessor.clip import Clip
from sound_foundry.pipeline.percentage_allocator import SourceAllocationResult


@dataclass(frozen=True, slots=True)
class SourceSelectionResult:
    allocation_result: SourceAllocationResult
    output_id_map: Mapping[int, Sequence[Clip]]


def select_source(
    source_allocations: Sequence[SourceAllocationResult],
) -> Sequence[SourceSelectionResult]:
    # todo
    pass

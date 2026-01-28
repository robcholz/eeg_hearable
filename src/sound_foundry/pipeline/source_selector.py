from dataclasses import dataclass
from typing import Sequence, Set, Dict, List, Tuple
import heapq

from sound_foundry.config import get_raw_dataset_path
from sound_foundry.data_accessor import get_audio_list_by_label
from sound_foundry.data_accessor.clip import Clip, Label
from sound_foundry.pipeline.percentage_allocator import SourceAllocationResult


@dataclass(frozen=True, slots=True)
class SourceSelectionResult:
    allocation_result: SourceAllocationResult
    # first sequence is the number of audio files
    # second sequence is all the sources in that audio
    outputs: Sequence[Sequence[Clip]]


class SourceSelector:
    def __init__(self):
        """Initialize caches used to track available and reused clips per label."""
        self.cache: Dict[Label, Set[Clip]] = {}
        self.conflict_map: Dict[Label, Set[Clip]] = {}
        self.ref_map: Dict[Label, List[Tuple[int, str, Clip]]] = {}

    def _get_source_by_label(self, label: Label) -> Clip:
        """Return a clip for the label, preferring unused clips then fair reuse.

        Args:
            label: Target label to select a clip for.

        Returns:
            A clip associated with the label.
        """
        sources_with_label = get_audio_list_by_label(
            get_raw_dataset_path(), None, label=label
        )

        # ensure cache
        if label not in self.cache:
            self.cache[label] = set(sources_with_label)

        sources_with_label = self.cache[label]

        if label not in self.conflict_map:
            self.conflict_map[label] = set()

        # fast path: anything not used yet for this label
        available_clips = sources_with_label.difference(self.conflict_map[label])
        if available_clips:
            x = next(iter(available_clips))
            self.conflict_map[label].add(x)
            return x

        # min-heap fallback for fair reuse (usage, unique_key, clip)
        if label not in self.ref_map:
            heap = [
                (0, str(clip.key), clip) for clip in self.cache[label]
            ]  # unique_key can be clip.key if you have it
            heapq.heapify(heap)
            self.ref_map[label] = heap

        usage, key, clip = heapq.heappop(self.ref_map[label])
        heapq.heappush(self.ref_map[label], (usage + 1, key, clip))
        return clip

    def select_source(
        self,
        source_allocations: Sequence[SourceAllocationResult],
    ) -> Sequence[SourceSelectionResult]:
        """Expand allocation results into concrete clip selections per audio set.

        Args:
            source_allocations: Allocation results with labels and counts to realize.

        Returns:
            A list of selection results pairing allocations with chosen clips.
        """
        results: list[SourceSelectionResult] = []
        for allocation_result in source_allocations:
            # each audio set
            outputs = []
            for audio_set_id in range(allocation_result.actual_size):
                # each single audio has multiple sources
                sources_for_one_audio = [
                    self._get_source_by_label(label)
                    for label in allocation_result.labels
                ]
                outputs.append(sources_for_one_audio)

            results.append(SourceSelectionResult(allocation_result, outputs))

        return results

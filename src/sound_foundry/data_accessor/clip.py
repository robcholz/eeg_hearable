import functools
from dataclasses import dataclass
from pathlib import Path

Label = str


@functools.total_ordering
@dataclass(frozen=True, slots=True)
class Clip:
    unified_label: Label
    underlying_label: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.path}"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Clip):
            return NotImplemented
        return self.key < other.key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Clip):
            return NotImplemented
        return self.key == other.key

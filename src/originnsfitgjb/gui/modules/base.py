from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PageFactory(Protocol):
    def __call__(self) -> object:
        """Create and return a QWidget instance."""


@dataclass(frozen=True)
class GuiModule:
    module_id: str
    title: str
    description: str
    create_page: PageFactory

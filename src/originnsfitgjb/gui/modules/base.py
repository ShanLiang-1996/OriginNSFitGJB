from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..settings import GuiSettings


class PageFactory(Protocol):
    def __call__(self, settings: GuiSettings) -> object:
        """Create and return a QWidget instance."""


@dataclass(frozen=True)
class GuiModule:
    module_id: str
    title: str
    description: str
    create_page: PageFactory

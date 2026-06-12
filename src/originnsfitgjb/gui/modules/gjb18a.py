from __future__ import annotations

from ..settings import GuiSettings
from .base import GuiModule


def create_gjb18a_module() -> GuiModule:
    return GuiModule(
        module_id="gjb18a",
        title="GJB/Z 18A 分析",
        description="批量执行 GJB/Z 18A 9.3.2 简化应变寿命拟合并生成 Origin 输出。",
        create_page=_create_page,
    )


def _create_page(settings: GuiSettings) -> object:
    from .gjb18a_page import Gjb18aPage

    return Gjb18aPage(settings)

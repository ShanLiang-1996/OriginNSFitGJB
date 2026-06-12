from __future__ import annotations

from .base import GuiModule
from .gjb18a import create_gjb18a_module


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: dict[str, GuiModule] = {}

    def register(self, module: GuiModule) -> None:
        if module.module_id in self._modules:
            raise ValueError(f"Duplicate GUI module: {module.module_id}")
        self._modules[module.module_id] = module

    def get(self, module_id: str) -> GuiModule:
        return self._modules[module_id]

    def all_modules(self) -> tuple[GuiModule, ...]:
        return tuple(self._modules.values())


def build_default_registry() -> ModuleRegistry:
    registry = ModuleRegistry()
    registry.register(create_gjb18a_module())
    return registry

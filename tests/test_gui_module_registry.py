from __future__ import annotations

import unittest

from originnsfitgjb.gui.modules.gjb18a import create_gjb18a_module
from originnsfitgjb.gui.modules.registry import ModuleRegistry, build_default_registry


class GuiModuleRegistryTests(unittest.TestCase):
    def test_default_registry_contains_gjb18a_module(self) -> None:
        registry = build_default_registry()

        module = registry.get("gjb18a")

        self.assertEqual(module.module_id, "gjb18a")
        self.assertEqual(module.title, "GJB/Z 18A 分析")

    def test_register_rejects_duplicate_module_id(self) -> None:
        registry = ModuleRegistry()
        module = create_gjb18a_module()
        registry.register(module)

        with self.assertRaisesRegex(ValueError, "Duplicate GUI module"):
            registry.register(module)

    def test_all_modules_preserve_registration_order(self) -> None:
        registry = ModuleRegistry()
        module = create_gjb18a_module()
        registry.register(module)

        self.assertEqual(registry.all_modules(), (module,))

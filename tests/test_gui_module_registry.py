from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from originnsfitgjb.gui.modules.gjb18a_page import Gjb18aPage
from originnsfitgjb.gui.modules.gjb18a import create_gjb18a_module
from originnsfitgjb.gui.modules.registry import ModuleRegistry, build_default_registry
from originnsfitgjb.gui.settings import GuiSettings


class GuiModuleRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_default_registry_contains_gjb18a_module(self) -> None:
        registry = build_default_registry()

        module = registry.get("gjb18a")

        self.assertEqual(module.module_id, "gjb18a")
        self.assertEqual(module.title, "GJB/Z 18A 分析")

    def test_default_gjb18a_factory_accepts_settings(self) -> None:
        module = build_default_registry().get("gjb18a")

        page = module.create_page(GuiSettings())
        self.addCleanup(page.deleteLater)

        self.assertIsInstance(page, Gjb18aPage)

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

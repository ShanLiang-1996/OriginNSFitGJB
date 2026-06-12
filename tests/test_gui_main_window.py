from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QWidget

from originnsfitgjb.gui.main_window import MainWindow
from originnsfitgjb.gui.modules.base import GuiModule
from originnsfitgjb.gui.modules.registry import ModuleRegistry
from originnsfitgjb.gui.settings import GuiSettings


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_module_page_factories_receive_window_settings(self) -> None:
        received_settings: list[GuiSettings] = []
        settings = GuiSettings(recent_input_dir="C:/expected")

        def create_page(settings_arg: GuiSettings) -> QWidget:
            received_settings.append(settings_arg)
            return QWidget()

        window = MainWindow(self._registry_with_factory(create_page), settings)
        self.addCleanup(window.deleteLater)

        self.assertEqual(received_settings, [settings])

    def test_close_event_preserves_existing_settings_with_latest_window_size(self) -> None:
        existing_settings = GuiSettings(recent_input_dir="C:/kept", window_width=10, window_height=20)

        def create_page(*_args: object) -> QWidget:
            return QWidget()

        window = MainWindow(self._registry_with_factory(create_page), GuiSettings())
        self.addCleanup(window.deleteLater)
        window.resize(1003, 701)
        expected_settings = replace(
            existing_settings,
            window_width=window.width(),
            window_height=window.height(),
        )

        with (
            patch("originnsfitgjb.gui.main_window.load_settings", return_value=existing_settings, create=True)
            as load_settings_mock,
            patch("originnsfitgjb.gui.main_window.save_settings", create=True) as save_settings_mock,
        ):
            window.closeEvent(QCloseEvent())

        load_settings_mock.assert_called_once_with()
        save_settings_mock.assert_called_once_with(expected_settings)

    def test_close_event_shuts_down_module_pages(self) -> None:
        shutdown_calls: list[str] = []

        class ShutdownPage(QWidget):
            def shutdown(self) -> None:
                shutdown_calls.append("shutdown")

        def create_page(*_args: object) -> QWidget:
            return ShutdownPage()

        window = MainWindow(self._registry_with_factory(create_page), GuiSettings())
        self.addCleanup(window.deleteLater)

        with (
            patch("originnsfitgjb.gui.main_window.load_settings", return_value=GuiSettings(), create=True),
            patch("originnsfitgjb.gui.main_window.save_settings", create=True),
        ):
            window.closeEvent(QCloseEvent())

        self.assertEqual(shutdown_calls, ["shutdown"])

    @staticmethod
    def _registry_with_factory(create_page: object) -> ModuleRegistry:
        registry = ModuleRegistry()
        registry.register(
            GuiModule(
                module_id="dummy",
                title="Dummy",
                description="Dummy module",
                create_page=create_page,
            )
        )
        return registry


if __name__ == "__main__":
    unittest.main()

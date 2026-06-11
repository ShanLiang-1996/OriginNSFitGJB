from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from originnsfitgjb.gui.settings import GuiSettings, load_settings, save_settings


class GuiSettingsTests(unittest.TestCase):
    def test_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            settings = GuiSettings(
                recent_input_dir="C:/data",
                recent_output_dir="C:/out",
                recent_patterns=("*.csv", "*.xlsx"),
                life_column="life",
                response_column="strain",
                status_column="status",
                audit=True,
                audit_workbook=True,
                hidden_origin=True,
                window_width=1200,
                window_height=760,
            )

            save_settings(settings, path)
            loaded = load_settings(path)

            self.assertEqual(loaded, settings)

    def test_missing_settings_file_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_settings(Path(tmp) / "settings.json")

            self.assertEqual(loaded.recent_patterns, ("*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls"))
            self.assertEqual(loaded.window_width, 1120)
            self.assertEqual(loaded.window_height, 720)

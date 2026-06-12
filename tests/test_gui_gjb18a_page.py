from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from originnsfitgjb.analysis_service import AnalysisConfig
from originnsfitgjb.gui.modules.gjb18a_page import Gjb18aPage


class Gjb18aPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_page_instantiates_with_run_button_and_default_status(self) -> None:
        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)

        self.assertEqual(page._run_button.text(), "开始全流程分析")
        self.assertEqual(page._status_label.text(), "等待运行")

    def test_config_from_form_maps_form_values_to_analysis_config(self) -> None:
        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)

        with tempfile.TemporaryDirectory() as input_tmp, tempfile.TemporaryDirectory() as output_tmp:
            input_dir = Path(input_tmp)
            output_dir = Path(output_tmp)
            page._input_dir.setText(str(input_dir))
            page._output_dir.setText(str(output_dir))
            page._patterns.setText("*.csv; *.xlsx,*.txt")
            page._life_column.setText("life")
            page._response_column.setText("strain")
            page._status_column.setText("status")
            page._level_column.setText("level")

            config = page._config_from_form()

        self.assertIsInstance(config, AnalysisConfig)
        self.assertEqual(config.input_dir, input_dir)
        self.assertEqual(config.output_dir, output_dir)
        self.assertEqual(config.patterns, ("*.csv", "*.xlsx", "*.txt"))
        self.assertEqual(config.status_column, "status")
        self.assertIs(config.dry_run, False)
        self.assertEqual(config.fit_points, 300)

    def test_config_from_form_rejects_missing_input_directory(self) -> None:
        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)

        with tempfile.TemporaryDirectory() as tmp:
            missing_input = Path(tmp) / "missing"
            page._input_dir.setText(str(missing_input))

            with self.assertRaisesRegex(ValueError, "输入目录不存在"):
                page._config_from_form()


if __name__ == "__main__":
    unittest.main()

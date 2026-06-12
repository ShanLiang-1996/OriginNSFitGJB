from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from originnsfitgjb.analysis_service import AnalysisConfig, AnalysisRunResult
from originnsfitgjb.gui.modules.gjb18a_page import Gjb18aPage


class _StoppedThread:
    def isRunning(self) -> bool:
        return False


class _FakeRunningThread:
    def __init__(self) -> None:
        self.interruption_requested = False
        self.quit_called = False
        self.wait_timeout_ms: int | None = None

    def isRunning(self) -> bool:
        return True

    def requestInterruption(self) -> None:
        self.interruption_requested = True

    def quit(self) -> None:
        self.quit_called = True

    def wait(self, timeout_ms: int) -> bool:
        self.wait_timeout_ms = timeout_ms
        return False


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

    def test_config_from_form_rejects_non_finite_confidence(self) -> None:
        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)

        with tempfile.TemporaryDirectory() as input_tmp:
            page._input_dir.setText(input_tmp)
            page._confidence.setText("nan")

            with self.assertRaisesRegex(ValueError, "置信度"):
                page._config_from_form()

    def test_config_from_form_rejects_out_of_range_confidence(self) -> None:
        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)

        with tempfile.TemporaryDirectory() as input_tmp:
            page._input_dir.setText(input_tmp)
            invalid_values = ("0", "-0.1", "1.0", "100", "101")
            for value in invalid_values:
                with self.subTest(value=value):
                    page._confidence.setText(value)

                    with self.assertRaisesRegex(ValueError, "置信度"):
                        page._config_from_form()

    def test_on_finished_restores_ui_outputs_and_clears_stopped_worker_refs(self) -> None:
        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "gjb_summary.csv"
            output_path.write_text("summary", encoding="utf-8")
            worker = object()
            page._run_button.setEnabled(False)
            page._status_label.setText("正在启动分析")
            page._thread = _StoppedThread()
            page._worker = worker

            page._on_finished(AnalysisRunResult(completed=True, output_paths=(output_path,)))

            self.assertTrue(page._run_button.isEnabled())
            self.assertEqual(page._status_label.text(), "分析完成")
            self.assertEqual([button.text() for button in page._output_buttons], [output_path.name])
            self.assertEqual(page._output_buttons[0].toolTip(), str(output_path))
            self.assertIsNone(page._thread)
            self.assertIsNone(page._worker)

    def test_shutdown_worker_requests_thread_stop_and_keeps_refs_when_still_running(self) -> None:
        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)
        thread = _FakeRunningThread()
        worker = object()
        page._thread = thread
        page._worker = worker

        page._shutdown_worker(timeout_ms=25)

        self.assertTrue(thread.interruption_requested)
        self.assertTrue(thread.quit_called)
        self.assertEqual(thread.wait_timeout_ms, 25)
        self.assertIs(page._thread, thread)
        self.assertIs(page._worker, worker)


if __name__ == "__main__":
    unittest.main()

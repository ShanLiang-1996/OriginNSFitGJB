from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from originnsfitgjb.analysis_service import (
    AnalysisConfig,
    AnalysisProgress,
    AnalysisRunResult,
    AnalysisTableFailure,
)
from originnsfitgjb.gui.modules.gjb18a_page import Gjb18aPage
from originnsfitgjb.gui.settings import GuiSettings


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


class _FakeSignal:
    def __init__(self) -> None:
        self.slots: list[object] = []

    def connect(self, slot: object) -> None:
        self.slots.append(slot)


class Gjb18aPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_page_instantiates_with_run_button_and_default_status(self) -> None:
        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)

        self.assertEqual(page._run_button.text(), "开始全流程分析")
        self.assertEqual(page._status_label.text(), "等待运行")

    def test_settings_are_applied_to_form_fields(self) -> None:
        settings = GuiSettings(
            recent_input_dir="C:/input",
            recent_output_dir="C:/output",
            recent_patterns=("*.dat", "*.csv"),
            life_column="cycles",
            response_column="strain_pct",
            status_column="state",
            level_column="stress",
            confidence=0.99,
            fit_points=42,
            outlier_mode="report-only",
            audit=True,
            audit_workbook=True,
            audit_json=True,
            hidden_origin=True,
            linearized_graph=True,
            no_runout_arrows=True,
            graph_template_path="C:/templates/gjb.otp",
            project_path="C:/output/gjb.opj",
        )

        page = Gjb18aPage(settings)
        self.addCleanup(page.deleteLater)

        self.assertEqual(page._input_dir.text(), "C:/input")
        self.assertEqual(page._output_dir.text(), "C:/output")
        self.assertEqual(page._patterns.text(), "*.dat;*.csv")
        self.assertEqual(page._life_column.text(), "cycles")
        self.assertEqual(page._response_column.text(), "strain_pct")
        self.assertEqual(page._status_column.text(), "state")
        self.assertEqual(page._level_column.text(), "stress")
        self.assertEqual(page._confidence.text(), "0.99")
        self.assertEqual(page._fit_points.value(), 42)
        self.assertEqual(page._outlier_mode.currentData(), "report-only")
        self.assertTrue(page._audit.isChecked())
        self.assertTrue(page._audit_workbook.isChecked())
        self.assertTrue(page._audit_json.isChecked())
        self.assertTrue(page._hidden_origin.isChecked())
        self.assertTrue(page._linearized_graph.isChecked())
        self.assertTrue(page._no_runout_arrows.isChecked())
        self.assertEqual(page._graph_template_path.text(), "C:/templates/gjb.otp")
        self.assertEqual(page._project_path.text(), "C:/output/gjb.opj")

    def test_settings_from_form_returns_current_form_values_and_window_size(self) -> None:
        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)
        page.resize(901, 678)
        page._input_dir.setText("C:/data")
        page._output_dir.setText("C:/out")
        page._patterns.setText("*.csv; *.xlsx,*.txt")
        page._life_column.setText("life")
        page._response_column.setText("strain")
        page._status_column.setText("status")
        page._level_column.setText("level")
        page._confidence.setText("95")
        page._fit_points.setValue(123)
        page._outlier_mode.setCurrentIndex(page._outlier_mode.findData("report-only"))
        page._audit.setChecked(True)
        page._audit_workbook.setChecked(True)
        page._audit_json.setChecked(True)
        page._hidden_origin.setChecked(True)
        page._linearized_graph.setChecked(True)
        page._no_runout_arrows.setChecked(True)
        page._graph_template_path.setText("C:/template.otp")
        page._project_path.setText("C:/project.opj")

        settings = page._settings_from_form()

        self.assertEqual(
            settings,
            GuiSettings(
                recent_input_dir="C:/data",
                recent_output_dir="C:/out",
                recent_patterns=("*.csv", "*.xlsx", "*.txt"),
                life_column="life",
                response_column="strain",
                status_column="status",
                level_column="level",
                confidence=95.0,
                fit_points=123,
                outlier_mode="report-only",
                audit=True,
                audit_workbook=True,
                audit_json=True,
                hidden_origin=True,
                linearized_graph=True,
                no_runout_arrows=True,
                graph_template_path="C:/template.otp",
                project_path="C:/project.opj",
                window_width=901,
                window_height=678,
            ),
        )

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

    def test_start_run_saves_current_settings_before_starting_worker(self) -> None:
        events: list[str] = []

        class FakeThread:
            def __init__(self) -> None:
                self.started = _FakeSignal()
                self.finished = _FakeSignal()

            def start(self) -> None:
                events.append("start")

            def quit(self) -> None:
                pass

            def deleteLater(self) -> None:
                pass

            def isRunning(self) -> bool:
                return False

        class FakeWorker:
            def __init__(self, config: AnalysisConfig) -> None:
                self.config = config
                self.progress = _FakeSignal()
                self.log = _FakeSignal()
                self.finished = _FakeSignal()

            def moveToThread(self, thread: object) -> None:
                self.thread = thread

            def run(self) -> None:
                pass

            def deleteLater(self) -> None:
                pass

        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)

        with tempfile.TemporaryDirectory() as input_tmp, tempfile.TemporaryDirectory() as output_tmp:
            page._input_dir.setText(input_tmp)
            page._output_dir.setText(output_tmp)
            with (
                patch(
                    "originnsfitgjb.gui.modules.gjb18a_page.save_settings",
                    side_effect=lambda _settings: events.append("save"),
                    create=True,
                ) as save_settings_mock,
                patch("originnsfitgjb.gui.modules.gjb18a_page.QThread", FakeThread),
                patch("originnsfitgjb.gui.modules.gjb18a_page.AnalysisWorker", FakeWorker),
            ):
                page._start_run()

        save_settings_mock.assert_called_once_with(page._settings_from_form())
        self.assertEqual(events, ["save", "start"])

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

    def test_on_finished_limits_output_buttons_for_many_files(self) -> None:
        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            paths = []
            for name in (
                "gjb_summary.csv",
                "gjb_fit_data.csv",
                "gjb_curve.csv",
                "gjb_level_stats.csv",
                "gjb_decisionlog.csv",
                "gjb_finalmle.csv",
            ):
                path = output_dir / name
                path.write_text("data", encoding="utf-8")
                paths.append(path)
            workbook = output_dir / "audit" / "gjb_audit_workbook.xlsx"
            workbook.parent.mkdir()
            workbook.write_text("workbook", encoding="utf-8")
            paths.append(workbook)

            page._on_finished(AnalysisRunResult(completed=True, output_paths=tuple(paths)))

            button_texts = [button.text() for button in page._output_buttons]
            self.assertLessEqual(len(button_texts), 3)
            self.assertIn("打开输出目录", button_texts)
            self.assertIn("gjb_summary.csv", button_texts)

    def test_progress_events_are_appended_to_log(self) -> None:
        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)

        page._on_progress(
            AnalysisProgress(
                phase="fit",
                message="Running GJB analyses.",
                current=1,
                total=3,
            )
        )

        self.assertIn("Running GJB analyses. (1/3)", page._log.toPlainText())

    def test_on_finished_displays_partial_success_failures(self) -> None:
        page = Gjb18aPage()
        self.addCleanup(page.deleteLater)

        page._on_finished(
            AnalysisRunResult(
                completed=True,
                table_failures=(AnalysisTableFailure(label="bad", message="cannot read"),),
            )
        )

        self.assertEqual(page._status_label.text(), "部分完成")
        self.assertIn("bad: cannot read", page._log.toPlainText())

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

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from originnsfitgjb.analysis_service import AnalysisConfig, run_analysis
from originnsfitgjb.origin_client import OriginAutomationError


class AnalysisServiceConfigTests(unittest.TestCase):
    def test_default_patterns_match_cli_defaults(self) -> None:
        config = AnalysisConfig()

        self.assertEqual(
            config.patterns,
            ("*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls"),
        )

    def test_audit_dir_defaults_under_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            config = AnalysisConfig(output_dir=output_dir)

            self.assertEqual(config.resolved_audit_dir(), output_dir / "audit")

    def test_explicit_audit_dir_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "review"
            config = AnalysisConfig(audit_dir=audit_dir)

            self.assertEqual(config.resolved_audit_dir(), audit_dir)

    def test_project_path_defaults_under_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            config = AnalysisConfig(output_dir=output_dir)

            self.assertEqual(config.resolved_project_path(), output_dir / "gjb_analysis.opj")


ROOT = Path(__file__).resolve().parents[1]


class AnalysisServiceRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="analysis_service_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dry_run_writes_expected_csv_outputs_and_progress(self) -> None:
        output_dir = self.tmpdir / "out"
        progress: list[str] = []
        logs: list[str] = []

        result = run_analysis(
            AnalysisConfig(
                input_dir=ROOT / "examples",
                output_dir=output_dir,
                patterns=("gjb18a_strain_example.csv",),
                status_column="status",
                dry_run=True,
            ),
            progress_callback=lambda event: progress.append(event.phase),
            log_callback=logs.append,
        )

        self.assertTrue(result.completed)
        self.assertTrue((output_dir / "gjb_summary.csv").exists())
        self.assertTrue((output_dir / "gjb_fit_data.csv").exists())
        self.assertTrue((output_dir / "gjb_curve.csv").exists())
        self.assertIn("discover", progress)
        self.assertIn("fit", progress)
        self.assertIn("write_outputs", progress)
        self.assertIn("complete", progress)
        self.assertTrue(any("Wrote" in message for message in logs))

    def test_origin_automation_error_surfaces_log_path_and_message(self) -> None:
        output_dir = self.tmpdir / "out"
        log_path = output_dir / "origin_automation.log"
        logs: list[str] = []

        with patch(
            "originnsfitgjb.analysis_service.OriginClient",
            side_effect=OriginAutomationError("boom"),
        ):
            result = run_analysis(
                AnalysisConfig(
                    input_dir=ROOT / "examples",
                    output_dir=output_dir,
                    patterns=("gjb18a_strain_example.csv",),
                    status_column="status",
                    dry_run=False,
                ),
                log_callback=logs.append,
            )

        expected_message = f"Wrote Origin automation log {log_path}"
        self.assertTrue(result.completed)
        self.assertIsNotNone(result.origin_error)
        self.assertIn("boom", result.origin_error)
        self.assertTrue(log_path.exists())
        self.assertIn(log_path, result.output_paths)
        self.assertIn(expected_message, result.messages)
        self.assertIn(expected_message, logs)

    def test_read_failure_records_table_failure_and_continues_other_files(self) -> None:
        input_dir = self.tmpdir / "input"
        output_dir = self.tmpdir / "out"
        input_dir.mkdir()
        shutil.copy(ROOT / "examples" / "gjb18a_strain_example.csv", input_dir / "good.csv")
        (input_dir / "bad.xlsx").write_text("not an excel workbook", encoding="utf-8")
        progress_events: list[object] = []

        result = run_analysis(
            AnalysisConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                patterns=("*.csv", "*.xlsx"),
                status_column="status",
                dry_run=True,
            ),
            progress_callback=progress_events.append,
        )

        fit_progress = [
            (event.current, event.total)
            for event in progress_events
            if getattr(event, "phase", None) == "fit"
        ]
        self.assertTrue(result.completed)
        self.assertEqual(len(result.table_failures), 1)
        self.assertEqual(result.table_failures[0].label, "bad")
        self.assertTrue((output_dir / "gjb_summary.csv").exists())
        self.assertIn((2, 2), fit_progress)


if __name__ == "__main__":
    unittest.main()

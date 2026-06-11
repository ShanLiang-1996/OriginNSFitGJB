from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from originnsfitgjb.analysis_service import AnalysisConfig, run_analysis


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


if __name__ == "__main__":
    unittest.main()

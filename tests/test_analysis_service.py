from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from originnsfitgjb.analysis_service import AnalysisConfig


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


if __name__ == "__main__":
    unittest.main()

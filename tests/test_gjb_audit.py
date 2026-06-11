from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STEP_FILES = [
    "step00_input_checked",
    "step01_initial_ols",
    "step02_initial_nls",
    "step03_variance_analysis",
    "step04_refit_data",
    "step05_refit_result",
    "step06_parameter_significance",
    "step07_fixed_a4_linear_fit",
    "step08_residuals_outliers",
    "step09_final_mle",
    "step10_final_residual_statistics",
    "step11_model_assessment",
    "step12_r2_document_style",
]
REQUIRED_OLD_OUTPUTS = [
    "gjb_summary.csv",
    "gjb_fit_data.csv",
    "gjb_runout_data.csv",
    "gjb_curve.csv",
    "gjb_level_stats.csv",
    "gjb_initialols.csv",
    "gjb_initialnls.csv",
    "gjb_varianceanalysis.csv",
    "gjb_refitdata.csv",
    "gjb_refitresult.csv",
    "gjb_parametersignificance.csv",
    "gjb_fixeda4linearfit.csv",
    "gjb_residuals.csv",
    "gjb_outlieriterations.csv",
    "gjb_finalmle.csv",
    "gjb_likelihood.csv",
    "gjb_modelchecks.csv",
]


class GJBAuditCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="gjb_audit_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dry_run_audit_outputs_and_legacy_csvs(self) -> None:
        output_dir = self.tmpdir / "out"
        self._run_cli(
            "--input",
            str(ROOT / "examples"),
            "--output",
            str(output_dir),
            "--pattern",
            "gjb18a_strain_example.csv",
            "--status",
            "status",
            "--dry-run",
            "--audit",
            "--audit-workbook",
        )
        label_dir = output_dir / "audit" / "tables" / "gjb18a_strain_example"
        json_dir = output_dir / "audit" / "json" / "gjb18a_strain_example"
        self.assertTrue(label_dir.is_dir())
        self.assertTrue(json_dir.is_dir())
        for name in STEP_FILES:
            self.assertTrue((label_dir / f"{name}.csv").exists(), name)
            self.assertTrue((json_dir / f"{name}.json").exists(), name)
        self.assertTrue((output_dir / "audit" / "gjb_decision_log.csv").exists())
        self.assertTrue((output_dir / "audit" / "gjb_decision_log.json").exists())
        self.assertTrue((output_dir / "audit" / "gjb_audit_workbook.xlsx").exists())
        self.assertTrue((output_dir / "audit" / "gjb_manual_checklist.md").exists())
        for name in REQUIRED_OLD_OUTPUTS:
            self.assertTrue((output_dir / name).exists(), name)

        decision_log = pd.read_csv(output_dir / "audit" / "gjb_decision_log.csv")
        steps = set(decision_log["step_id"])
        for step_id in {
            "Step03_VarianceAnalysis",
            "Step06_ParameterSignificance",
            "Step07_FixedA4LinearFit",
            "Step08_ResidualsOutliers",
            "Step09_FinalMLE",
        }:
            self.assertIn(step_id, steps)

        variance = pd.read_csv(label_dir / "step03_variance_analysis.csv")
        np.testing.assert_allclose(variance["weight"], 1.0 / np.power(variance["h"], 2))
        refit = pd.read_csv(label_dir / "step04_refit_data.csv")
        self.assertTrue(refit["gjb18a_runout_treated_as_failure"].any())
        significance = pd.read_csv(label_dir / "step06_parameter_significance.csv")
        a4_row = significance.loc[significance["parameter"] == "A4"].iloc[0]
        self.assertLess(float(a4_row["lower_90"]), 0.0)
        self.assertEqual(str(a4_row["decision"]), "fix A4=0 and refit linearly")
        step07 = pd.read_csv(label_dir / "step07_fixed_a4_linear_fit.csv")
        self.assertTrue(step07["performed"].astype(bool).any())
        self.assertTrue(np.allclose(step07.loc[step07["performed"].astype(bool), "A4_fixed"], 0.0))

    def test_mle_likelihood_records_logpdf_and_logsf(self) -> None:
        input_dir = self.tmpdir / "mle_in"
        output_dir = self.tmpdir / "mle_out"
        input_dir.mkdir()
        self._write_mle_dataset(input_dir / "mle.csv")
        self._run_cli(
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--pattern",
            "mle.csv",
            "--status",
            "status",
            "--dry-run",
            "--audit",
            "--outlier-mode",
            "report-only",
        )
        mle = pd.read_csv(output_dir / "audit" / "tables" / "mle" / "step09_final_mle.csv")
        failure_types = set(mle.loc[mle["gjb_is_failure"].astype(bool), "likelihood_type"])
        included_runout = mle[
            (~mle["gjb_is_failure"].astype(bool)) & mle["included_in_final_mle"].astype(bool)
        ]
        ignored_runout = mle[
            (~mle["gjb_is_failure"].astype(bool)) & ~mle["included_in_final_mle"].astype(bool)
        ]
        self.assertEqual(failure_types, {"logpdf"})
        self.assertEqual(set(included_runout["likelihood_type"]), {"logsf"})
        self.assertEqual(set(ignored_runout["likelihood_type"]), {"ignored_response_le_A4"})
        self.assertTrue((ignored_runout["gjb_response"] <= ignored_runout["A4_final_mle"]).all())
        fit_rows = len(pd.read_csv(output_dir / "gjb_fit_data.csv"))
        self.assertEqual(len(mle), fit_rows)

        refit = pd.read_csv(output_dir / "audit" / "tables" / "mle" / "step04_refit_data.csv")
        refit_excluded_runouts = refit[
            (~refit["gjb_is_failure"].astype(bool)) & (~refit["included_in_refit"].astype(bool))
        ]
        self.assertFalse(refit_excluded_runouts.empty)
        excluded_refit_ids = set(refit_excluded_runouts["gjb_row_id"].astype(int))
        mle_runout_ids = set(mle.loc[mle["likelihood_type"] == "logsf", "gjb_row_id"].astype(int))
        mle_ignored_ids = set(
            mle.loc[mle["likelihood_type"] == "ignored_response_le_A4", "gjb_row_id"].astype(int)
        )
        self.assertTrue(excluded_refit_ids.issubset(mle_runout_ids | mle_ignored_ids))

    def test_a2_not_significant_is_note_only_and_mle_continues(self) -> None:
        input_dir = self.tmpdir / "weak_a2_in"
        output_dir = self.tmpdir / "weak_a2_out"
        input_dir.mkdir()
        self._write_weak_a2_dataset(input_dir / "weak.csv")
        self._run_cli(
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--pattern",
            "weak.csv",
            "--status",
            "status",
            "--dry-run",
            "--audit",
            "--outlier-mode",
            "report-only",
        )
        significance = pd.read_csv(
            output_dir / "audit" / "tables" / "weak" / "step06_parameter_significance.csv"
        )
        a2_row = significance.loc[significance["parameter"] == "A2"].iloc[0]
        self.assertFalse(str(a2_row["passed"]).strip().lower() == "true")
        self.assertFalse(str(a2_row["affects_workflow"]).strip().lower() == "true")
        self.assertIn("workflow continues", str(a2_row["decision"]))
        self.assertTrue((output_dir / "audit" / "tables" / "weak" / "step09_final_mle.csv").exists())
        mle = pd.read_csv(output_dir / "audit" / "tables" / "weak" / "step09_final_mle.csv")
        self.assertTrue(mle["included_in_final_mle"].astype(bool).any())

    def test_weighted_step07_executes_and_unweighted_skips(self) -> None:
        weighted_in = self.tmpdir / "weighted_in"
        weighted_out = self.tmpdir / "weighted_out"
        weighted_in.mkdir()
        self._write_weighted_dataset(weighted_in / "w.csv")
        self._run_cli(
            "--input",
            str(weighted_in),
            "--output",
            str(weighted_out),
            "--pattern",
            "w.csv",
            "--status",
            "status",
            "--dry-run",
            "--audit",
        )
        step07 = pd.read_csv(weighted_out / "audit" / "tables" / "w" / "step07_fixed_a4_linear_fit.csv")
        self.assertTrue(step07["performed"].astype(bool).any())

        unweighted_in = self.tmpdir / "unweighted_in"
        unweighted_out = self.tmpdir / "unweighted_out"
        unweighted_in.mkdir()
        self._write_mle_dataset(unweighted_in / "u.csv", include_below_a4=False)
        self._run_cli(
            "--input",
            str(unweighted_in),
            "--output",
            str(unweighted_out),
            "--pattern",
            "u.csv",
            "--status",
            "status",
            "--dry-run",
            "--audit",
        )
        step07_skip = pd.read_csv(
            unweighted_out / "audit" / "tables" / "u" / "step07_fixed_a4_linear_fit.csv"
        )
        self.assertFalse(step07_skip["performed"].astype(bool).any())
        self.assertIn("Skipped", str(step07_skip["reason"].iloc[0]))

    def test_outlier_report_only_does_not_delete_auto_still_deletes(self) -> None:
        input_dir = self.tmpdir / "outlier_in"
        input_dir.mkdir()
        self._write_outlier_dataset(input_dir / "o.csv")

        auto_out = self.tmpdir / "auto_out"
        self._run_cli(
            "--input",
            str(input_dir),
            "--output",
            str(auto_out),
            "--pattern",
            "o.csv",
            "--status",
            "status",
            "--dry-run",
            "--audit",
            "--outlier-mode",
            "auto",
        )
        report_out = self.tmpdir / "report_out"
        self._run_cli(
            "--input",
            str(input_dir),
            "--output",
            str(report_out),
            "--pattern",
            "o.csv",
            "--status",
            "status",
            "--dry-run",
            "--audit",
            "--outlier-mode",
            "report-only",
        )
        auto_rows = len(pd.read_csv(auto_out / "gjb_fit_data.csv"))
        report_rows = len(pd.read_csv(report_out / "gjb_fit_data.csv"))
        self.assertLess(auto_rows, report_rows)
        report_decisions = pd.read_csv(report_out / "audit" / "gjb_decision_log.csv")
        self.assertTrue(report_decisions["decision"].str.contains("reported only").any())

    def _run_cli(self, *args: str) -> None:
        command = [sys.executable, "-m", "originnsfitgjb", *args]
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            self.fail(f"CLI failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    @staticmethod
    def _write_mle_dataset(path: Path, *, include_below_a4: bool = True) -> None:
        a1, a2, a4 = 0.8, -2.1, 0.0015
        responses = [
            0.014,
            0.0125,
            0.011,
            0.0095,
            0.008,
            0.0068,
            0.0058,
            0.0048,
            0.004,
            0.0034,
            0.0029,
            0.0025,
            0.0020,
        ]
        if include_below_a4:
            responses.append(0.0010)
        response = np.array(responses)
        y = np.empty_like(response)
        domain = response > a4
        y[domain] = a1 + a2 * np.log10(response[domain] - a4)
        y[~domain] = 6.0
        status = ["failure"] * len(response)
        status[4] = "runout"
        status[8] = "runout"
        status[12] = "runout"
        if include_below_a4:
            status[13] = "runout"
        pd.DataFrame({"strain": response, "life": np.round(10**y).astype(int), "status": status}).to_csv(
            path,
            index=False,
        )

    @staticmethod
    def _write_weighted_dataset(path: Path) -> None:
        a1, a2, a4 = 0.8, -2.1, 0.0015
        response = np.array(
            [
                0.016,
                0.0145,
                0.013,
                0.0115,
                0.010,
                0.0086,
                0.0074,
                0.0064,
                0.0055,
                0.0047,
                0.0040,
                0.0035,
                0.0031,
                0.0028,
                0.00255,
                0.00235,
            ]
        )
        signs = np.array([1, -1] * 8)
        y = a1 + a2 * np.log10(response - a4) + signs * 0.00002 / response
        status = ["failure"] * len(response)
        status[5] = "runout"
        status[10] = "runout"
        pd.DataFrame({"strain": response, "life": np.round(10**y).astype(int), "status": status}).to_csv(
            path,
            index=False,
        )

    @staticmethod
    def _write_weak_a2_dataset(path: Path) -> None:
        response = np.array(
            [
                0.016,
                0.0145,
                0.013,
                0.0115,
                0.010,
                0.0086,
                0.0074,
                0.0064,
                0.0055,
                0.0047,
            ]
        )
        noise = np.array([0.08, -0.06, 0.05, -0.09, 0.06, -0.04, 0.07, -0.08, 0.04, -0.05])
        y = 4.8 - 0.18 * np.log10(response) + noise
        status = ["failure"] * 8 + ["runout", "runout"]
        pd.DataFrame({"strain": response, "life": np.round(10**y).astype(int), "status": status}).to_csv(
            path,
            index=False,
        )

    @staticmethod
    def _write_outlier_dataset(path: Path) -> None:
        a1, a2, a4 = 0.8, -2.1, 0.0015
        response = np.array(
            [
                0.016,
                0.0145,
                0.013,
                0.0115,
                0.010,
                0.0086,
                0.0074,
                0.0064,
                0.0055,
                0.0047,
                0.0040,
                0.0035,
                0.0031,
                0.0028,
                0.00255,
                0.00235,
            ]
        )
        y = a1 + a2 * np.log10(response - a4)
        y[7] += 0.2
        status = ["failure"] * len(response)
        status[5] = "runout"
        status[10] = "runout"
        pd.DataFrame({"strain": response, "life": np.round(10**y).astype(int), "status": status}).to_csv(
            path,
            index=False,
        )


if __name__ == "__main__":
    unittest.main()

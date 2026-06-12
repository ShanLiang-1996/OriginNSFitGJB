from __future__ import annotations

import unittest
from unittest.mock import patch

from originnsfitgjb.analysis_service import AnalysisConfig, AnalysisRunResult
from originnsfitgjb.gui.worker import AnalysisWorker


class AnalysisWorkerTests(unittest.TestCase):
    def test_run_emits_failed_result_and_log_for_unexpected_exception(self) -> None:
        worker = AnalysisWorker(AnalysisConfig(dry_run=True))
        finished: list[AnalysisRunResult] = []
        logs: list[str] = []
        worker.finished.connect(finished.append)
        worker.log.connect(logs.append)

        with patch("originnsfitgjb.gui.worker.run_analysis", side_effect=RuntimeError("boom")):
            worker.run()

        self.assertEqual(len(finished), 1)
        result = finished[0]
        expected_message = "Unexpected GUI worker failure: boom"
        self.assertIs(result.completed, False)
        self.assertEqual(result.messages, (expected_message,))
        self.assertIn("boom", result.origin_error or "")
        self.assertEqual(logs, [expected_message])

    def test_run_emits_successful_analysis_result_once(self) -> None:
        worker = AnalysisWorker(AnalysisConfig(dry_run=True))
        expected = AnalysisRunResult(completed=True)
        finished: list[AnalysisRunResult] = []
        worker.finished.connect(finished.append)

        with patch("originnsfitgjb.gui.worker.run_analysis", return_value=expected):
            worker.run()

        self.assertEqual(finished, [expected])


if __name__ == "__main__":
    unittest.main()

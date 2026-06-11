from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from originnsfitgjb.origin_client import OriginClient


class OriginCleanupTests(TestCase):
    def test_parse_tasklist_origin_pids_ignores_packaged_exe(self) -> None:
        output = "\n".join(
            [
                '"Origin64.exe","1234","Console","1","100,000 K"',
                '"Origin.exe","5678","Console","1","100,000 K"',
                '"OriginNSFitGJB.exe","9999","Console","1","100,000 K"',
                '"notepad.exe","1111","Console","1","100,000 K"',
            ]
        )
        self.assertEqual(OriginClient._parse_tasklist_origin_pids(output), {1234, 5678})

    def test_started_pid_tracking_keeps_only_new_origin_processes(self) -> None:
        client = object.__new__(OriginClient)
        client._origin_pids_before = {100}
        client._origin_pids_started = set()
        with patch.object(OriginClient, "_origin_process_ids", return_value={100, 200, 300}):
            client._remember_started_origin_processes()
        self.assertEqual(client._origin_pids_started, {200, 300})

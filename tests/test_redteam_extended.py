"""Tests for extended active red-team probes (container integrity, MCP security, and anti-spoofing)."""
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._helpers import patched_dpapi
from voice_eval.dims.redteam import evaluate_redteam


class TestRedteamExtended(unittest.TestCase):
    def test_evaluate_redteam_includes_container_mcp_and_spoof_probes_all_green(self):
        with patched_dpapi():
            res = evaluate_redteam()
        self.assertTrue(res['passed'], f'redteam failed: {res}')
        self.assertEqual(res['yellow'], 0)
        self.assertEqual(res['red'], 0)

        by_id = {c['id']: c for c in res['checks']}
        expected_ids = [
            'redteam_container_magic_mismatch',
            'redteam_container_truncated_wav',
            'redteam_quality_pure_silence',
            'redteam_quality_severe_clipping',
            'redteam_mcp_no_roots_blocks_path',
            'redteam_mcp_path_escape_blocked',
            'redteam_mcp_base64_data_uri_and_magic',
            'redteam_mcp_revoke_requires_two_step_and_readonly',
            'redteam_spoof_watermarked_clone_blocked',
            'redteam_spoof_clean_genuine_preserved',
        ]
        for cid in expected_ids:
            self.assertIn(cid, by_id, f'missing expected redteam check {cid}')
            self.assertEqual(by_id[cid]['status'], 'GREEN', f'check {cid} not GREEN: {by_id[cid]}')

        self.assertEqual(
            res.get('spoof_summary'),
            {'watermarked_clone_far': 0.0, 'clean_genuine_frr': 0.0},
        )

    def test_symlink_probe_covers_inroot_and_keeps_single_skip_on_permission_error(self):
        with patched_dpapi(), patch.object(Path, 'symlink_to', side_effect=PermissionError('privilege not held')):
            res = evaluate_redteam()
        self.assertTrue(res['passed'])
        self.assertEqual(res['skipped'], 1)
        by_id = {c['id']: c for c in res['checks']}
        self.assertEqual(by_id['path_symlink_escape']['status'], 'SKIPPED_NO_PRIVILEGE')


if __name__ == '__main__':
    unittest.main()

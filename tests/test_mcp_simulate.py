"""Tests for dynamic voiceprint_simulate MCP tool exposure, type validation and loopback isolation."""
import base64
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent import mcp_server
from agent.mcp_server import build_default_server
from tests._helpers import patched_dpapi
from voice_eval.dims._sandbox import (
    ToneEmbedder,
    generate_speaker_wav,
    sandbox_cohort,
    sandbox_enroll_takes,
)
from voice_eval.dims.response import FORBIDDEN_NON_DETAIL_KEYS, FORBIDDEN_NON_DETAIL_TERMS
from voiceprint.audio import sniff_audio_format
from voiceprint.synth import AcousticFormantSynthesizer


def _call_tool(server, name: str, arguments: dict | None = None, req_id: int = 1) -> dict:
    resp = server.handle_request({
        'jsonrpc': '2.0',
        'id': req_id,
        'method': 'tools/call',
        'params': {'name': name, 'arguments': arguments or {}},
    })
    assert resp is not None
    return resp


def _parse_payload(resp: dict) -> dict:
    return json.loads(resp['result']['content'][0]['text'])


def _wav_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode('ascii')


class TestMCPSimulate(unittest.TestCase):
    def test_mcp_simulate_hidden_by_default_in_build_default_server_and_exposed_when_enabled(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcph_') as tmp:
            root = Path(tmp)
            srv_default = build_default_server(root / 'db1', embedder)
            try:
                tools_resp = srv_default.handle_request({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list', 'params': {}})
                names = {t['name'] for t in tools_resp['result']['tools']}
                self.assertEqual(len(names), 8)
                self.assertNotIn('voiceprint_simulate', names)
                res_health = srv_default.handle_request({
                    'jsonrpc': '2.0', 'id': 2, 'method': 'resources/read', 'params': {'uri': 'voiceprint://health'}
                })
                health = json.loads(res_health['result']['contents'][0]['text'])
                self.assertFalse(health['simulation_enabled'])
                self.assertIsNone(health['synthesizer'])
            finally:
                srv_default.agent.close()

            srv_enabled = build_default_server(root / 'db2', embedder, synthesizer=AcousticFormantSynthesizer())
            try:
                tools_resp2 = srv_enabled.handle_request({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/list', 'params': {}})
                names2 = {t['name'] for t in tools_resp2['result']['tools']}
                self.assertEqual(len(names2), 9)
                self.assertIn('voiceprint_simulate', names2)
                health2 = _parse_payload(_call_tool(srv_enabled, 'voiceprint_health'))
                self.assertTrue(health2['simulation_enabled'])
                self.assertEqual(health2['synthesizer']['backend'], 'acoustic_formant')
            finally:
                srv_enabled.agent.close()

    def test_mcp_simulate_disabled_when_no_synthesizer_and_rejects_bool_max_attempts(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcpdis_') as tmp:
            root = Path(tmp)
            takes = sandbox_enroll_takes(root / 'audio', 'jia', speaker_index=0)
            b64_takes = [_wav_to_b64(p) for p in takes]

            srv_off = build_default_server(root / 'db1', embedder, synthesizer=None)
            try:
                resp = _call_tool(srv_off, 'voiceprint_simulate', {
                    'name': '甲',
                    'consent': True,
                    'reference_audio_base64': b64_takes[0],
                    'text': '测试禁用态',
                })
                self.assertTrue(resp['result'].get('isError'))
                self.assertEqual(_parse_payload(resp)['error'], 'synthesizer_disabled')
            finally:
                srv_off.agent.close()

            srv_on = build_default_server(root / 'db2', embedder, synthesizer=AcousticFormantSynthesizer())
            try:
                enr = _call_tool(srv_on, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_base64_list': b64_takes,
                    'session_ids': ['s1', 's1', 's2'],
                })
                self.assertFalse(enr['result'].get('isError', False), enr)

                r_str_consent = _call_tool(srv_on, 'voiceprint_simulate', {
                    'name': '甲',
                    'consent': 'true',
                    'reference_audio_base64': b64_takes[0],
                    'text': '测试字符串布尔',
                })
                self.assertTrue(r_str_consent['result'].get('isError'))
                self.assertEqual(_parse_payload(r_str_consent)['error'], 'invalid_arguments')

                r_false_consent = _call_tool(srv_on, 'voiceprint_simulate', {
                    'name': '甲',
                    'consent': False,
                    'reference_audio_base64': b64_takes[0],
                    'text': '测试未授权',
                })
                self.assertTrue(r_false_consent['result'].get('isError'))
                self.assertEqual(_parse_payload(r_false_consent)['error'], 'consent_required')

                for bad_attempts in (True, False, 0, 6, '3'):
                    r_bad_att = _call_tool(srv_on, 'voiceprint_simulate', {
                        'name': '甲',
                        'consent': True,
                        'reference_audio_base64': b64_takes[0],
                        'text': '测试非法次数',
                        'max_attempts': bad_attempts,
                    })
                    self.assertTrue(r_bad_att['result'].get('isError'), f'failed to reject max_attempts={bad_attempts!r}')
                    self.assertEqual(_parse_payload(r_bad_att)['error'], 'invalid_arguments')
            finally:
                srv_on.agent.close()

    def test_mcp_simulate_readonly_blocked_and_forbids_arbitrary_output_path(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcpro_') as tmp:
            root = Path(tmp)
            allowed = root / 'allowed'
            allowed.mkdir()
            takes = sandbox_enroll_takes(allowed, 'jia', speaker_index=0)
            src = generate_speaker_wav(allowed / 'src.wav', speaker_index=1, duration=2.5, seed=7)

            ro_srv = build_default_server(
                root / 'db_ro',
                embedder,
                audio_roots=[allowed],
                read_only=True,
                synthesizer=AcousticFormantSynthesizer(),
            )
            try:
                ro_resp = _call_tool(ro_srv, 'voiceprint_simulate', {
                    'name': '甲',
                    'consent': True,
                    'reference_audio_path': str(takes[0]),
                    'source_audio_path': str(src),
                })
                self.assertTrue(ro_resp['result'].get('isError'))
                self.assertEqual(_parse_payload(ro_resp)['error'], 'read_only_mode')
            finally:
                ro_srv.agent.close()

            rw_srv = build_default_server(
                root / 'db_rw',
                embedder,
                audio_roots=[allowed],
                synthesizer=AcousticFormantSynthesizer(),
            )
            try:
                _call_tool(rw_srv, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_paths': [str(p) for p in takes],
                })
                # Mixed source_audio_path + text must fail
                mixed = _call_tool(rw_srv, 'voiceprint_simulate', {
                    'name': '甲',
                    'consent': True,
                    'reference_audio_path': str(takes[0]),
                    'source_audio_path': str(src),
                    'text': '不允许同时传',
                })
                self.assertTrue(mixed['result'].get('isError'))
                self.assertEqual(_parse_payload(mixed)['error'], 'invalid_arguments')

                # Even if caller smuggles an unexpected output_path arg, file stays inside simulated_dir
                evil_target = root / 'pwned.wav'
                sim_ok = _call_tool(rw_srv, 'voiceprint_simulate', {
                    'name': '甲',
                    'consent': True,
                    'reference_audio_path': str(takes[0]),
                    'source_audio_path': str(src),
                    'output_path': str(evil_target),
                })
                self.assertFalse(sim_ok['result'].get('isError', False), sim_ok)
                self.assertFalse(evil_target.exists())
                out_path = Path(_parse_payload(sim_ok)['result']['output_path']).resolve()
                self.assertEqual(out_path.parent, rw_srv.agent.simulated_dir.resolve())
            finally:
                rw_srv.agent.close()

    def test_mcp_simulate_no_roots_base64_roundtrip_and_with_roots_path_loopback(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcplb_') as tmp:
            root = Path(tmp)
            allowed = root / 'allowed'
            allowed.mkdir()
            takes = sandbox_enroll_takes(allowed, 'jia', speaker_index=0)
            src = generate_speaker_wav(allowed / 'src.wav', speaker_index=1, duration=2.5, seed=9)
            b64_takes = [_wav_to_b64(p) for p in takes]

            # Case 1: no audio_roots -> path args blocked; base64 + include_audio_base64 works and loops back to verify spoof block
            srv_no_roots = build_default_server(
                root / 'db_b64',
                embedder,
                audio_roots=None,
                detail=False,
                synthesizer=AcousticFormantSynthesizer(),
            )
            try:
                enr_b64 = _call_tool(srv_no_roots, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_base64_list': b64_takes,
                    'session_ids': ['s1', 's1', 's2'],
                })
                self.assertFalse(enr_b64['result'].get('isError', False), enr_b64)
                blocked_path = _call_tool(srv_no_roots, 'voiceprint_simulate', {
                    'name': '甲',
                    'consent': True,
                    'reference_audio_path': str(takes[0]),
                    'text': '路径被拦截测试',
                })
                self.assertTrue(blocked_path['result'].get('isError'))
                self.assertEqual(_parse_payload(blocked_path)['error'], 'path_not_allowed')

                sim_b64 = _call_tool(srv_no_roots, 'voiceprint_simulate', {
                    'name': '甲',
                    'consent': True,
                    'reference_audio_base64': b64_takes[0],
                    'text': '请验证我的声纹水印模拟闭环。',
                    'reference_text': '参考文本样本',
                    'include_audio_base64': True,
                })
                self.assertFalse(sim_b64['result'].get('isError', False), sim_b64)
                payload = _parse_payload(sim_b64)
                res = payload['result']
                for k in FORBIDDEN_NON_DETAIL_KEYS:
                    self.assertNotIn(k, res)
                for term in FORBIDDEN_NON_DETAIL_TERMS:
                    self.assertNotIn(term, payload['reply'])
                raw_out = base64.b64decode(res['audio_base64'])
                self.assertEqual(sniff_audio_format(raw_out), '.wav')

                ver_b64 = _call_tool(srv_no_roots, 'voiceprint_verify', {
                    'name': '甲',
                    'audio_base64': res['audio_base64'],
                })
                self.assertTrue(ver_b64['result'].get('isError'))
                self.assertEqual(_parse_payload(ver_b64)['result']['reason'], 'spoof_detected')
            finally:
                srv_no_roots.agent.close()

            # Case 2: with audio_roots -> output_path passes _ephemeral_roots directly into voiceprint_verify
            srv_roots = build_default_server(
                root / 'db_roots',
                embedder,
                audio_roots=[allowed],
                detail=False,
                synthesizer=AcousticFormantSynthesizer(),
            )
            try:
                _call_tool(srv_roots, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_paths': [str(p) for p in takes],
                })
                sim_p = _call_tool(srv_roots, 'voiceprint_simulate', {
                    'name': '甲',
                    'consent': True,
                    'reference_audio_path': str(takes[0]),
                    'source_audio_path': str(src),
                })
                self.assertFalse(sim_p['result'].get('isError', False), sim_p)
                sim_out_path = _parse_payload(sim_p)['result']['output_path']

                ver_p = _call_tool(srv_roots, 'voiceprint_verify', {
                    'name': '甲',
                    'audio_path': sim_out_path,
                })
                self.assertTrue(ver_p['result'].get('isError'))
                self.assertEqual(_parse_payload(ver_p)['result']['reason'], 'spoof_detected')
            finally:
                srv_roots.agent.close()

    def test_mcp_cli_defaults_to_acoustic_synthesizer(self):
        parser = mcp_server.build_arg_parser()
        args = parser.parse_args(['--data-dir', '/tmp/db'])
        self.assertEqual(args.synthesizer, 'acoustic')
        self.assertEqual(args.max_sim_attempts, 3)

        args_off = parser.parse_args(['--data-dir', '/tmp/db', '--synthesizer', 'none'])
        self.assertEqual(args_off.synthesizer, 'none')

    def test_mcp_simulate_clears_pending_revoke_and_catches_missing_neural_cmd(self):
        from voiceprint.synth import SubprocessNeuralSynthesizer

        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcp_fix_') as tmp:
            root = Path(tmp)
            allowed = root / 'allowed'
            allowed.mkdir()
            takes = sandbox_enroll_takes(allowed, 'jia', speaker_index=0)
            src = generate_speaker_wav(allowed / 'src.wav', speaker_index=1, duration=2.0, seed=19)

            srv = build_default_server(
                root / 'db_pending',
                embedder,
                audio_roots=[allowed],
                synthesizer=AcousticFormantSynthesizer(),
            )
            try:
                _call_tool(srv, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_paths': [str(p) for p in takes],
                })
                ask_rev = _call_tool(srv, 'voiceprint_revoke', {'name': '甲', 'confirm': False})
                self.assertFalse(ask_rev['result'].get('isError', False), ask_rev)

                sim = _call_tool(srv, 'voiceprint_simulate', {
                    'name': '甲',
                    'consent': True,
                    'reference_audio_path': str(takes[0]),
                    'source_audio_path': str(src),
                })
                self.assertFalse(sim['result'].get('isError', False), sim)

                # Intermediate voiceprint_simulate must invalidate pending two-step revoke
                conf_rev = _call_tool(srv, 'voiceprint_revoke', {'name': '甲', 'confirm': True})
                self.assertTrue(conf_rev['result'].get('isError'))
                self.assertTrue(srv.agent.store.exists('甲'))
            finally:
                srv.agent.close()

            srv_bad = build_default_server(
                root / 'db_missing_bin',
                embedder,
                audio_roots=[allowed],
                synthesizer=SubprocessNeuralSynthesizer(['/nonexistent/voiceprint_missing_bin/python'], timeout_s=5.0),
            )
            try:
                _call_tool(srv_bad, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_paths': [str(p) for p in takes],
                })
                before_audits = srv_bad.agent.store.db.execute(
                    "select count(*) from audit where action='simulate'"
                ).fetchone()[0]
                resp = srv_bad.handle_request({
                    'jsonrpc': '2.0',
                    'id': 99,
                    'method': 'tools/call',
                    'params': {
                        'name': 'voiceprint_simulate',
                        'arguments': {
                            'name': '甲',
                            'consent': True,
                            'reference_audio_path': str(takes[0]),
                            'text': '缺失可执行文件测试',
                        },
                    },
                })
                self.assertIsNotNone(resp)
                self.assertIn('result', resp)
                self.assertTrue(resp['result'].get('isError'))
                payload = _parse_payload(resp)
                self.assertFalse(payload['ok'])
                self.assertEqual(payload['result']['reason'], 'synthesis_failed')
                after_audits = srv_bad.agent.store.db.execute(
                    "select count(*) from audit where action='simulate'"
                ).fetchone()[0]
                self.assertEqual(after_audits, before_audits + 1)
            finally:
                srv_bad.agent.close()


if __name__ == '__main__':
    unittest.main()

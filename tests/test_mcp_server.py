"""Tests for zero-dependency JSON-RPC 2.0 stdio MCP Server."""
import base64
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

from agent.mcp_server import MAX_BASE64_BYTES, VoiceprintMCPServer, build_default_server
from tests._helpers import patched_dpapi
from voice_eval.audio_synth import apply_watermark_spoof
from voice_eval.dims._sandbox import (
    ToneEmbedder,
    generate_speaker_wav,
    sandbox_cohort,
    sandbox_enroll_takes,
)
from voice_eval.dims.response import FORBIDDEN_NON_DETAIL_KEYS, FORBIDDEN_NON_DETAIL_TERMS


def _call_tool(server: VoiceprintMCPServer, name: str, arguments: dict | None = None, req_id: int = 1) -> dict:
    resp = server.handle_request({
        'jsonrpc': '2.0',
        'id': req_id,
        'method': 'tools/call',
        'params': {'name': name, 'arguments': arguments or {}},
    })
    assert resp is not None
    return resp


def _parse_tool_payload(resp: dict) -> dict:
    content = resp['result']['content']
    return json.loads(content[0]['text'])


class TestMCPServer(unittest.TestCase):
    def test_mcp_initialize_tools_list_and_resources(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcp_init_') as tmp:
            server = build_default_server(Path(tmp) / 'db', embedder)
            try:
                init = server.handle_request({
                    'jsonrpc': '2.0',
                    'id': 1,
                    'method': 'initialize',
                    'params': {'protocolVersion': '2024-11-05', 'capabilities': {}, 'clientInfo': {'name': 'test'}},
                })
                self.assertEqual(init['result']['protocolVersion'], '2024-11-05')
                self.assertIn('tools', init['result']['capabilities'])
                self.assertIn('resources', init['result']['capabilities'])

                tools_resp = server.handle_request({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}})
                names = {t['name'] for t in tools_resp['result']['tools']}
                self.assertEqual(names, {
                    'voiceprint_enroll',
                    'voiceprint_verify',
                    'voiceprint_identify',
                    'voiceprint_list_speakers',
                    'voiceprint_revoke',
                    'voiceprint_rotate_key',
                    'voiceprint_check_audio_quality',
                    'voiceprint_health',
                })

                res_list = server.handle_request({'jsonrpc': '2.0', 'id': 3, 'method': 'resources/list', 'params': {}})
                uris = {r['uri'] for r in res_list['result']['resources']}
                self.assertEqual(uris, {'voiceprint://speakers', 'voiceprint://health'})

                res_health = server.handle_request({
                    'jsonrpc': '2.0',
                    'id': 4,
                    'method': 'resources/read',
                    'params': {'uri': 'voiceprint://health'},
                })
                health = json.loads(res_health['result']['contents'][0]['text'])
                self.assertTrue(health['spoof_guard_enabled'])
                self.assertFalse(health['detail'])
                self.assertFalse(health['audio_roots_configured'])
            finally:
                server.agent.close()

    def test_mcp_enforces_audio_roots_boundary_and_blocks_escape(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcp_roots_') as tmp:
            root = Path(tmp)
            allowed = root / 'allowed'
            outside = root / 'outside'
            allowed.mkdir()
            outside.mkdir()
            out_wav = generate_speaker_wav(outside / 'out.wav', speaker_index=0, duration=2.0, seed=1)

            # No audio_roots -> ALL path inputs refused with path_not_allowed
            srv_no_roots = build_default_server(root / 'db1', embedder, audio_roots=None)
            try:
                resp = _call_tool(srv_no_roots, 'voiceprint_check_audio_quality', {'audio_path': str(out_wav)})
                self.assertTrue(resp['result'].get('isError'))
                payload = _parse_tool_payload(resp)
                self.assertEqual(payload['error'], 'path_not_allowed')
            finally:
                srv_no_roots.agent.close()

            # With audio_roots=[allowed] -> .. escape outside allowed is refused
            srv_roots = build_default_server(root / 'db2', embedder, audio_roots=[allowed])
            try:
                esc_path = str(allowed / '..' / 'outside' / 'out.wav')
                resp = _call_tool(srv_roots, 'voiceprint_check_audio_quality', {'audio_path': esc_path})
                self.assertTrue(resp['result'].get('isError'))
                payload = _parse_tool_payload(resp)
                self.assertIn('允许的目录', payload['reply'])
            finally:
                srv_roots.agent.close()

    def test_mcp_base64_data_uri_magic_sniffing_and_session_ids(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcp_b64_') as tmp:
            root = Path(tmp)
            wavs = [
                generate_speaker_wav(root / f'seg_{i}.wav', speaker_index=0, duration=3.0, phase=i * 0.8, seed=i)
                for i in range(3)
            ]
            b64_list = [
                'data:audio/wav;base64,' + base64.b64encode(p.read_bytes()).decode('ascii')
                for p in wavs
            ]

            server = build_default_server(root / 'db', embedder, audio_roots=None)
            try:
                # Invalid base64 refused
                bad_b64 = _call_tool(server, 'voiceprint_check_audio_quality', {'audio_base64': '!!!not-b64!!!'})
                self.assertTrue(bad_b64['result'].get('isError'))

                # Non-audio bytes in valid base64 refused by magic sniffing
                png_b64 = base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\x00' * 128).decode('ascii')
                bad_magic = _call_tool(server, 'voiceprint_check_audio_quality', {'audio_base64': png_b64})
                self.assertTrue(bad_magic['result'].get('isError'))

                # Oversized base64 rejected before decode
                huge_b64 = 'A' * (((MAX_BASE64_BYTES + 1024) * 4) // 3 + 16)
                too_big = _call_tool(server, 'voiceprint_check_audio_quality', {'audio_base64': huge_b64})
                self.assertTrue(too_big['result'].get('isError'))

                # Valid data: URI quality check works
                q_ok = _call_tool(server, 'voiceprint_check_audio_quality', {'audio_base64': b64_list[0]})
                self.assertFalse(q_ok['result'].get('isError', False))
                self.assertTrue(_parse_tool_payload(q_ok)['result']['passed_enroll_quality'])

                # 3 short (3s) base64 clips WITHOUT session_ids -> single temp dir (< 30s) -> REJECTED!
                enr_single = _call_tool(server, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_base64_list': b64_list,
                })
                self.assertTrue(enr_single['result'].get('isError'))
                self.assertIn('30 秒', _parse_tool_payload(enr_single)['reply'])

                # Invalid session_ids (path traversal or mismatched length) -> REJECTED
                enr_bad_sid = _call_tool(server, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_base64_list': b64_list,
                    'session_ids': ['../escape', 's2', 's2'],
                })
                self.assertTrue(enr_bad_sid['result'].get('isError'))

                enr_mismatch = _call_tool(server, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_base64_list': b64_list,
                    'session_ids': ['s1', 's2'],
                })
                self.assertTrue(enr_mismatch['result'].get('isError'))

                # Valid explicit session_ids across 2 sessions -> SUCCEEDS
                enr_ok = _call_tool(server, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_base64_list': b64_list,
                    'session_ids': ['sess_a', 'sess_b', 'sess_b'],
                })
                self.assertFalse(enr_ok['result'].get('isError', False))
                self.assertTrue(_parse_tool_payload(enr_ok)['ok'])
            finally:
                server.agent.close()

    def test_mcp_two_step_revoke_read_only_and_default_spoof_guard(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcp_sec_') as tmp:
            root = Path(tmp)
            allowed = root / 'allowed'
            allowed.mkdir()
            takes = sandbox_enroll_takes(allowed, 'jia', speaker_index=0)
            clean_probe = generate_speaker_wav(allowed / 'clean.wav', speaker_index=0, duration=3.0, phase=1.7, seed=5)
            wav, sr = sf.read(str(clean_probe), dtype='float32')
            wm_probe = allowed / 'wm.wav'
            sf.write(str(wm_probe), apply_watermark_spoof(wav, sr), sr)

            server = build_default_server(root / 'db', embedder, audio_roots=[allowed])
            try:
                enr = _call_tool(server, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_paths': [str(p) for p in takes],
                })
                self.assertFalse(enr['result'].get('isError', False))

                # Watermarked clone blocked by default spoof guard, with zero score leaks in detail=False
                v_wm = _call_tool(server, 'voiceprint_verify', {'name': '甲', 'audio_path': str(wm_probe)})
                wm_payload = _parse_tool_payload(v_wm)
                self.assertFalse(wm_payload['ok'])
                self.assertEqual(wm_payload['result'].get('reason'), 'spoof_detected')
                for k in FORBIDDEN_NON_DETAIL_KEYS | {'spoof_score'}:
                    self.assertNotIn(k, wm_payload['result'])

                # Clean genuine accepted with zero score leaks in detail=False
                v_clean = _call_tool(server, 'voiceprint_verify', {'name': '甲', 'audio_path': str(clean_probe)})
                clean_payload = _parse_tool_payload(v_clean)
                self.assertTrue(clean_payload['ok'])
                self.assertEqual(clean_payload['result'].get('decision'), 'accept')
                for k in FORBIDDEN_NON_DETAIL_KEYS:
                    self.assertNotIn(k, clean_payload['result'])
                for term in FORBIDDEN_NON_DETAIL_TERMS:
                    self.assertNotIn(term, clean_payload['reply'])

                # One-shot confirm=True revoke MUST FAIL (enforces 2-step state machine)
                rev_skip = _call_tool(server, 'voiceprint_revoke', {'name': '甲', 'confirm': True})
                self.assertTrue(rev_skip['result'].get('isError'))
                self.assertIn('甲', server.agent.store.names())

                # Two-step confirm=False -> confirm=True succeeds
                rev1 = _call_tool(server, 'voiceprint_revoke', {'name': '甲', 'confirm': False})
                self.assertFalse(rev1['result'].get('isError', False))
                rev2 = _call_tool(server, 'voiceprint_revoke', {'name': '甲', 'confirm': True})
                self.assertFalse(rev2['result'].get('isError', False))
                self.assertNotIn('甲', server.agent.store.names())
            finally:
                server.agent.close()

            # Read-only mode blocks write/destructive tools but allows read/verify tools
            ro_srv = build_default_server(root / 'db_ro', embedder, audio_roots=[allowed], read_only=True)
            try:
                for tool_name, args in (
                    ('voiceprint_enroll', {'name': '乙', 'consent': True, 'audio_paths': [str(p) for p in takes]}),
                    ('voiceprint_revoke', {'name': '乙', 'confirm': False}),
                    ('voiceprint_rotate_key', {'name': '乙'}),
                ):
                    r = _call_tool(ro_srv, tool_name, args)
                    self.assertTrue(r['result'].get('isError'))
                    self.assertEqual(_parse_tool_payload(r)['error'], 'read_only_mode')
                lst = _call_tool(ro_srv, 'voiceprint_list_speakers', {})
                self.assertFalse(lst['result'].get('isError', False))
            finally:
                ro_srv.agent.close()

    def test_mcp_stdio_stream_roundtrip_no_stdout_pollution(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcp_io_') as tmp:
            server = build_default_server(Path(tmp) / 'db', embedder)
            try:
                requests = [
                    {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2024-11-05'}},
                    {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
                    {'jsonrpc': '2.0', 'id': 2, 'method': 'ping'},
                    {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'voiceprint_health', 'arguments': {}}},
                ]
                in_buf = io.StringIO('\n'.join(json.dumps(r) for r in requests) + '\n')
                out_buf = io.StringIO()
                server.serve_streams(in_buf, out_buf)
                lines = [ln for ln in out_buf.getvalue().splitlines() if ln.strip()]
                # Notification produces no response -> 3 JSON-RPC responses
                self.assertEqual(len(lines), 3)
                parsed = [json.loads(ln) for ln in lines]
                self.assertEqual([p['id'] for p in parsed], [1, 2, 3])
            finally:
                server.agent.close()

    def test_direct_mcp_server_instantiation_enforces_audio_roots(self):
        from agent.service import VoiceAgent

        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcp_dir_') as tmp:
            root = Path(tmp)
            allowed = root / 'allowed'
            outside = root / 'outside'
            allowed.mkdir()
            outside.mkdir()
            takes = sandbox_enroll_takes(outside, 'out_jia', speaker_index=0)
            out_wav = takes[0]

            # VoiceAgent constructed with audio_roots=None, wrapped by VoiceprintMCPServer(audio_roots=[allowed])
            agent = VoiceAgent(root / 'db', embedder, detail=False, audio_roots=None)
            server = VoiceprintMCPServer(agent, audio_roots=[allowed])
            try:
                q_resp = _call_tool(server, 'voiceprint_check_audio_quality', {'audio_path': str(out_wav)})
                self.assertTrue(q_resp['result'].get('isError'))
                self.assertIn('允许的目录', _parse_tool_payload(q_resp)['reply'])

                e_resp = _call_tool(server, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_paths': [str(p) for p in takes],
                })
                self.assertTrue(e_resp['result'].get('isError'))
                self.assertIn('允许的目录', _parse_tool_payload(e_resp)['reply'])
                self.assertEqual(agent.store.names(), [])
            finally:
                agent.close()

    def test_mcp_rejects_non_bool_string_consent_and_confirm(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcp_bool_') as tmp:
            root = Path(tmp)
            allowed = root / 'allowed'
            allowed.mkdir()
            takes = sandbox_enroll_takes(allowed, 'jia', speaker_index=0)
            server = build_default_server(root / 'db', embedder, audio_roots=[allowed])
            try:
                # String "false" for consent MUST NOT coerce to True
                enr_str = _call_tool(server, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': 'false',
                    'audio_paths': [str(p) for p in takes],
                })
                self.assertTrue(enr_str['result'].get('isError'))
                self.assertEqual(_parse_tool_payload(enr_str)['error'], 'invalid_arguments')
                self.assertNotIn('甲', server.agent.store.names())

                # Enroll legitimately, enter pending revoke, then pass string "false" for confirm
                enr_ok = _call_tool(server, 'voiceprint_enroll', {
                    'name': '甲',
                    'consent': True,
                    'audio_paths': [str(p) for p in takes],
                })
                self.assertFalse(enr_ok['result'].get('isError', False))
                rev1 = _call_tool(server, 'voiceprint_revoke', {'name': '甲', 'confirm': False})
                self.assertFalse(rev1['result'].get('isError', False))

                rev_str = _call_tool(server, 'voiceprint_revoke', {'name': '甲', 'confirm': 'false'})
                self.assertTrue(rev_str['result'].get('isError'))
                self.assertEqual(_parse_tool_payload(rev_str)['error'], 'invalid_arguments')
                self.assertIn('甲', server.agent.store.names())
            finally:
                server.agent.close()

    def test_mcp_non_dict_params_returns_jsonrpc_error_without_crashing_stream(self):
        embedder = ToneEmbedder()
        with patched_dpapi(), sandbox_cohort(embedder), TemporaryDirectory(prefix='vpmcp_mal_') as tmp:
            server = build_default_server(Path(tmp) / 'db', embedder)
            try:
                requests = [
                    {'jsonrpc': '1.0', 'id': 9, 'method': 'ping'},
                    {'jsonrpc': '2.0', 'id': 10, 'method': 'tools/call', 'params': ['list-not-dict']},
                    {'jsonrpc': '2.0', 'id': 11, 'method': 'resources/read', 'params': 123},
                    {'jsonrpc': '2.0', 'id': 12, 'method': 'ping'},
                ]
                in_buf = io.StringIO('\n'.join(json.dumps(r) for r in requests) + '\n')
                out_buf = io.StringIO()
                server.serve_streams(in_buf, out_buf)
                parsed = [json.loads(ln) for ln in out_buf.getvalue().splitlines() if ln.strip()]
                self.assertEqual(len(parsed), 4)
                self.assertEqual(parsed[0]['id'], 9)
                self.assertEqual(parsed[0]['error']['code'], -32600)
                self.assertEqual(parsed[1]['id'], 10)
                self.assertEqual(parsed[1]['error']['code'], -32602)
                self.assertEqual(parsed[2]['id'], 11)
                self.assertEqual(parsed[2]['error']['code'], -32602)
                self.assertEqual(parsed[3]['id'], 12)
                self.assertIn('result', parsed[3])
            finally:
                server.agent.close()


if __name__ == '__main__':
    unittest.main()

"""Positive and negative defect-catching tests for tooling, response, and flow dimensions."""
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.service import VoiceAgent
from tests._helpers import patched_dpapi
from voice_eval.dims._sandbox import generate_speaker_wav
from voice_eval.dims.flow import evaluate_flow
from voice_eval.dims.response import check_reply_payload, evaluate_response
from voice_eval.dims.tooling import evaluate_tooling


class AgentDimensionChecksTests(unittest.TestCase):
    def test_evaluate_tooling_passes_100_percent(self):
        with patched_dpapi():
            res = evaluate_tooling()
        self.assertTrue(res['passed'], res)
        self.assertEqual(res['accuracy'], 1.0)
        self.assertEqual(res['passed_count'], res['total'])
        self.assertGreaterEqual(res['total'], 20)
        ids = {c['id'] for c in res['checks']}
        for required in (
            'enroll_synonym',
            'verify_synonym',
            'identify_synonym',
            'anonymize_boundary',
            'missing_consent',
            'outside_audio_roots',
            'too_short_audio',
            'unclosed_quote',
        ):
            self.assertIn(required, ids)

    def test_evaluate_tooling_catches_injected_agent_routing_bug(self):
        orig_handle = VoiceAgent.handle

        def broken_handle(self, text: str) -> dict:
            out = orig_handle(self, text)
            if text.strip().startswith('匿名'):
                out = dict(out)
                out['ok'] = True
            return out

        with patched_dpapi(), patch.object(VoiceAgent, 'handle', broken_handle):
            res = evaluate_tooling()
        self.assertFalse(res['passed'])
        self.assertLess(res['accuracy'], 1.0)
        failed = [c for c in res['checks'] if not c['passed']]
        self.assertTrue(any('anonymize' in c['id'] for c in failed), failed)

    def test_evaluate_response_passes_and_supports_numeric_speaker_names(self):
        real_checks = [
            {
                'label': 'real-verify-1089',
                'outcome': {'tool': 'verify', 'ok': True, 'result': {'decision': 'accept', 'task': 'verify'}, 'reply': '1089：结论：接受。'},
                'expect_decision': 'accept',
                'speaker_name': '1089',
            },
            {
                'label': 'real-identify-unknown-1089',
                'outcome': {
                    'tool': 'identify',
                    'ok': True,
                    'result': {'decision': 'unknown', 'candidate': '1089'},
                    'reply': '没有足够把握的匹配。最接近的是 1089，未达到辨认阈值。',
                },
                'expect_decision': 'unknown',
                'speaker_name': None,
            },
            {
                'label': 'real-identify-split-winner-1089-1272',
                'outcome': {
                    'tool': 'identify',
                    'ok': True,
                    'result': {'decision': 'uncertain', 'candidate': '1089'},
                    'reply': '没有足够把握。1089 过了自己的阈值，但分数最高的是 1272。请再录一段更长的录音。',
                },
                'expect_decision': 'uncertain',
                'speaker_name': None,
            },
            {
                'label': 'real-identify-gap-uncertain-1089',
                'outcome': {
                    'tool': 'identify',
                    'ok': True,
                    'result': {'decision': 'uncertain', 'candidate': '1089'},
                    'reply': '没有足够把握。最接近的是 1089。请再录一段更长的录音。',
                },
                'expect_decision': 'uncertain',
                'speaker_name': None,
            },
        ]
        with patched_dpapi():
            res = evaluate_response(real_non_detail_checks=real_checks)
        self.assertTrue(res['passed'], res)
        self.assertEqual(res['leak_count'], 0)
        self.assertEqual(res['passed_count'], res['total'])

    def test_check_reply_payload_catches_threshold_and_vector_leaks(self):
        # Negative test 1: threshold score string leak in detail=False
        ok_thr, msg_thr = check_reply_payload(
            {'tool': 'verify', 'ok': True, 'result': {'decision': 'accept', 'task': 'verify'}, 'reply': 'alice：结论：接受。阈值 0.42'},
            expect_decision='accept',
            detail=False,
            speaker_name='alice',
        )
        self.assertFalse(ok_thr, msg_thr)

        # Negative test 2: old service.py leak ('归一化相似度高于验证阈值') in detail=False
        ok_word, msg_word = check_reply_payload(
            {'tool': 'verify', 'ok': True, 'result': {'decision': 'accept', 'task': 'verify'}, 'reply': 'alice：结论：接受。归一化相似度高于验证阈值。'},
            expect_decision='accept',
            detail=False,
            speaker_name='alice',
        )
        self.assertFalse(ok_word, msg_word)

        # Negative test 3: forbidden score key in result under detail=False
        ok_key, msg_key = check_reply_payload(
            {'tool': 'verify', 'ok': True, 'result': {'decision': 'accept', 'task': 'verify', 'score': 1.2}, 'reply': 'alice：结论：接受。'},
            expect_decision='accept',
            detail=False,
            speaker_name='alice',
        )
        self.assertFalse(ok_key, msg_key)

        # Negative test 4: > 16 embedding list leak in result (both detail=False and detail=True)
        for detail_flag in (False, True):
            ok_vec, msg_vec = check_reply_payload(
                {
                    'tool': 'verify',
                    'ok': True,
                    'result': {'decision': 'accept', 'score': 2.0, 'embedding': [0.1] * 17}
                    if detail_flag
                    else {'decision': 'accept', 'embedding': [0.1] * 17},
                    'reply': 'alice：结论：接受。',
                },
                expect_decision='accept',
                detail=detail_flag,
                speaker_name='alice',
            )
            self.assertFalse(ok_vec, f'failed to catch >16 vector leak for detail={detail_flag}: {msg_vec}')

        # Negative test 5: split-winner lookalike with numeric score/threshold leak must still be caught
        for bad_reply in (
            '没有足够把握。1089 过了自己的阈值，但分数最高的是 1272 0.42。请再录一段更长的录音。',
            '没有足够把握。1089 过了自己的阈值，但分数最高的是 0.42。请再录一段更长的录音。',
            '没有足够把握。1089 过了自己的阈值 0.42，但分数最高的是 1272。请再录一段更长的录音。',
        ):
            ok_sw, msg_sw = check_reply_payload(
                {
                    'tool': 'identify',
                    'ok': True,
                    'result': {'decision': 'uncertain', 'candidate': '1089'},
                    'reply': bad_reply,
                },
                expect_decision='uncertain',
                detail=False,
                speaker_name=None,
            )
            self.assertFalse(ok_sw, f'failed to catch leaked split-winner reply {bad_reply!r}: {msg_sw}')

        # Negative test 6: verify cannot reuse identify split-winner threshold phrasing
        ok_ver_sw, msg_ver_sw = check_reply_payload(
            {
                'tool': 'verify',
                'ok': True,
                'result': {'decision': 'uncertain', 'task': 'verify'},
                'reply': '没有足够把握。alice 过了自己的阈值，但分数最高的是 bob。请再录一段更长的录音。',
            },
            expect_decision='uncertain',
            detail=False,
            speaker_name='alice',
        )
        self.assertFalse(ok_ver_sw, msg_ver_sw)

        # Negative test 7: evaluate_response flags leaked real_non_detail_checks cleanly
        with patched_dpapi():
            bad = evaluate_response(
                real_non_detail_checks=[
                    {
                        'label': 'injected-leak',
                        'outcome': {'tool': 'verify', 'ok': True, 'result': {'decision': 'accept'}, 'reply': 'alice：接受 阈值 0.42'},
                        'expect_decision': 'accept',
                        'speaker_name': 'alice',
                    }
                ]
            )
        self.assertFalse(bad['passed'])
        self.assertGreaterEqual(bad['leak_count'], 1)

    def test_evaluate_flow_passes_with_injected_and_default_separator(self):
        def fast_separator(_src: str, dest_dir: Path) -> dict:
            dest_dir.mkdir(parents=True, exist_ok=True)
            p1 = generate_speaker_wav(dest_dir / 'fast_spk1.wav', speaker_index=0, duration=2.0, phase=2.0, seed=11)
            p2 = generate_speaker_wav(dest_dir / 'fast_spk2.wav', speaker_index=1, duration=2.0, phase=2.0, seed=12)
            return {'sample_rate': 8000, 'paths': [str(p1), str(p2)], 'checkpoint': 'mock'}

        with patched_dpapi():
            res_fast = evaluate_flow(separator=fast_separator)
            self.assertTrue(res_fast['passed'], res_fast)
            self.assertEqual(res_fast['passed_count'], res_fast['total'])
            self.assertEqual(res_fast['total'], 7)

            res_default = evaluate_flow()
            self.assertTrue(res_default['passed'], res_default)
            self.assertEqual(res_default['passed_count'], res_default['total'])
            self.assertGreaterEqual(res_default['total'], 7)

    def test_evaluate_flow_catches_injected_pending_revoke_bug(self):
        orig_handle = VoiceAgent.handle

        def broken_handle(self, text: str) -> dict:
            saved_pending = self.pending
            out = orig_handle(self, text)
            if text.strip() == '名单':
                self.pending = saved_pending
            return out

        with patched_dpapi(), patch.object(VoiceAgent, 'handle', broken_handle):
            res = evaluate_flow(separator=lambda src, d: {'sample_rate': 8000, 'paths': [src, src], 'checkpoint': 'm'})
        self.assertFalse(res['passed'])
        failed = [c for c in res['checks'] if not c['passed']]
        self.assertTrue(any('revoke' in c['id'] for c in failed), failed)


class SecurityAndRobustnessAgentChecksTests(unittest.TestCase):
    def test_evaluate_redteam_passes_and_merges_security_rows(self):
        from voice_eval.dims.redteam import evaluate_redteam

        with patched_dpapi():
            res = evaluate_redteam()
        self.assertTrue(res['passed'], res)
        self.assertEqual(res['yellow'], 0)
        self.assertEqual(res['red'], 0)
        self.assertEqual(res['skipped'], 0)
        self.assertEqual(res['green'], res['total_executed'])
        self.assertGreaterEqual(res['total_executed'], 18)

        ids = {c['id'] for c in res['checks']}
        for required in (
            'export_zh_vector',
            'export_zh_raw',
            'export_unclosed_quote',
            'export_en_dump',
            'path_traversal_dotdot',
            'path_outside_root',
            'path_unc_backslash',
            'path_unc_slash',
            'path_symlink_escape',
            'name_nul_byte',
            'name_newline_inject',
            'name_overlong',
            'name_whitespace_only',
            'poison_empty_bytes',
            'poison_non_audio_wav',
            'poison_silence_zeros',
            'poison_nan_verify',
            'poison_nan_enroll',
            'poison_inf_verify',
            'poison_tiny_01s',
            'poison_bad_suffix',
            'enroll_missing_consent',
            'enroll_overwrite_existing',
        ):
            self.assertIn(required, ids)

        with patched_dpapi():
            merged_ok = evaluate_redteam(
                security_rows=[
                    {'id': 'security-no-plaintext', 'correct': True, 'reply': 'ok'},
                    {'id': 'security-rotate', 'correct': True, 'reply': 'ok'},
                ]
            )
        self.assertTrue(merged_ok['passed'], merged_ok)
        self.assertEqual(merged_ok['green'], res['green'] + 2)

        with patched_dpapi():
            merged_fail = evaluate_redteam(
                security_rows=[{'id': 'security-setup', 'correct': False, 'reply': 'failed setup'}]
            )
        self.assertFalse(merged_fail['passed'])
        self.assertEqual(merged_fail['red'], 1)

    def test_evaluate_redteam_symlink_oserror_classifies_skipped_no_privilege(self):
        from voice_eval.dims.redteam import evaluate_redteam

        with patched_dpapi(), patch('os.symlink', side_effect=PermissionError('privilege not held')):
            res = evaluate_redteam()
        self.assertTrue(res['passed'], res)
        self.assertEqual(res['yellow'], 0)
        self.assertEqual(res['red'], 0)
        self.assertEqual(res['skipped'], 1)
        by_id = {c['id']: c for c in res['checks']}
        self.assertEqual(by_id['path_symlink_escape']['status'], 'SKIPPED_NO_PRIVILEGE')
        self.assertEqual(by_id['path_outside_root']['status'], 'GREEN')

    def test_evaluate_redteam_catches_injected_leaks_and_bypasses(self):
        from voice_eval.dims.redteam import evaluate_redteam

        orig_handle = VoiceAgent.handle

        def leaking_export(self, text: str) -> dict:
            out = orig_handle(self, text)
            if 'dump embedding' in text.lower():
                out = dict(out)
                out['result'] = {'embedding': [0.1] * 32}
            return out

        with patched_dpapi(), patch.object(VoiceAgent, 'handle', leaking_export):
            res_leak = evaluate_redteam()
        self.assertFalse(res_leak['passed'])
        self.assertGreaterEqual(res_leak['red'], 1)

        def crashing_verify(self, text: str) -> dict:
            out = orig_handle(self, text)
            if 'bad.py' in text:
                return {'tool': 'error', 'ok': False, 'result': {'error': 'RuntimeError'}, 'reply': 'Traceback (most recent call last)'}
            return out

        with patched_dpapi(), patch.object(VoiceAgent, 'handle', crashing_verify):
            res_yellow = evaluate_redteam()
        self.assertFalse(res_yellow['passed'])
        self.assertGreaterEqual(res_yellow['yellow'], 1)

        def allow_outside_root(self, text: str) -> dict:
            out = orig_handle(self, text)
            if 'secret.wav' in text and '..' not in text:
                return {'tool': 'verify', 'ok': True, 'result': {'decision': 'accept'}, 'reply': 'bypassed'}
            return out

        with patched_dpapi(), patch('os.symlink', side_effect=PermissionError('privilege not held')), \
                patch.object(VoiceAgent, 'handle', allow_outside_root):
            res_escape = evaluate_redteam()
        self.assertFalse(res_escape['passed'])
        self.assertGreaterEqual(res_escape['red'], 1)
        self.assertEqual({c['id']: c for c in res_escape['checks']}['path_outside_root']['status'], 'RED')

        with patched_dpapi(), patch('voiceprint.audio._assert_finite_audio', lambda *_a, **_kw: None):
            res_nan_reg = evaluate_redteam()
        self.assertFalse(res_nan_reg['passed'])
        self.assertGreaterEqual(res_nan_reg['yellow'] + res_nan_reg['red'], 1)

    def test_evaluate_redteam_unc_backslash_triggers_text_layer_and_catches_fallback(self):
        import agent.service as service_mod
        from voice_eval.dims.redteam import evaluate_redteam
        from voiceprint.audio import AudioError

        orig_check_audio = service_mod.check_audio
        seen_paths: list[str] = []

        def spy_check_audio(path: str, roots=None):
            seen_paths.append(str(path))
            return orig_check_audio(path, roots)

        with patched_dpapi(), patch.object(service_mod, 'check_audio', spy_check_audio):
            res_ok = evaluate_redteam()
        self.assertTrue(res_ok['passed'], res_ok)
        self.assertTrue(any(p.startswith('\\\\') for p in seen_paths), f'no UNC \\\\ path seen: {seen_paths}')

        def buggy_check_audio_no_unc_backslash(path: str, roots=None):
            text = str(path).strip()
            if text.startswith('\\\\'):
                raise AudioError(f'找不到录音：{text}')
            return orig_check_audio(path, roots)

        with patched_dpapi(), patch.object(service_mod, 'check_audio', buggy_check_audio_no_unc_backslash):
            res_bad = evaluate_redteam()
        self.assertFalse(res_bad['passed'], res_bad)
        by_id = {c['id']: c for c in res_bad['checks']}
        self.assertEqual(by_id['path_unc_backslash']['status'], 'RED')

    def test_evaluate_fault_recovery_passes_and_catches_failure(self):
        from voice_eval.dims.robustness import evaluate_fault_recovery

        with patched_dpapi():
            res = evaluate_fault_recovery()
        self.assertTrue(res['passed'], res)
        self.assertEqual(res['fault_total'], 4)
        self.assertEqual(res['recovered'], 4)
        self.assertAlmostEqual(res['recovery_rate'], 1.0, places=6)

        orig_handle = VoiceAgent.handle

        def corrupt_after_fault(self, text: str) -> dict:
            out = orig_handle(self, text)
            if 'corrupt.wav' in text:
                self._faulted = True
            elif getattr(self, '_faulted', False) and text.startswith('验证 '):
                out = dict(out)
                out['ok'] = False
                out['result'] = {'decision': 'reject'}
            return out

        with patched_dpapi(), patch.object(VoiceAgent, 'handle', corrupt_after_fault):
            bad = evaluate_fault_recovery()
        self.assertFalse(bad['passed'])
        self.assertLess(bad['recovery_rate'], 1.0)

    def test_evaluate_consistency_passes_and_catches_non_determinism(self):
        from voice_eval.dims.robustness import evaluate_consistency

        with patched_dpapi():
            res = evaluate_consistency(repeats=3)
        self.assertTrue(res['passed'], res)
        self.assertEqual(res['trials'], 5)
        self.assertEqual(res['repeats'], 3)
        self.assertAlmostEqual(res['agreement_rate'], 1.0, places=6)
        self.assertLess(res['max_score_spread'], 1e-4)

        orig_score = VoiceAgent._score
        counter = {'n': 0}

        def unstable_score(self, name: str, probe, task: str = 'verify') -> dict:
            out = dict(orig_score(self, name, probe, task=task))
            counter['n'] += 1
            if counter['n'] % 2 == 0:
                out['score'] = -5.0
                out['decision'] = 'reject'
            else:
                out['score'] = 5.0
                out['decision'] = 'accept'
            return out

        with patched_dpapi(), patch.object(VoiceAgent, '_score', unstable_score):
            bad = evaluate_consistency(repeats=3)
        self.assertFalse(bad['passed'])
        self.assertLess(bad['agreement_rate'], 1.0)


if __name__ == '__main__':
    unittest.main()

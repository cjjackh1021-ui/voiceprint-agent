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


if __name__ == '__main__':
    unittest.main()

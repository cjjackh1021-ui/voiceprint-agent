"""Voiceprint agent: Chinese commands in, one tool call out.

The model answers with a decision and an explanation. It never returns the embedding itself.
Exact scores are only put in the reply when the agent is started with detail=True: a caller who
can read fine-grained scores and retry freely can climb towards an accepted voice.
"""
import os
import re
import shlex
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
import uuid

import numpy as np
import soundfile as sf

from voiceprint.audio import AudioError, MIN_SECONDS, check_audio, duration_seconds, file_digest
from voiceprint.audio_quality import analyze_audio_quality
from voiceprint.embed import cohort_matrix
from voiceprint.features import SR
from voiceprint.scoring import decide, profile_for
from voiceprint.security import apply_projection, projection, undo_projection
from voiceprint.spoof import DEFAULT_WATERMARK_PAYLOAD, SpreadSpectrumWatermarkDetector, embed_spread_watermark
from voiceprint.store import ModelMismatch, Store
from voiceprint.synth import (
    AcousticFormantSynthesizer,
    MAX_SIM_TEXT_CHARS,
    SimulationRequest,
    SynthesisError,
    VoiceSynthesizer,
    validate_simulation_result,
)

TRAIN_ROOT = Path(__file__).resolve().parents[1] / 'data' / 'LibriSpeech' / 'train-clean-100'
DECISION_ZH = {'accept': '接受', 'reject': '拒绝', 'uncertain': '不确定', 'unknown': '库里没有匹配的人'}
EXPORT_WORDS = ('明文', '原始声纹', '导出模板', '声纹向量')
ENGLISH_EXPORT_TRIGGERS = (
    'export_template',
    'dump embedding',
    'print embedding',
    'export embedding',
    'raw embedding',
)
NEAR_DUPLICATE_COSINE = 0.995
MAX_FAILED_VERIFY = 5
LOCK_MINUTES = 10
NAME_MAX = 64
_UNSET = object()
_SAFE_NAME_RE = re.compile(r'[^A-Za-z0-9_-]+')


def _safe_filename_token(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub('', name).strip('_-')
    return (cleaned[:32] if cleaned else 'spk')


class VoiceAgent:
    def __init__(
        self,
        data_dir: str | Path,
        embedder,
        detail: bool = False,
        audio_roots: list[str | Path] | None = None,
        spoof_detector=None,
        synthesizer: VoiceSynthesizer | None | object = _UNSET,
        max_sim_attempts: int = 3,
    ):
        """detail: put exact scores in replies and results (evaluation, local study).
        audio_roots: when given, recordings are only read from inside these folders.
        spoof_detector: optional SpoofDetector guarding enroll/verify/identify.
        synthesizer: voice simulation backend; unset uses AcousticFormantSynthesizer(), None disables simulation."""
        if isinstance(max_sim_attempts, bool) or not isinstance(max_sim_attempts, int):
            raise ValueError('max_sim_attempts 必须为整数。')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.model = (embedder.name, embedder.model_id)
        self.profile = profile_for(embedder.name)
        self.detail = detail
        self.spoof_detector = spoof_detector
        self.synthesizer = AcousticFormantSynthesizer() if synthesizer is _UNSET else synthesizer
        self.max_sim_attempts = max(1, min(5, int(max_sim_attempts)))
        self._watermark_guard = SpreadSpectrumWatermarkDetector()
        self.separated_dir = self.data_dir / 'separated'
        self.simulated_dir = self.data_dir / 'simulated'
        self.audio_roots = (
            [Path(root) for root in audio_roots] + [self.separated_dir, self.simulated_dir]
            if audio_roots
            else None
        )
        self.store = Store(self.data_dir / 'voice.db', self.data_dir / 'master.key')
        self.cohort, self.cohort_names = cohort_matrix(embedder, TRAIN_ROOT)
        self.pending = None

    def close(self):
        self.store.close()

    def handle(self, text: str) -> dict:
        try:
            outcome = self._dispatch(text.strip())
        except Exception as exc:
            outcome = {'tool': 'error', 'ok': False, 'result': {'error': type(exc).__name__},
                       'reply': f'没有完成：{exc}'}
        self.store.audit(outcome['tool'], outcome['reply'])
        outcome['model'] = self.embedder.describe()
        return outcome

    def _dispatch(self, text: str) -> dict:
        if not text:
            self.pending = None
            return self._reply('help', True, {}, '请输入命令。输入“帮助”查看用法。')
        wants_export = any(word in text for word in EXPORT_WORDS)
        wants_english_export = any(trigger in text.lower() for trigger in ENGLISH_EXPORT_TRIGGERS)
        commands = {
            '注册': self._enroll, '录入': self._enroll,
            '验证': self._verify, '核对': self._verify,
            '辨认': self._identify, '识别': self._identify,
            '撤销': self._ask_revoke, '确认撤销': self._confirm_revoke,
            '轮换密钥': self._rotate, '漂移': self._drift,
            '分离再辨认': self._separate_and_identify, '分离': self._separate,
            '质检': self._quality,
            '模拟': self._simulate,
        }
        try:
            tokens = shlex.split(text, posix=True)
        except ValueError as exc:
            head_guess = text.split(maxsplit=1)[0]
            if head_guess != '确认撤销':
                self.pending = None
            if wants_export or (head_guess not in commands and wants_english_export):
                return self._refuse_export()
            return self._reply('error', False, {}, f'命令没有解析成功：{exc}')
        if not tokens:
            self.pending = None
            return self._reply('help', True, {}, '请输入命令。输入“帮助”查看用法。')
        head, rest = tokens[0], tokens[1:]
        if head != '确认撤销':
            self.pending = None
        if head in ('帮助', 'help'):
            return self._reply('help', True, {}, HELP)
        if head in ('退出',):
            return self._reply('help', True, {}, '结束。')
        if head in ('名单',):
            return self._reply('list', True, {'names': self.store.names()},
                               '已注册：' + ('、'.join(self.store.names()) or '（空）'))
        if head in ('匿名', '变声'):
            return self._reply(
                'anonymize', False, {},
                '当前变声模型只在 VCC2016 的四个说话人上训练过，不能当作任意声音的匿名化，也不是密码学加密。')
        if head in commands:
            # A known command is run as that command even if a name or a path happens to contain
            # one of the export words; none of the commands returns a vector.
            return commands[head](rest)
        if wants_export or wants_english_export:
            return self._refuse_export()
        return self._reply('help', False, {}, '没有对应的工具。\n' + HELP)

    def _refuse_export(self) -> dict:
        return self._reply('refuse', True, {'requested': 'export_template'},
                           '不能导出声纹明文。模板只以加密形式保存，比对在进程内完成。')

    def _audio(self, path: str) -> str:
        return str(check_audio(path, self.audio_roots))

    def _check_spoof(self, path: str, tool: str) -> dict | None:
        if self.spoof_detector is None:
            return None
        info = sf.info(path)
        wav, sr = sf.read(path, frames=int(30.0 * info.samplerate), dtype='float32', always_2d=False)
        res = self.spoof_detector.detect(wav, int(sr))
        if not res.is_spoof:
            return None
        payload: dict = {'decision': 'reject', 'reason': 'spoof_detected'}
        if self.detail:
            payload['detector'] = res.detector
            payload['spoof_score'] = float(res.score)
            payload['spoof_reason'] = res.reason
        return self._reply(tool, False, payload, '拒绝：检测到疑似合成或水印伪造语音。')

    def _enroll(self, rest: list[str]) -> dict:
        if len(rest) < 5 or rest[1] != '本人同意':
            return self._reply('enroll', False, {}, '注册需要姓名、本人同意，以及至少 3 段录音。')
        name, paths = rest[0], rest[2:]
        if not name.strip() or len(name) > NAME_MAX or any(ord(ch) < 32 for ch in name):
            return self._reply('enroll', False, {}, f'姓名不能为空、不能含控制字符，且不超过 {NAME_MAX} 个字符。')
        if len(paths) < 3:
            return self._reply('enroll', False, {}, '至少需要 3 段录音。')
        if self.store.exists(name):
            # Overwriting here would let anyone replace a registered voice with their own.
            return self._reply('enroll', False, {'decision': 'reject', 'reason': 'exists'},
                               f'已经有 {name} 的模板，不会覆盖。要更换，请先“撤销 {name}”并确认，再重新注册。')
        try:
            paths = [self._audio(p) for p in paths]
        except AudioError as exc:
            return self._reply('enroll', False, {}, str(exc))
        for p in paths:
            q_rep = analyze_audio_quality(p)
            enroll_issues = [r for r in q_rep.reject_reasons if r in ('empty', 'silence', 'clipping')]
            if enroll_issues:
                return self._reply(
                    'enroll',
                    False,
                    {'decision': 'reject', 'reason': 'poor_audio_quality', 'quality_reasons': enroll_issues},
                    f'拒绝：注册录音质量不达标（{", ".join(enroll_issues)}），请重新录制。',
                )
            spoof_block = self._check_spoof(p, 'enroll')
            if spoof_block is not None:
                return spoof_block
        if len({file_digest(p) for p in paths}) < len(paths):
            return self._reply('enroll', False, {'decision': 'reject', 'reason': 'duplicate'},
                               '有内容完全相同的录音。请提供至少 3 段不同的录音。')
        vectors = np.stack([self.embedder.embed_file(p) for p in paths])
        problem = self._enrollment_problem(paths, vectors)
        if problem:
            return self._reply('enroll', False, {'decision': 'reject'}, problem)
        seed = os.urandom(32)
        projected = apply_projection(vectors, projection(seed, vectors.shape[1]))
        self.store.save_speaker(name, seed, projected, *self.model)
        total = sum(duration_seconds(p) for p in paths)
        extra = '' if total >= 30 else f' 当前总时长 {total:.0f} 秒，建议再补到 30 秒。'
        return self._reply('enroll', True, {'name': name, 'segments': len(paths), 'seconds': round(total, 1)},
                           f'已注册 {name}，共 {len(paths)} 段。模板已加密，并做了可撤销投影。{extra}')

    def _enrollment_problem(self, paths: list[str], vectors: np.ndarray) -> str:
        sims = [float(np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-8))
                for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
        if min(sims) < self.profile['enroll_min']:
            return f'这几段之间的最低相似度只有 {min(sims):.2f}，不像同一个人。请重录后再注册。'
        if max(sims) > NEAR_DUPLICATE_COSINE:
            return '有两段录音几乎完全相同，像是同一段录音的副本。请提供不同的录音。'
        folders = {Path(path).parent.name for path in paths}
        total = sum(duration_seconds(path) for path in paths)
        if len(folders) < 2 and total < 30:
            return (f'这几段都在同一个文件夹里，而且总时长只有 {total:.0f} 秒。'
                    '请换一次录音，或把同一个文件夹里的录音补到 30 秒后再注册。')
        return ''

    def _verify(self, rest: list[str]) -> dict:
        if len(rest) != 2:
            return self._reply('verify', False, {}, '用法：验证 姓名 "录音路径"')
        name, path = rest
        if not self.store.exists(name):
            return self._reply('verify', False, {'decision': 'unknown'}, f'没有 {name} 的模板。')
        failures = self.store.recent_failures(name, LOCK_MINUTES)
        if failures >= MAX_FAILED_VERIFY:
            return self._reply('verify', False, {'decision': 'locked', 'failures': failures},
                               f'{name} 已连续 {failures} 次验证未通过，暂停验证。请 {LOCK_MINUTES} 分钟后再试。')
        try:
            path = self._audio(path)
        except AudioError as exc:
            return self._reply('verify', False, {}, str(exc))
        q_rep = analyze_audio_quality(path)
        if q_rep.active_seconds + 1e-8 < MIN_SECONDS or 'empty' in q_rep.reject_reasons:
            return self._reply(
                'verify',
                False,
                {'decision': 'reject', 'reason': 'poor_audio_quality'},
                '拒绝：录音中有效语音不足或无效，请重新录制。',
            )
        spoof_block = self._check_spoof(path, 'verify')
        if spoof_block is not None:
            return spoof_block
        score = self._score(name, self.embedder.embed_file(path), task='verify')
        self.store.event(name, 'verify', score['score'], score['decision'])
        if self.detail:
            reply = (f'{name}：归一化相似度 {score["score"]:.2f}，原始余弦 {score["raw_score"]:.3f}，'
                     f'夹角 {score["theta_deg"]:.1f}°，阈值 {score["threshold"]:.2f}。'
                     f'结论：{DECISION_ZH[score["decision"]]}。{score["reason"]}')
            return self._reply('verify', True, score, reply)
        result = {'decision': score['decision'], 'task': 'verify'}
        reply = f'{name}：结论：{DECISION_ZH[score["decision"]]}。'
        if score['decision'] == 'uncertain':
            reply += '请再录一段更长、更安静的录音。'
        return self._reply('verify', True, result, reply)

    def _identify(self, rest: list[str]) -> dict:
        if len(rest) != 1:
            return self._reply('identify', False, {}, '用法：辨认 "录音路径"')
        names = self.store.names(*self.model)
        if not names:
            if self.store.names():
                return self._reply('identify', False, {'decision': 'unknown'},
                                   '库里的模板都是用其他声纹模型注册的，当前模型不能和它们比对。')
            return self._reply('identify', False, {'decision': 'unknown'}, '还没有注册任何人。')
        try:
            path = self._audio(rest[0])
        except AudioError as exc:
            return self._reply('identify', False, {}, str(exc))
        q_rep = analyze_audio_quality(path)
        if q_rep.active_seconds + 1e-8 < MIN_SECONDS or 'empty' in q_rep.reject_reasons:
            return self._reply(
                'identify',
                False,
                {'decision': 'reject', 'reason': 'poor_audio_quality'},
                '拒绝：录音中有效语音不足或无效，请重新录制。',
            )
        spoof_block = self._check_spoof(path, 'identify')
        if spoof_block is not None:
            return spoof_block
        probe = self.embedder.embed_file(path)
        ranked = []
        for name in names:
            score = self._score(name, probe, task='identify')
            ranked.append((score['score'], name, score))
        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, best_name, best = ranked[0]
        gap = best_score - ranked[1][0] if len(ranked) > 1 else best_score
        gap_min = self.profile['gap']
        # Thresholds are per speaker. If the top score misses its own threshold while a lower one
        # passes, neither "no match" nor that lower name is a safe answer.
        passed = [item for item in ranked if item[2]['decision'] == 'accept']
        if best['decision'] != 'accept' and passed:
            other_score, other_name, _ = passed[0]
            result = {'decision': 'uncertain', 'candidate': other_name}
            if self.detail:
                result.update(score=other_score, gap=other_score - best_score)
            return self._reply('identify', True, result,
                               f'没有足够把握。{other_name} 过了自己的阈值，但分数最高的是 {best_name}。'
                               '请再录一段更长的录音。')
        if best['decision'] == 'uncertain' or (best['decision'] == 'accept' and gap < gap_min):
            result = {'decision': 'uncertain', 'candidate': best_name}
            reply = f'没有足够把握。最接近的是 {best_name}。请再录一段更长的录音。'
            if self.detail:
                result.update(score=best_score, gap=gap)
                reply = (f'没有足够把握。最接近的是 {best_name}，归一化相似度 {best_score:.2f}，'
                         f'与下一名相差 {gap:.2f}。请再录一段更长的录音。')
            return self._reply('identify', True, result, reply)
        if best['decision'] != 'accept':
            result = {'decision': 'unknown', 'candidate': best_name}
            reply = f'没有足够把握的匹配。最接近的是 {best_name}，未达到辨认阈值。'
            if self.detail:
                result.update(score=best_score, gap=gap)
                reply = f'没有足够把握的匹配。最接近的是 {best_name}，归一化相似度 {best_score:.2f}，未达到辨认阈值。'
            return self._reply('identify', True, result, reply)
        result = {'decision': 'accept', 'name': best_name}
        reply = f'最匹配的是 {best_name}。'
        if self.detail:
            result.update(score=best_score, theta_deg=best['theta_deg'], gap=gap)
            reply = (f'最匹配的是 {best_name}，归一化相似度 {best_score:.2f}，夹角 {best["theta_deg"]:.1f}°，'
                     f'与下一名相差 {gap:.2f}。')
        self.store.event(best_name, 'identify', best_score, 'accept')
        return self._reply('identify', True, result, reply)

    def _ask_revoke(self, rest: list[str]) -> dict:
        if len(rest) != 1:
            return self._reply('revoke', False, {}, '用法：撤销 姓名')
        name = rest[0]
        if not self.store.exists(name):
            return self._reply('revoke', False, {}, f'没有 {name} 的模板。')
        self.pending = name
        return self._reply('revoke', True, {'pending': name}, f'请再输入：确认撤销 {name}')

    def _confirm_revoke(self, rest: list[str]) -> dict:
        if len(rest) != 1 or rest[0] != self.pending:
            return self._reply('revoke', False, {}, '没有待确认的撤销，或姓名不一致。')
        name = self.pending
        self.store.delete(name)
        self.pending = None
        return self._reply('revoke', True, {'deleted': name}, f'已删除 {name} 的模板和比对记录。')

    def _rotate(self, rest: list[str]) -> dict:
        if len(rest) != 1:
            return self._reply('rotate', False, {}, '用法：轮换密钥 姓名')
        name = rest[0]
        try:
            seed, segments = self.store.load_speaker(name, *self.model)
        except KeyError:
            return self._reply('rotate', False, {}, f'没有 {name} 的模板。')
        raw = undo_projection(segments, projection(seed, segments.shape[1]))
        new_seed = os.urandom(32)
        projected = apply_projection(raw, projection(new_seed, raw.shape[1]))
        self.store.save_speaker(name, new_seed, projected, *self.model, replace=True)
        return self._reply('rotate', True, {'name': name},
                           f'已更换 {name} 的投影密钥。旧模板不能再拿来比对，已注册的声音仍可验证。')

    def _drift(self, rest: list[str]) -> dict:
        if len(rest) != 1:
            return self._reply('drift', False, {}, '用法：漂移 姓名')
        name = rest[0]
        if not self.store.exists(name):
            return self._reply('drift', False, {'decision': 'unknown'}, f'没有 {name} 的模板。')
        rows = self.store.events(name)
        scores = [row[2] for row in rows if row[1] in ('verify', 'identify')]
        if len(scores) < 2:
            return self._reply('drift', True, {'n': len(scores)}, f'{name} 的验证记录不足，还不能判断漂移。')
        early, last = float(np.mean(scores[:-1])), scores[-1]
        drifted = last < early - 0.05
        # These are cohort-normalized scores, not cosines, so no angle is derived from them.
        text = f'{name} 共 {len(scores)} 次比对，先前平均归一化相似度 {early:.3f}，最近一次 {last:.3f}。'
        text += '最近一次明显低于先前水平，建议重新注册。' if drifted else '没有超出当前这条简单漂移带。'
        return self._reply('drift', True, {'n': len(scores), 'early': early, 'last': last, 'drift': drifted}, text)

    def _separate_and_identify(self, rest: list[str]) -> dict:
        if len(rest) != 1:
            return self._reply('separate', False, {}, '用法：分离再辨认 "混合录音路径"')
        separated = self._separate(rest)
        if not separated['ok']:
            return separated
        lines = [separated['reply']]
        names = []
        for path in separated['result']['paths']:
            found = self._identify([path])
            lines.append(found['reply'])
            if found['result'].get('decision') == 'accept':
                names.append(found['result'].get('name'))
        return self._reply('separate', True, {'paths': separated['result']['paths'], 'names': names},
                           '\n'.join(lines))

    def _separate(self, rest: list[str]) -> dict:
        if len(rest) != 1:
            return self._reply('separate', False, {}, '用法：分离 "混合录音路径"')
        try:
            path = self._audio(rest[0])
        except AudioError as exc:
            return self._reply('separate', False, {}, str(exc))
        from voiceprint.separate import separate_file
        result = separate_file(path, self.separated_dir)
        return self._reply('separate', True, {'paths': result['paths']},
                           '已分离成两路（8 kHz）：' + ' 和 '.join(result['paths']))

    def _quality(self, rest: list[str]) -> dict:
        if len(rest) != 1:
            return self._reply('quality', False, {}, '用法：质检 "录音路径"')
        try:
            path = self._audio(rest[0])
        except AudioError as exc:
            return self._reply('quality', False, {}, str(exc))
        report = analyze_audio_quality(path)
        summary = '通过' if report.passed_enroll_quality else f'未通过（{", ".join(report.reject_reasons)}）'
        return self._reply(
            'quality',
            True,
            report.to_dict(),
            f'录音质检{summary}：时长 {report.duration:.2f}s，有效语音 {report.active_seconds:.2f}s，削波占比 {report.clipping_ratio:.1%}。',
        )

    def _score(self, name: str, probe: np.ndarray, task: str = 'verify') -> dict:
        seed, segments = self.store.load_speaker(name, *self.model)
        matrix = projection(seed, segments.shape[1])
        projected = apply_projection(probe, matrix)
        cohort = apply_projection(self.cohort, matrix) if len(self.cohort) else np.zeros((0, segments.shape[1]))
        return decide(projected, segments, cohort, task=task, profile=self.embedder.name)

    def _parse_simulate_tokens(self, rest: list[str]) -> dict | None:
        if len(rest) not in (5, 7):
            return None
        name, consent_tok, ref_path, mode_tok, payload_tok = rest[:5]
        reference_text = None
        if len(rest) == 7:
            if rest[5] != '参考文本':
                return None
            reference_text = rest[6]
        if mode_tok == '源音频':
            mode: Literal['audio', 'text'] = 'audio'
            src_path, text_val = payload_tok, None
        elif mode_tok == '文本':
            mode = 'text'
            src_path, text_val = None, payload_tok
        else:
            return None
        return {
            'name': name,
            'consent': (consent_tok == '本人授权'),
            'ref_path': ref_path,
            'mode': mode,
            'src_path': src_path,
            'text': text_val,
            'reference_text': reference_text,
        }

    def _simulate(self, rest: list[str]) -> dict:
        parsed = self._parse_simulate_tokens(rest)
        if parsed is None:
            return self._reply(
                'simulate',
                False,
                {'reason': 'invalid_syntax'},
                '用法：模拟 姓名 本人授权 "参考录音" 源音频 "源录音" 或 模拟 姓名 本人授权 "参考录音" 文本 "合成文本" [参考文本 "参考文本"]',
            )
        outcome, _ = self._run_simulate_core(
            parsed['name'],
            consent=parsed['consent'],
            ref_path=parsed['ref_path'],
            mode=parsed['mode'],
            src_path=parsed['src_path'],
            text=parsed['text'],
            reference_text=parsed['reference_text'],
        )
        return outcome

    def _run_simulate_core(
        self,
        name: str,
        *,
        consent: bool,
        ref_path: str,
        mode: Literal['audio', 'text'],
        src_path: str | None = None,
        text: str | None = None,
        reference_text: str | None = None,
        max_attempts: int | None = None,
        collect_best_candidate: bool = False,
    ) -> tuple[dict, np.ndarray | None]:
        def _fail(reason: str, reply: str, extra: dict | None = None, cand_wav: np.ndarray | None = None):
            res = {'decision': 'reject', 'reason': reason, 'converged': False}
            if extra:
                res.update(extra)
            return self._reply('simulate', False, res, reply), (cand_wav if collect_best_candidate else None)

        if self.synthesizer is None:
            return _fail('synthesizer_disabled', '拒绝：当前未启用声音模拟合成器。')
        if consent is not True:
            return _fail('consent_required', '拒绝：声音模拟必须获得目标说话人明确授权（本人授权）。')
        if mode not in self.synthesizer.supported_modes:
            return _fail('unsupported_mode', f'拒绝：当前合成后端 {self.synthesizer.name} 不支持 {mode} 模式。')
        if not name.strip() or len(name) > NAME_MAX or any(ord(ch) < 32 for ch in name):
            return _fail('invalid_name', f'姓名不能为空、不能含控制字符，且不超过 {NAME_MAX} 个字符。')

        eff_attempts = self.max_sim_attempts
        if max_attempts is not None:
            if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
                return _fail('invalid_arguments', 'max_attempts 必须为整数。')
            eff_attempts = max(1, min(5, int(max_attempts)))

        if not self.store.exists(name):
            return _fail('unknown', f'没有 {name} 的注册模板，不能执行模拟。')
        try:
            self.store.load_speaker(name, *self.model)
        except ModelMismatch as exc:
            return _fail('model_mismatch', str(exc))
        failures = self.store.recent_failures(name, LOCK_MINUTES)
        if failures >= MAX_FAILED_VERIFY:
            return _fail('locked', f'{name} 当前处于安全验证锁定期，暂停声纹相关操作。')

        try:
            safe_ref = self._audio(ref_path)
        except AudioError as exc:
            return _fail('invalid_reference_audio', str(exc))
        q_ref = analyze_audio_quality(safe_ref)
        ref_bad = [r for r in q_ref.reject_reasons if r in ('empty', 'silence', 'clipping', 'invalid')]
        if ref_bad or q_ref.active_seconds + 1e-8 < MIN_SECONDS:
            return _fail('poor_audio_quality', f'拒绝：参考录音质量不达标（{", ".join(ref_bad) or "too_short"}）。')

        ref_info = sf.info(safe_ref)
        ref_wav, ref_sr = sf.read(safe_ref, frames=int(30.0 * ref_info.samplerate), dtype='float32', always_2d=False)
        wm_guard_res = self._watermark_guard.detect(ref_wav, int(ref_sr))
        if wm_guard_res.is_spoof:
            return _fail('watermark_detected', '拒绝：参考音频已包含溯源水印，禁止递归二次克隆。')
        if self.spoof_detector is not None and self.spoof_detector is not self._watermark_guard:
            sp_res = self.spoof_detector.detect(ref_wav, int(ref_sr))
            if sp_res.is_spoof:
                return _fail('spoof_detected', '拒绝：参考音频未通过防伪检测，禁止用于模拟。')

        ref_emb = self.embedder.embed_file(safe_ref)
        owner_score = self._score(name, ref_emb, task='verify')
        if owner_score['decision'] != 'accept':
            return _fail('reference_speaker_mismatch', f'拒绝：参考录音未通过 {name} 的本人声纹归属核验。')

        safe_src: str | None = None
        clean_text: str | None = None
        clean_ref_text: str | None = None
        if reference_text is not None:
            clean_ref_text = str(reference_text).strip()
            if len(clean_ref_text) > MAX_SIM_TEXT_CHARS:
                return _fail('invalid_arguments', f'参考文本不能超过 {MAX_SIM_TEXT_CHARS} 个字符。')
        if mode == 'audio':
            if not src_path:
                return _fail('invalid_arguments', '源音频模式必须提供源录音路径。')
            try:
                safe_src = self._audio(src_path)
            except AudioError as exc:
                return _fail('invalid_source_audio', str(exc))
            q_src = analyze_audio_quality(safe_src)
            src_bad = [r for r in q_src.reject_reasons if r in ('empty', 'silence', 'invalid')]
            if src_bad or q_src.active_seconds + 1e-8 < MIN_SECONDS:
                return _fail('poor_audio_quality', '拒绝：源音频有效语音不足或无效。')
        else:
            clean_text = (text or '').strip()
            if not clean_text or len(clean_text) > MAX_SIM_TEXT_CHARS:
                return _fail('invalid_arguments', f'合成文本必须非空且不超过 {MAX_SIM_TEXT_CHARS} 个字符。')

        formant_blend = 0.65
        pitch_pull = 0.70
        gain_scale = 0.85
        wm_strength = 0.02
        attempt_history: list[dict] = []
        best_rank: tuple | None = None
        best_payload: tuple[np.ndarray, dict, dict] | None = None
        converged = False
        win_wav: np.ndarray | None = None
        win_meta: dict = {}
        win_score: dict = {}
        last_error: str | None = None

        with TemporaryDirectory(prefix='vpsim_cand_') as tmp_dir:
            tmp_root = Path(tmp_dir)
            for attempt in range(1, eff_attempts + 1):
                req = SimulationRequest(
                    target_name=name,
                    mode=mode,
                    reference_audio_path=safe_ref,
                    reference_text=clean_ref_text,
                    source_audio_path=safe_src,
                    text=clean_text,
                    target_sr=SR,
                    formant_blend=formant_blend,
                    pitch_pull=pitch_pull,
                    gain_scale=gain_scale,
                    seed=attempt - 1,
                    attempt_index=attempt,
                )
                try:
                    sim_res = self.synthesizer.synthesize(req)
                    clean_wav = validate_simulation_result(sim_res)
                except SynthesisError as exc:
                    last_error = str(exc)
                    attempt_history.append({'attempt': attempt, 'error': last_error})
                    if '超时' in last_error:
                        break
                    gain_scale = max(0.35, gain_scale * 0.85)
                    continue

                wm_wav = embed_spread_watermark(
                    clean_wav,
                    SR,
                    payload=DEFAULT_WATERMARK_PAYLOAD,
                    strength=wm_strength,
                )
                wm_check = self._watermark_guard.detect(wm_wav, SR)
                wm_ok = bool(
                    wm_check.is_spoof
                    and wm_check.metadata.get('payload') == DEFAULT_WATERMARK_PAYLOAD
                )

                cand_path = tmp_root / f'cand_{attempt}.wav'
                sf.write(str(cand_path), wm_wav, SR)
                q_cand = analyze_audio_quality(cand_path)
                fatal_q = [r for r in q_cand.reject_reasons if r in ('empty', 'silence', 'invalid', 'clipping')]
                has_clip = ('clipping' in fatal_q) or (q_cand.clipping_ratio > 0.05)
                has_dead = any(r in ('empty', 'silence', 'invalid') for r in fatal_q) or (
                    q_cand.active_seconds + 1e-8 < MIN_SECONDS
                )
                quality_ok = (not fatal_q) and (not has_clip) and (not has_dead)

                cand_emb = self.embedder.embed_file(str(cand_path))
                sv_score = self._score(name, cand_emb, task='verify')
                margin = float(sv_score['score'] - sv_score['threshold'])

                attempt_rec = {
                    'attempt': attempt,
                    'quality_ok': quality_ok,
                    'watermarked': wm_ok,
                    'watermark_z': float(wm_check.score),
                    'decision': sv_score['decision'],
                    'score': float(sv_score['score']),
                    'raw_score': float(sv_score['raw_score']),
                    'threshold': float(sv_score['threshold']),
                    'formant_blend': round(formant_blend, 3),
                    'pitch_pull': round(pitch_pull, 3),
                    'gain_scale': round(gain_scale, 3),
                }
                attempt_history.append(attempt_rec)

                if wm_ok and not has_dead:
                    rank = (1 if quality_ok else 0, margin, float(sv_score['raw_score']))
                    if best_rank is None or rank > best_rank:
                        best_rank = rank
                        best_payload = (wm_wav, sim_res.metadata, sv_score)

                if wm_ok and quality_ok and sv_score['decision'] == 'accept':
                    converged = True
                    win_wav = wm_wav
                    win_meta = sim_res.metadata
                    win_score = sv_score
                    break

                if has_clip or q_cand.clipping_ratio > 0.01:
                    gain_scale = max(0.35, gain_scale * 0.70)
                elif 'silence' in q_cand.reject_reasons or 'too_quiet' in q_cand.reject_reasons:
                    gain_scale = min(1.05, gain_scale * 1.15)
                if not wm_ok or wm_check.score < (self._watermark_guard.z_threshold + 0.5):
                    wm_strength = min(0.035, wm_strength + 0.008)
                if sv_score['decision'] != 'accept':
                    formant_blend = min(1.0, formant_blend + 0.18)
                    pitch_pull = min(1.0, pitch_pull + 0.15)

        if not converged or win_wav is None:
            if best_payload is None:
                return _fail(
                    'synthesis_failed' if last_error else 'watermark_verification_failed',
                    f'拒绝：声音模拟未通过合成或溯源水印校验（{last_error or "水印/质量未通过"}）。',
                    extra={'attempts': len(attempt_history)},
                )
            best_wav, _, best_sv = best_payload
            reason = 'similarity_not_converged'
            extra_fail: dict = {
                'attempts': len(attempt_history),
                'backend': self.synthesizer.name,
                'mode': mode,
            }
            if self.detail:
                extra_fail['attempt_history'] = attempt_history
                extra_fail['best_decision'] = best_sv.get('decision')
            return _fail(
                reason,
                f'拒绝：经 {len(attempt_history)} 轮自检调优后仍未达到目标声纹验收门禁，已拦截未释放模拟音频。',
                extra=extra_fail,
                cand_wav=best_wav,
            )

        self.simulated_dir.mkdir(parents=True, exist_ok=True)
        safe_tok = _safe_filename_token(name)
        out_name = f'sim_{safe_tok}_{uuid.uuid4().hex[:8]}.wav'
        out_path = (self.simulated_dir / out_name).resolve()
        tmp_out = out_path.with_name(out_name + '.tmp.wav')
        sf.write(str(tmp_out), win_wav, SR)
        os.replace(tmp_out, out_path)

        payload_hex = f'0x{DEFAULT_WATERMARK_PAYLOAD:04X}'
        result: dict = {
            'output_path': str(out_path),
            'mode': mode,
            'backend': self.synthesizer.name,
            'watermarked': True,
            'watermark_payload': payload_hex,
            'attempts': len(attempt_history),
            'converged': True,
        }
        if self.detail:
            result['score'] = float(win_score['score'])
            result['raw_score'] = float(win_score['raw_score'])
            result['threshold'] = float(win_score['threshold'])
            result['theta_deg'] = float(win_score['theta_deg'])
            result['attempt_history'] = attempt_history
            result['backend_metadata'] = win_meta
            reply = (
                f'已生成 {name} 的可控模拟语音（共 {len(attempt_history)} 轮收敛，已嵌入 {payload_hex} 溯源水印，'
                f'归一化相似度 {win_score["score"]:.2f} / 阈值 {win_score["threshold"]:.2f}）：{out_path}'
            )
        else:
            reply = (
                f'已完成 {name} 的授权声音模拟（共尝试 {len(attempt_history)} 轮，已嵌入溯源水印并通过质检与声纹自检）：{out_path}'
            )
        return self._reply('simulate', True, result, reply), (win_wav if collect_best_candidate else None)

    def _reply(self, tool: str, ok: bool, result: dict, reply: str) -> dict:
        return {'tool': tool, 'ok': ok, 'result': result, 'reply': reply}


HELP = """命令：
  注册 姓名 本人同意 "录音1" "录音2" "录音3"
  验证 姓名 "录音"
  辨认 "录音"
  分离再辨认 "混合录音"
  模拟 姓名 本人授权 "参考录音" 源音频 "源录音"
  模拟 姓名 本人授权 "参考录音" 文本 "待合成文本" [参考文本 "参考文本"]
  撤销 姓名
  确认撤销 姓名
  轮换密钥 姓名
  漂移 姓名
  分离 "混合录音"
  名单
注册的录音至少要来自两个文件夹，或同一个文件夹里合计满 30 秒。已有模板的姓名不能直接重新注册，要先撤销。
声纹明文不会被导出。撤销需要再确认一次。同一姓名连续 5 次验证未通过会暂停 10 分钟。
声音模拟必须本人授权、参考音通过防伪与本人核验，产物默认强制嵌入不可见溯源水印。
只有明确说“分离再辨认”时才会先做语音分离。"""

"""Voiceprint agent: Chinese commands in, one tool call out.

The model answers with a decision and an explanation. It never returns the embedding itself.
Exact scores are only put in the reply when the agent is started with detail=True: a caller who
can read fine-grained scores and retry freely can climb towards an accepted voice.
"""
import os
import shlex
from pathlib import Path

import numpy as np

from voiceprint.audio import AudioError, check_audio, duration_seconds, file_digest
from voiceprint.embed import cohort_matrix
from voiceprint.scoring import decide, profile_for
from voiceprint.security import apply_projection, projection, undo_projection
from voiceprint.store import Store

TRAIN_ROOT = Path(__file__).resolve().parents[1] / 'data' / 'LibriSpeech' / 'train-clean-100'
DECISION_ZH = {'accept': '接受', 'reject': '拒绝', 'uncertain': '不确定', 'unknown': '库里没有匹配的人'}
EXPORT_WORDS = ('明文', '原始声纹', '导出模板', '声纹向量')
NEAR_DUPLICATE_COSINE = 0.995
MAX_FAILED_VERIFY = 5
LOCK_MINUTES = 10
NAME_MAX = 64


class VoiceAgent:
    def __init__(self, data_dir: str | Path, embedder, detail: bool = False,
                 audio_roots: list[str | Path] | None = None):
        """detail: put exact scores in replies and results (evaluation, local study).
        audio_roots: when given, recordings are only read from inside these folders."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.model = (embedder.name, embedder.model_id)
        self.profile = profile_for(embedder.name)
        self.detail = detail
        self.separated_dir = self.data_dir / 'separated'
        self.audio_roots = [Path(root) for root in audio_roots] + [self.separated_dir] if audio_roots else None
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
            return self._reply('help', True, {}, '请输入命令。输入“帮助”查看用法。')
        wants_export = any(word in text for word in EXPORT_WORDS)
        try:
            tokens = shlex.split(text, posix=True)
        except ValueError as exc:
            if wants_export:
                return self._refuse_export()
            return self._reply('error', False, {}, f'命令没有解析成功：{exc}')
        head, rest = tokens[0], tokens[1:]
        commands = {
            '注册': self._enroll, '录入': self._enroll,
            '验证': self._verify, '核对': self._verify,
            '辨认': self._identify, '识别': self._identify,
            '撤销': self._ask_revoke, '确认撤销': self._confirm_revoke,
            '轮换密钥': self._rotate, '漂移': self._drift,
            '分离再辨认': self._separate_and_identify, '分离': self._separate,
        }
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
            if self.pending and head != '确认撤销':
                self.pending = None
            return commands[head](rest)
        if wants_export:
            return self._refuse_export()
        return self._reply('help', False, {}, '没有对应的工具。\n' + HELP)

    def _refuse_export(self) -> dict:
        return self._reply('refuse', True, {'requested': 'export_template'},
                           '不能导出声纹明文。模板只以加密形式保存，比对在进程内完成。')

    def _audio(self, path: str) -> str:
        return str(check_audio(path, self.audio_roots))

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
        score = self._score(name, self.embedder.embed_file(path), task='verify')
        self.store.event(name, 'verify', score['score'], score['decision'])
        if self.detail:
            reply = (f'{name}：归一化相似度 {score["score"]:.2f}，原始余弦 {score["raw_score"]:.3f}，'
                     f'夹角 {score["theta_deg"]:.1f}°，阈值 {score["threshold"]:.2f}。'
                     f'结论：{DECISION_ZH[score["decision"]]}。{score["reason"]}')
            return self._reply('verify', True, score, reply)
        result = {'decision': score['decision'], 'task': 'verify'}
        return self._reply('verify', True, result,
                           f'{name}：结论：{DECISION_ZH[score["decision"]]}。{score["reason"]}')

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

    def _score(self, name: str, probe: np.ndarray, task: str = 'verify') -> dict:
        seed, segments = self.store.load_speaker(name, *self.model)
        matrix = projection(seed, segments.shape[1])
        projected = apply_projection(probe, matrix)
        cohort = apply_projection(self.cohort, matrix) if len(self.cohort) else np.zeros((0, segments.shape[1]))
        return decide(projected, segments, cohort, task=task, profile=self.embedder.name)

    def _reply(self, tool: str, ok: bool, result: dict, reply: str) -> dict:
        return {'tool': tool, 'ok': ok, 'result': result, 'reply': reply}


HELP = """命令：
  注册 姓名 本人同意 "录音1" "录音2" "录音3"
  验证 姓名 "录音"
  辨认 "录音"
  分离再辨认 "混合录音"
  撤销 姓名
  确认撤销 姓名
  轮换密钥 姓名
  漂移 姓名
  分离 "混合录音"
  名单
注册的录音至少要来自两个文件夹，或同一个文件夹里合计满 30 秒。已有模板的姓名不能直接重新注册，要先撤销。
声纹明文不会被导出。撤销需要再确认一次。同一姓名连续 5 次验证未通过会暂停 10 分钟。
只有明确说“分离再辨认”时才会先做语音分离。"""

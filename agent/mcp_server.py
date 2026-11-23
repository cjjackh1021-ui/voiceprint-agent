"""Zero-dependency JSON-RPC 2.0 stdio MCP Server for voiceprint-agent.

Run standalone:
    python -m agent.mcp_server --data-dir ./data --audio-roots ./recordings
    python -m agent.mcp_server --data-dir ./data --read-only
"""
import argparse
import base64
import binascii
import json
import logging
import os
import re
import shlex
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.service import VoiceAgent
from voiceprint.audio import sniff_audio_format
from voiceprint.spoof import SpreadSpectrumWatermarkDetector
from voiceprint.synth import VoiceSynthesizer, load_synthesizer

PROTOCOL_VERSION = '2024-11-05'
SERVER_NAME = 'voiceprint-agent-mcp'
SERVER_VERSION = '1.0.0'
MAX_BASE64_BYTES = 32 * 1024 * 1024
_MAX_BASE64_CHARS = ((MAX_BASE64_BYTES + 2) // 3) * 4 + 64
_DATA_URI_RE = re.compile(r'^data:[^;,]+;base64,', re.IGNORECASE)
_SESSION_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,32}$')
_WRITE_TOOLS = frozenset({'voiceprint_enroll', 'voiceprint_revoke', 'voiceprint_rotate_key', 'voiceprint_simulate'})

_TOOL_SCHEMAS = [
    {
        'name': 'voiceprint_enroll',
        'description': 'Enroll a new speaker with explicit consent and at least 3 audio recordings (paths or base64).',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'consent': {'type': 'boolean'},
                'audio_paths': {'type': 'array', 'items': {'type': 'string'}},
                'audio_base64_list': {'type': 'array', 'items': {'type': 'string'}},
                'session_ids': {'type': 'array', 'items': {'type': 'string'}},
            },
            'required': ['name', 'consent'],
        },
    },
    {
        'name': 'voiceprint_verify',
        'description': 'Verify whether an audio recording matches a claimed enrolled speaker.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'audio_path': {'type': 'string'},
                'audio_base64': {'type': 'string'},
            },
            'required': ['name'],
        },
    },
    {
        'name': 'voiceprint_identify',
        'description': 'Identify the speaker of an audio recording against enrolled templates (open-set).',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'audio_path': {'type': 'string'},
                'audio_base64': {'type': 'string'},
            },
        },
    },
    {
        'name': 'voiceprint_list_speakers',
        'description': 'List currently enrolled speaker names (never returns templates or embeddings).',
        'inputSchema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'voiceprint_revoke',
        'description': 'Two-step speaker revocation: first call with confirm=false, then confirm=true.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'confirm': {'type': 'boolean', 'default': False},
            },
            'required': ['name'],
        },
    },
    {
        'name': 'voiceprint_rotate_key',
        'description': 'Rotate the cancelable orthogonal projection key for an enrolled speaker.',
        'inputSchema': {
            'type': 'object',
            'properties': {'name': {'type': 'string'}},
            'required': ['name'],
        },
    },
    {
        'name': 'voiceprint_check_audio_quality',
        'description': 'Run model-free container integrity and 100ms DSP signal quality check on an audio clip.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'audio_path': {'type': 'string'},
                'audio_base64': {'type': 'string'},
            },
        },
    },
    {
        'name': 'voiceprint_health',
        'description': 'Return voiceprint agent health, active embedder metadata and security guardrail flags.',
        'inputSchema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'voiceprint_simulate',
        'description': 'Consent-gated voice simulation with forced 16-bit provenance watermark and Self-Refine speaker checks.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'consent': {'type': 'boolean'},
                'reference_audio_path': {'type': 'string'},
                'reference_audio_base64': {'type': 'string'},
                'reference_text': {'type': 'string'},
                'source_audio_path': {'type': 'string'},
                'source_audio_base64': {'type': 'string'},
                'text': {'type': 'string'},
                'max_attempts': {'type': 'integer', 'minimum': 1, 'maximum': 5},
                'include_audio_base64': {'type': 'boolean', 'default': False},
            },
            'required': ['name', 'consent'],
        },
    },
]

_RESOURCES = [
    {
        'uri': 'voiceprint://speakers',
        'name': 'Enrolled Speakers',
        'description': 'Names and count of currently enrolled speakers (no secrets or vectors).',
        'mimeType': 'application/json',
    },
    {
        'uri': 'voiceprint://health',
        'name': 'Voiceprint Agent Health',
        'description': 'Runtime embedder metadata and security guardrail status.',
        'mimeType': 'application/json',
    },
]


def _decode_base64_audio(b64_text: str, target_dir: Path, stem: str) -> Path:
    if not isinstance(b64_text, str) or not b64_text.strip():
        raise ValueError('空的 base64 音频输入。')
    cleaned = _DATA_URI_RE.sub('', b64_text.strip(), count=1)
    cleaned = re.sub(r'\s+', '', cleaned)
    if len(cleaned) > _MAX_BASE64_CHARS:
        raise ValueError(f'base64 音频超过 {MAX_BASE64_BYTES // (1024 * 1024)} MiB 上限。')
    try:
        raw = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError('无效的 base64 编码。') from None
    if not raw:
        raise ValueError('解码后的音频为空。')
    if len(raw) > MAX_BASE64_BYTES:
        raise ValueError(f'解码后的音频超过 {MAX_BASE64_BYTES // (1024 * 1024)} MiB 上限。')
    ext = sniff_audio_format(raw)
    if ext is None:
        raise ValueError('无法从音频字节头识别受支持的音频格式（WAV/FLAC/OGG/MP3）。')
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f'{stem}{ext}'
    out_path.write_bytes(raw)
    return out_path


def _tool_response(payload: dict, *, is_error: bool = False) -> dict:
    return {
        'content': [{'type': 'text', 'text': json.dumps(payload, ensure_ascii=False)}],
        'isError': bool(is_error),
    }


def _jsonrpc_ok(req_id, result: dict) -> dict:
    return {'jsonrpc': '2.0', 'id': req_id, 'result': result}


def _jsonrpc_err(req_id, code: int, message: str) -> dict:
    return {'jsonrpc': '2.0', 'id': req_id, 'error': {'code': code, 'message': message}}


class VoiceprintMCPServer:
    """Thread-safe, zero-dependency JSON-RPC 2.0 stdio MCP server wrapping VoiceAgent."""

    def __init__(
        self,
        agent: VoiceAgent,
        *,
        audio_roots: list[Path] | None = None,
        read_only: bool = False,
    ):
        self.agent = agent
        self.audio_roots = [Path(r).resolve() for r in audio_roots] if audio_roots else []
        self.read_only = bool(read_only)
        self._lock = threading.RLock()

    def _health_dict(self) -> dict:
        return {
            'ok': True,
            'server': SERVER_NAME,
            'version': SERVER_VERSION,
            'model': self.agent.embedder.describe(),
            'audio_roots_configured': bool(self.audio_roots),
            'spoof_guard_enabled': self.agent.spoof_detector is not None,
            'simulation_enabled': self.agent.synthesizer is not None,
            'synthesizer': self.agent.synthesizer.describe() if self.agent.synthesizer is not None else None,
            'read_only': self.read_only,
            'detail': bool(self.agent.detail),
            'offline_env': os.environ.get('HF_HUB_OFFLINE') == '1',
        }

    @contextmanager
    def _ephemeral_roots(self, extra_dirs: list[Path]):
        orig_roots = self.agent.audio_roots
        try:
            if self.audio_roots or extra_dirs:
                base = (
                    list(self.audio_roots)
                    + [self.agent.separated_dir, self.agent.simulated_dir]
                    + list(extra_dirs)
                )
                self.agent.audio_roots = base
            yield
        finally:
            self.agent.audio_roots = orig_roots

    def _resolve_single_audio(
        self,
        args: dict,
        tmp_root: Path,
        *,
        path_key: str = 'audio_path',
        b64_key: str = 'audio_base64',
    ) -> tuple[str | None, list[Path], dict | None]:
        has_path = bool(args.get(path_key))
        has_b64 = bool(args.get(b64_key))
        if has_path == has_b64:
            return None, [], {'ok': False, 'error': 'invalid_arguments', 'reply': f'必须且只能提供 {path_key} 或 {b64_key} 之一。'}
        if has_path:
            if not self.audio_roots:
                return None, [], {
                    'ok': False,
                    'error': 'path_not_allowed',
                    'reply': '未配置 audio_roots 安全边界，拒绝路径型音频入参。请使用 audio_base64 或配置 --audio-roots。',
                }
            return str(args[path_key]), [], None
        try:
            decoded = _decode_base64_audio(args[b64_key], tmp_root / 'single', 'input')
        except ValueError as exc:
            return None, [], {'ok': False, 'error': 'invalid_base64_audio', 'reply': str(exc)}
        return str(decoded), [tmp_root], None

    def _call_tool_locked(self, name: str, args: dict) -> dict:
        if not isinstance(args, dict):
            return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': '工具参数必须为对象。'}, is_error=True)
        if self.read_only and name in _WRITE_TOOLS:
            return _tool_response({
                'ok': False,
                'error': 'read_only_mode',
                'reply': f'当前 MCP 服务运行在只读安全模式，禁止执行 {name}。',
            }, is_error=True)

        if name == 'voiceprint_health':
            return _tool_response(self._health_dict(), is_error=False)

        if name == 'voiceprint_list_speakers':
            out = self.agent.handle('名单')
            return _tool_response(out, is_error=not out['ok'])

        if name == 'voiceprint_revoke':
            spk = str(args.get('name') or '').strip()
            if not spk:
                return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': '缺少 name 参数。'}, is_error=True)
            confirm = args.get('confirm', False)
            if not isinstance(confirm, bool):
                return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': 'confirm 必须为布尔值。'}, is_error=True)
            cmd = f'确认撤销 {shlex.quote(spk)}' if confirm is True else f'撤销 {shlex.quote(spk)}'
            out = self.agent.handle(cmd)
            return _tool_response(out, is_error=not out['ok'])

        if name == 'voiceprint_rotate_key':
            spk = str(args.get('name') or '').strip()
            if not spk:
                return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': '缺少 name 参数。'}, is_error=True)
            out = self.agent.handle(f'轮换密钥 {shlex.quote(spk)}')
            return _tool_response(out, is_error=not out['ok'])

        if name == 'voiceprint_check_audio_quality':
            with TemporaryDirectory(prefix='vpmcp_q_') as tmp:
                audio_path, extra_roots, err = self._resolve_single_audio(args, Path(tmp))
                if err is not None:
                    return _tool_response(err, is_error=True)
                with self._ephemeral_roots(extra_roots):
                    out = self.agent.handle(f'质检 {shlex.quote(audio_path)}')
                return _tool_response(out, is_error=not out['ok'])

        if name == 'voiceprint_verify':
            spk = str(args.get('name') or '').strip()
            if not spk:
                return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': '缺少 name 参数。'}, is_error=True)
            with TemporaryDirectory(prefix='vpmcp_v_') as tmp:
                audio_path, extra_roots, err = self._resolve_single_audio(args, Path(tmp))
                if err is not None:
                    return _tool_response(err, is_error=True)
                with self._ephemeral_roots(extra_roots):
                    out = self.agent.handle(f'验证 {shlex.quote(spk)} {shlex.quote(audio_path)}')
                return _tool_response(out, is_error=not out['ok'])

        if name == 'voiceprint_identify':
            with TemporaryDirectory(prefix='vpmcp_i_') as tmp:
                audio_path, extra_roots, err = self._resolve_single_audio(args, Path(tmp))
                if err is not None:
                    return _tool_response(err, is_error=True)
                with self._ephemeral_roots(extra_roots):
                    out = self.agent.handle(f'辨认 {shlex.quote(audio_path)}')
                return _tool_response(out, is_error=not out['ok'])

        if name == 'voiceprint_enroll':
            spk = str(args.get('name') or '').strip()
            consent = args.get('consent', False)
            if not spk:
                return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': '缺少 name 参数。'}, is_error=True)
            if not isinstance(consent, bool):
                return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': 'consent 必须为布尔值。'}, is_error=True)
            if consent is not True:
                return _tool_response({'ok': False, 'error': 'consent_required', 'reply': '注册必须获得本人明确同意（consent=true）。'}, is_error=True)

            raw_paths = args.get('audio_paths')
            raw_b64 = args.get('audio_base64_list')
            has_paths = bool(raw_paths)
            has_b64 = bool(raw_b64)
            if has_paths == has_b64:
                return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': '必须且只能提供 audio_paths 或 audio_base64_list 之一。'}, is_error=True)

            if has_paths:
                if not self.audio_roots:
                    return _tool_response({
                        'ok': False,
                        'error': 'path_not_allowed',
                        'reply': '未配置 audio_roots 安全边界，拒绝路径型音频入参。',
                    }, is_error=True)
                if not isinstance(raw_paths, list):
                    return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': 'audio_paths 必须为字符串列表。'}, is_error=True)
                quoted = ' '.join(shlex.quote(str(p)) for p in raw_paths)
                with self._ephemeral_roots([]):
                    out = self.agent.handle(f'注册 {shlex.quote(spk)} 本人同意 {quoted}')
                return _tool_response(out, is_error=not out['ok'])

            if not isinstance(raw_b64, list):
                return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': 'audio_base64_list 必须为字符串列表。'}, is_error=True)
            session_ids = args.get('session_ids')
            if session_ids is not None:
                if not isinstance(session_ids, list) or len(session_ids) != len(raw_b64):
                    return _tool_response({
                        'ok': False,
                        'error': 'invalid_session_ids',
                        'reply': 'session_ids 长度必须与 audio_base64_list 一致。',
                    }, is_error=True)
                for sid in session_ids:
                    if not isinstance(sid, str) or not _SESSION_ID_RE.match(sid):
                        return _tool_response({
                            'ok': False,
                            'error': 'invalid_session_ids',
                            'reply': f'非法的 session_id：{sid!r}（仅允许字母、数字、下划线和短横线）。',
                        }, is_error=True)

            with TemporaryDirectory(prefix='vpmcp_enr_') as tmp:
                tmp_root = Path(tmp)
                decoded_paths: list[str] = []
                for idx, item in enumerate(raw_b64):
                    subdir = tmp_root / (f'sess_{session_ids[idx]}' if session_ids is not None else 'session_0')
                    try:
                        p = _decode_base64_audio(item, subdir, f'take_{idx}')
                    except ValueError as exc:
                        return _tool_response({'ok': False, 'error': 'invalid_base64_audio', 'reply': str(exc)}, is_error=True)
                    decoded_paths.append(str(p))
                quoted = ' '.join(shlex.quote(p) for p in decoded_paths)
                with self._ephemeral_roots([tmp_root]):
                    out = self.agent.handle(f'注册 {shlex.quote(spk)} 本人同意 {quoted}')
                return _tool_response(out, is_error=not out['ok'])

        if name == 'voiceprint_simulate':
            if self.agent.synthesizer is None:
                return _tool_response({
                    'ok': False,
                    'error': 'synthesizer_disabled',
                    'reply': '当前 MCP 服务未启用声音模拟合成器。',
                }, is_error=True)

            raw_spk = args.get('name')
            if not isinstance(raw_spk, str) or not raw_spk.strip():
                return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': '缺少合法的 name 参数。'}, is_error=True)
            spk = raw_spk.strip()

            consent = args.get('consent', False)
            if not isinstance(consent, bool):
                return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': 'consent 必须为布尔值。'}, is_error=True)
            if consent is not True:
                return _tool_response({'ok': False, 'error': 'consent_required', 'reply': '声音模拟必须获得明确授权（consent=true）。'}, is_error=True)

            reference_text = args.get('reference_text')
            if reference_text is not None and not isinstance(reference_text, str):
                return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': 'reference_text 必须为字符串。'}, is_error=True)

            max_attempts = args.get('max_attempts')
            if max_attempts is not None:
                if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1 or max_attempts > 5:
                    return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': 'max_attempts 必须为 1 到 5 之间的整数。'}, is_error=True)

            include_b64 = args.get('include_audio_base64', False)
            if not isinstance(include_b64, bool):
                return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': 'include_audio_base64 必须为布尔值。'}, is_error=True)

            has_src_path = 'source_audio_path' in args and args.get('source_audio_path') is not None
            has_src_b64 = 'source_audio_base64' in args and args.get('source_audio_base64') is not None
            has_text = 'text' in args and args.get('text') is not None
            if int(has_src_path) + int(has_src_b64) + int(has_text) != 1:
                return _tool_response({
                    'ok': False,
                    'error': 'invalid_arguments',
                    'reply': '必须且只能提供 source_audio_path、source_audio_base64 或 text 之一。',
                }, is_error=True)

            with TemporaryDirectory(prefix='vpmcp_sim_') as tmp:
                tmp_root = Path(tmp)
                ref_path, ref_extra, ref_err = self._resolve_single_audio(
                    args,
                    tmp_root / 'ref',
                    path_key='reference_audio_path',
                    b64_key='reference_audio_base64',
                )
                if ref_err is not None or ref_path is None:
                    return _tool_response(ref_err or {'ok': False, 'error': 'invalid_arguments'}, is_error=True)

                extra_roots = list(ref_extra)
                if has_text:
                    if not isinstance(args['text'], str) or not args['text'].strip():
                        return _tool_response({'ok': False, 'error': 'invalid_arguments', 'reply': 'text 必须为非空字符串。'}, is_error=True)
                    sim_mode = 'text'
                    src_path = None
                    text_val = args['text']
                else:
                    sim_mode = 'audio'
                    src_path, src_extra, src_err = self._resolve_single_audio(
                        args,
                        tmp_root / 'src',
                        path_key='source_audio_path',
                        b64_key='source_audio_base64',
                    )
                    if src_err is not None or src_path is None:
                        return _tool_response(src_err or {'ok': False, 'error': 'invalid_arguments'}, is_error=True)
                    extra_roots.extend(src_extra)
                    text_val = None

                with self._ephemeral_roots(extra_roots):
                    out, _ = self.agent._run_simulate_core(
                        spk,
                        consent=True,
                        ref_path=ref_path,
                        mode=sim_mode,
                        src_path=src_path,
                        text=text_val,
                        reference_text=reference_text,
                        max_attempts=max_attempts,
                    )
                self.agent.store.audit(out['tool'], out['reply'])
                out['model'] = self.agent.embedder.describe()
                if out['ok'] and include_b64:
                    out_p = Path(out['result']['output_path'])
                    raw_out = out_p.read_bytes()
                    if len(raw_out) > MAX_BASE64_BYTES:
                        return _tool_response({'ok': False, 'error': 'audio_too_large', 'reply': '生成音频超出 base64 返回大小上限。'}, is_error=True)
                    out['result']['audio_base64'] = base64.b64encode(raw_out).decode('ascii')
                return _tool_response(out, is_error=not out['ok'])

        return _tool_response({'ok': False, 'error': 'unknown_tool', 'reply': f'未知工具：{name}'}, is_error=True)

    def handle_request(self, message: dict) -> dict | None:
        if not isinstance(message, dict):
            return _jsonrpc_err(None, -32600, 'Invalid JSON-RPC 2.0 request')
        raw_id = message.get('id')
        safe_id = raw_id if isinstance(raw_id, (str, int, type(None))) and not isinstance(raw_id, bool) else None
        if message.get('jsonrpc') != '2.0':
            return _jsonrpc_err(safe_id, -32600, 'Invalid JSON-RPC 2.0 request')

        method = message.get('method')
        req_id = safe_id

        # Notifications have no 'id' and must produce no response.
        if 'id' not in message:
            return None

        raw_params = message.get('params')
        if raw_params is None:
            params: dict = {}
        elif isinstance(raw_params, dict):
            params = raw_params
        else:
            return _jsonrpc_err(req_id, -32602, 'Invalid params: must be an object')

        if method == 'initialize':
            return _jsonrpc_ok(req_id, {
                'protocolVersion': PROTOCOL_VERSION,
                'serverInfo': {'name': SERVER_NAME, 'version': SERVER_VERSION},
                'capabilities': {'tools': {}, 'resources': {}},
            })

        if method == 'ping':
            return _jsonrpc_ok(req_id, {})

        if method == 'tools/list':
            with self._lock:
                tools = [
                    t for t in _TOOL_SCHEMAS
                    if self.agent.synthesizer is not None or t['name'] != 'voiceprint_simulate'
                ]
            return _jsonrpc_ok(req_id, {'tools': tools})

        if method == 'tools/call':
            tool_name = params.get('name')
            tool_args = params.get('arguments') or {}
            if not isinstance(tool_name, str) or not tool_name:
                return _jsonrpc_err(req_id, -32602, 'Missing tool name')
            with self._lock:
                res = self._call_tool_locked(tool_name, tool_args)
            return _jsonrpc_ok(req_id, res)

        if method == 'resources/list':
            return _jsonrpc_ok(req_id, {'resources': list(_RESOURCES)})

        if method == 'resources/read':
            uri = params.get('uri')
            with self._lock:
                if uri == 'voiceprint://speakers':
                    names = self.agent.store.names()
                    body = json.dumps({'count': len(names), 'speakers': names}, ensure_ascii=False)
                elif uri == 'voiceprint://health':
                    body = json.dumps(self._health_dict(), ensure_ascii=False)
                else:
                    return _jsonrpc_err(req_id, -32602, f'Unknown resource URI: {uri}')
            return _jsonrpc_ok(req_id, {
                'contents': [{'uri': uri, 'mimeType': 'application/json', 'text': body}],
            })

        return _jsonrpc_err(req_id, -32601, f'Method not found: {method}')

    def serve_streams(self, in_stream, out_stream) -> None:
        for raw_line in in_stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                resp = _jsonrpc_err(None, -32700, 'Parse error')
            else:
                try:
                    resp = self.handle_request(msg)
                except Exception as exc:
                    logging.getLogger(SERVER_NAME).exception('Unhandled MCP request error')
                    raw_id = msg.get('id') if isinstance(msg, dict) else None
                    safe_id = raw_id if isinstance(raw_id, (str, int, type(None))) and not isinstance(raw_id, bool) else None
                    resp = _jsonrpc_err(safe_id, -32603, f'Internal error: {type(exc).__name__}')
            if resp is not None:
                out_stream.write(json.dumps(resp, ensure_ascii=False) + '\n')
                out_stream.flush()


def build_default_server(
    data_dir: str | Path,
    embedder,
    *,
    audio_roots: list[str | Path] | None = None,
    detail: bool = False,
    spoof_guard: bool = True,
    read_only: bool = False,
    synthesizer: VoiceSynthesizer | None = None,
    max_sim_attempts: int = 3,
) -> VoiceprintMCPServer:
    """Construct a VoiceprintMCPServer with safe defaults (detail=False, spoof_guard=True)."""
    detector = SpreadSpectrumWatermarkDetector() if spoof_guard else None
    resolved_roots = [Path(r).resolve() for r in audio_roots] if audio_roots else None
    agent = VoiceAgent(
        data_dir,
        embedder,
        detail=detail,
        audio_roots=resolved_roots,
        spoof_detector=detector,
        synthesizer=synthesizer,
        max_sim_attempts=max_sim_attempts,
    )
    return VoiceprintMCPServer(agent, audio_roots=resolved_roots, read_only=read_only)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Voiceprint Agent JSON-RPC 2.0 stdio MCP Server')
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--embedder', default='vpnet', choices=['stats', 'vpnet', 'ecapa'])
    parser.add_argument('--audio-roots', nargs='*', default=None)
    parser.add_argument('--detail', action='store_true')
    parser.add_argument('--no-spoof-guard', action='store_true')
    parser.add_argument('--read-only', action='store_true')
    parser.add_argument('--synthesizer', default='acoustic', choices=['acoustic', 'neural_subprocess', 'none'])
    parser.add_argument('--neural-cmd', default=None)
    parser.add_argument('--neural-timeout', type=float, default=120.0)
    parser.add_argument('--max-sim-attempts', type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='[%(levelname)s] %(message)s')
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    env_roots = os.environ.get('VOICEPRINT_MCP_AUDIO_ROOTS')
    roots = args.audio_roots
    if roots is None and env_roots:
        roots = [p for p in env_roots.split(os.pathsep) if p.strip()]

    from voiceprint.embed import load_embedder

    embedder = load_embedder(args.embedder)
    synthesizer = load_synthesizer(
        args.synthesizer,
        neural_cmd=args.neural_cmd,
        timeout_s=args.neural_timeout,
    )
    server = build_default_server(
        args.data_dir,
        embedder,
        audio_roots=roots,
        detail=args.detail,
        spoof_guard=not args.no_spoof_guard,
        read_only=args.read_only,
        synthesizer=synthesizer,
        max_sim_attempts=args.max_sim_attempts,
    )
    try:
        server.serve_streams(sys.stdin, sys.stdout)
    finally:
        server.agent.close()


if __name__ == '__main__':
    main()

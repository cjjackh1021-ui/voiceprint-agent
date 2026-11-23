"""Interactive voiceprint agent.

    python -m agent.cli
    python -m agent.cli --embedder vpnet --data agent_data --show-scores
"""
import argparse
from pathlib import Path

from agent.service import VoiceAgent
from voiceprint.embed import load_embedder
from voiceprint.spoof import SpreadSpectrumWatermarkDetector
from voiceprint.synth import load_synthesizer

ROOT = Path(__file__).resolve().parents[1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='声纹注册、验证、辨认与可控声音模拟')
    parser.add_argument('--embedder', choices=['auto', 'ecapa', 'vpnet', 'stats'], default='auto')
    parser.add_argument('--data', default=str(ROOT / 'agent_data'))
    parser.add_argument('--show-scores', action='store_true',
                        help='在回复里显示精确分数和阈值（本机研究用；对外服务不要开）')
    parser.add_argument('--audio-root', action='append', default=None,
                        help='只允许读取这个文件夹里的录音，可重复指定；不指定则不限制本机路径')
    parser.add_argument('--spoof-guard', action='store_true',
                        help='在验证/辨认/注册入口启用 16-bit 溯源水印防伪拦截')
    parser.add_argument('--synthesizer', choices=['acoustic', 'neural_subprocess', 'none'], default='acoustic',
                        help='声音模拟合成后端（默认 zero-dep acoustic；none 关闭模拟）')
    parser.add_argument('--neural-cmd', default=None,
                        help='neural_subprocess 外部命令行，例如 \'"D:/voice confirm/.venv_tts/Scripts/python.exe" plugins/neural_tts_worker.py --engine f5_tts\'')
    parser.add_argument('--neural-timeout', type=float, default=120.0,
                        help='外部神经克隆子进程超时秒数（默认 120.0）')
    parser.add_argument('--max-sim-attempts', type=int, default=3,
                        help='声音模拟 Self-Refine 最大自检重试轮次（1..5）')
    return parser


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # auto: ECAPA when its files are already on disk, otherwise VPNet.
    embedder = load_embedder(args.embedder)
    spoof_detector = SpreadSpectrumWatermarkDetector() if args.spoof_guard else None
    synthesizer = load_synthesizer(
        args.synthesizer,
        neural_cmd=args.neural_cmd,
        timeout_s=args.neural_timeout,
    )
    agent = VoiceAgent(
        args.data,
        embedder,
        detail=args.show_scores,
        audio_roots=args.audio_root,
        spoof_detector=spoof_detector,
        synthesizer=synthesizer,
        max_sim_attempts=args.max_sim_attempts,
    )
    print(f'声纹 agent，嵌入模型 {embedder.name}。输入“帮助”查看命令，输入“退出”结束。')
    try:
        while True:
            try:
                line = input('> ')
            except EOFError:
                break
            if line.strip() in ('退出', 'exit', 'quit'):
                break
            print(agent.handle(line)['reply'])
    finally:
        agent.close()


if __name__ == '__main__':
    main()

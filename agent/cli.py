"""Interactive voiceprint agent.

    python -m agent.cli
    python -m agent.cli --embedder vpnet --data agent_data --show-scores
"""
import argparse
from pathlib import Path

from agent.service import VoiceAgent
from voiceprint.embed import load_embedder

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='声纹注册、验证和辨认')
    parser.add_argument('--embedder', choices=['auto', 'ecapa', 'vpnet', 'stats'], default='auto')
    parser.add_argument('--data', default=str(ROOT / 'agent_data'))
    parser.add_argument('--show-scores', action='store_true',
                        help='在回复里显示精确分数和阈值（本机研究用；对外服务不要开）')
    parser.add_argument('--audio-root', action='append', default=None,
                        help='只允许读取这个文件夹里的录音，可重复指定；不指定则不限制本机路径')
    args = parser.parse_args()
    # auto: ECAPA when its files are already on disk, otherwise VPNet.
    embedder = load_embedder(args.embedder)
    agent = VoiceAgent(args.data, embedder, detail=args.show_scores, audio_roots=args.audio_root)
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

"""
QNetwork の forward 挙動を検証するテスト。

任意の入力に対して有限な出力ベクトルを返すか、学習中に NaN が混入しないかをチェックする。
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from agent.network.q_network import set_network  # noqa: E402


def test_qnetwork_forward_shape_and_finiteness() -> None:
    """3×3 盤に対する QNetwork の出力形状と有限性を確認する。"""
    device = torch.device("cpu")
    policy_net, _ = set_network(board_side=3, num_team_values=2, device=device)

    board = torch.randn(4, 1, 3, 3, device=device)  # バッチサイズ 4
    teams = torch.zeros(4, dtype=torch.long, device=device)

    output = policy_net(board, teams)

    assert output.shape == (4, 9)
    assert torch.isfinite(output).all()

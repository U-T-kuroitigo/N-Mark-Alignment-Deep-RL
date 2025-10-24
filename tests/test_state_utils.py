"""
state_utils の変換関数を検証するテスト。

盤面リストから Tensor への変換が期待通りの shape/dtype/device になるか確認する。
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from agent.utils.state_utils import make_state_tensor, make_tensor  # noqa: E402


def test_make_tensor_shape_and_dtype() -> None:
    """make_tensor が (1, board_side, board_side) の float32 Tensor を返すことを確認する。"""
    board = [0, 1, -1, 1]
    tensor = make_tensor(board, board_side=2)

    assert tensor.shape == (1, 2, 2)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.tensor([[[0.0, 1.0], [-1.0, 1.0]]]))


class DummyEnv:
    """簡易的に盤面を返すダミー環境。"""

    def __init__(self, board):
        self._board = board

    def get_board(self):
        return self._board


def test_make_state_tensor_moves_to_device() -> None:
    """make_state_tensor が指定された device に Tensor を移動させることを確認する。"""
    board = [0, 0, 1, -1]
    env = DummyEnv(board)
    device = torch.device("cpu")

    tensor = make_state_tensor(env, board_side=2, device=device)

    assert tensor.device == device
    assert tensor.shape == (1, 2, 2)
    assert torch.allclose(tensor, torch.tensor([[[0.0, 0.0], [1.0, -1.0]]]))

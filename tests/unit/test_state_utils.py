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

from agent.utils.state_utils import (  # noqa: E402
    StateTensor,
    build_board_tensor,
    build_state_from_env,
    build_valid_action_mask,
    extract_valid_actions,
    make_state_tensor,
    make_tensor,
)


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


def test_build_board_tensor_respects_dtype_and_device() -> None:
    """build_board_tensor が dtype/device 指定を反映することを確認する。"""
    board = [0, 1, 2, 3]
    tensor = build_board_tensor(
        board, board_side=2, dtype=torch.float64, device=torch.device("cpu")
    )

    assert tensor.shape == (1, 2, 2)
    assert tensor.dtype == torch.float64
    assert tensor.device.type == "cpu"
    assert torch.allclose(
        tensor, torch.tensor([[[0.0, 1.0], [2.0, 3.0]]], dtype=torch.float64)
    )


def test_build_valid_action_mask_and_extract_valid_actions() -> None:
    """空きマスのマスク生成とインデックス抽出が正しく機能することを確認する。"""
    board = [0, -1, 1, -1]
    mask = build_valid_action_mask(board, empty_value=-1, device=torch.device("cpu"))

    assert mask.dtype == torch.bool
    assert mask.shape == (4,)
    assert mask.tolist() == [False, True, False, True]
    assert extract_valid_actions(mask) == [1, 3]


def test_build_state_from_env_returns_dataclass() -> None:
    """build_state_from_env が StateTensor を返し、各属性が期待通りになることを確認する。"""
    board = [-1, 0, -1, 1]
    env = DummyEnv(board)
    state = build_state_from_env(
        env,
        board_side=2,
        device=torch.device("cpu"),
        empty_value=-1,
        team_value=2,
    )

    assert isinstance(state, StateTensor)
    assert state.board_tensor.shape == (1, 2, 2)
    assert state.team_tensor is not None
    assert torch.equal(state.team_tensor, torch.tensor([2], dtype=torch.long))
    assert state.valid_actions == [0, 2]
    assert state.valid_action_mask.tolist() == [True, False, True, False]
    assert state.board_side == 2
    assert state.empty_value == -1

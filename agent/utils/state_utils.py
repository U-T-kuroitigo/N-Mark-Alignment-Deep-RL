"""
状態変換系ユーティリティ
DQNや他の強化学習エージェントで共通して使用される
盤面の状態をPyTorchのテンソルへ変換する補助関数を提供
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass
class StateTensor:
    """
    XAI 連携向けに盤面テンソルと関連情報を束ねるデータ構造。

    Attributes:
        board_tensor: 形状 (1, board_side, board_side) の盤面テンソル。
        team_tensor: 手番チーム値を保持した (1,) 形状の long テンソル。
        valid_action_mask: 有効手が True となる一次元ブールテンソル。
        valid_actions: 有効手インデックスのリスト表現。
        board_side: 盤面一辺の長さ。
        empty_value: 空きマスを示す整数値。
    """

    board_tensor: torch.Tensor
    team_tensor: Optional[torch.Tensor]
    valid_action_mask: torch.Tensor
    valid_actions: List[int]
    board_side: int
    empty_value: int


def build_board_tensor(
    board: Sequence[int],
    board_side: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    一次元配列の盤面情報を PyTorch テンソルへ変換する純粋関数。

    Args:
        board: 盤面の一次元配列。
        board_side: 盤面一辺のサイズ。
        dtype: 生成するテンソルの dtype。
        device: 配置先デバイス。

    Returns:
        torch.Tensor: 形状 (1, board_side, board_side) のテンソル。
    """

    tensor = torch.as_tensor(board, dtype=dtype, device=device)
    return tensor.view(1, board_side, board_side)


def build_team_tensor(
    team_value: int, *, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    手番チーム値を long Tensor に変換する。

    Args:
        team_value: 手番を表す整数値。
        device: 配置先デバイス。

    Returns:
        torch.Tensor: 形状 (1,) の long テンソル。
    """

    return torch.tensor([team_value], dtype=torch.long, device=device)


def build_valid_action_mask(
    board: Sequence[int],
    *,
    empty_value: int = -1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    盤面から「空きマスのみ True」なブールマスクを生成する。

    Args:
        board: 盤面の一次元配列。
        empty_value: 空きマスを示す値。
        device: 配置先デバイス。

    Returns:
        torch.Tensor: 1 次元のブールテンソル。
    """

    flat_board = torch.as_tensor(board, dtype=torch.long, device=device)
    return flat_board == empty_value


def extract_valid_actions(valid_action_mask: torch.Tensor) -> List[int]:
    """
    有効手マスクからインデックスのリストを抽出する。

    Args:
        valid_action_mask: `build_valid_action_mask` が返したブールテンソル。

    Returns:
        list[int]: True になっている位置のインデックス一覧。
    """

    return torch.nonzero(valid_action_mask, as_tuple=False).view(-1).tolist()


def build_state_from_env(
    env,
    board_side: int,
    device: torch.device,
    *,
    empty_value: int = -1,
    team_value: Optional[int] = None,
) -> StateTensor:
    """
    環境オブジェクトから盤面テンソル・手番情報・有効手マスクをまとめて生成する。

    Args:
        env: `get_board()` を実装した環境。
        board_side: 盤面一辺のサイズ。
        device: 配置先デバイス。
        empty_value: 空きマスを表す値。
        team_value: 手番チーム値。不要な場合は None。

    Returns:
        StateTensor: 盤面テンソルと関連する情報をまとめたデータ構造。
    """

    board = list(env.get_board())
    board_tensor = build_board_tensor(board, board_side, device=device)
    team_tensor = (
        build_team_tensor(team_value, device=device) if team_value is not None else None
    )
    valid_action_mask = build_valid_action_mask(
        board, empty_value=empty_value, device=device
    )
    valid_actions = extract_valid_actions(valid_action_mask)
    return StateTensor(
        board_tensor=board_tensor,
        team_tensor=team_tensor,
        valid_action_mask=valid_action_mask,
        valid_actions=valid_actions,
        board_side=board_side,
        empty_value=empty_value,
    )


def make_tensor(board, board_side):
    """
    盤面情報（list[int]）をPyTorchのテンソルに変換する。

    Args:
        board (list[int]): 一次元の盤面配列
        board_side (int): 盤面の一辺のサイズ

    Returns:
        torch.Tensor: 形状 (1, board_side, board_side) のテンソル
    """
    return build_board_tensor(board, board_side)


def make_state_tensor(env, board_side, device):
    """
    環境から現在の盤面を取得し、テンソル形式に整形する。

    Args:
        env (object): 盤面状態を持つ環境オブジェクト
        board_side (int): 盤面の一辺のサイズ
        device (torch.device): 使用するPyTorchデバイス

    Returns:
        torch.Tensor: 現在の盤面を示すテンソル
    """
    board = env.get_board()
    return build_board_tensor(board, board_side, device=device)

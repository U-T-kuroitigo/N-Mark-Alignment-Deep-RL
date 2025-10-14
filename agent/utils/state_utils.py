"""
状態変換系ユーティリティ
DQNや他の強化学習エージェントで共通して使用される
盤面の状態をPyTorchのテンソルへ変換する補助関数を提供
"""

import torch


def make_tensor(board, board_side):
    """
    盤面情報（list[int]）をPyTorchのテンソルに変換する。

    Args:
        board (list[int]): 一次元の盤面配列
        board_side (int): 盤面の一辺のサイズ

    Returns:
        torch.Tensor: 形状 (1, board_side, board_side) のテンソル
    """
    state = torch.tensor(board, dtype=torch.float32).view(1, board_side, board_side)
    return state


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
    return make_tensor(board, board_side).to(device)

"""
ReplayBuffer の挙動を検証するテスト。

容量超過時の上書き、append/clear、sample の返却サイズなどを確認する。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.buffer.replay_buffer import ReplayBuffer  # noqa: E402


def test_append_and_len_updates_until_capacity() -> None:
    """append 時に長さが容量まで増え、その後は上限値を保つことを確認する。"""
    buffer = ReplayBuffer(capacity=3)
    for i in range(5):
        buffer.append({"id": i})
    assert len(buffer) == 3
    # 最後の 3 件が保持されている想定
    assert sorted(item["id"] for item in buffer.buffer) == [2, 3, 4]


def test_sample_returns_requested_size() -> None:
    """sample が指定件数の遷移を返すことを確認する。"""
    random.seed(0)
    buffer = ReplayBuffer(capacity=5)
    for i in range(5):
        buffer.append({"id": i})

    batch = buffer.sample(batch_size=3)
    assert len(batch) == 3
    assert all("id" in item for item in batch)


def test_clear_resets_buffer_and_position() -> None:
    """clear でバッファと書き込み位置がリセットされることを確認する。"""
    buffer = ReplayBuffer(capacity=2)
    buffer.append({"id": 1})
    buffer.append({"id": 2})
    assert len(buffer) == 2

    buffer.clear()

    assert len(buffer) == 0
    buffer.append({"id": 3})
    assert buffer.buffer[0]["id"] == 3

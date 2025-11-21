"""
フックで受け渡す共通ペイロード定義。
XAI 用フックが増えても同じ構造でやり取りできるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class MatchResultPayload:
    """
    総当たり評価の 1 マッチ分の結果を保持するペイロード。
    """

    agent_a_name: str
    agent_b_name: str
    agent_a_icon: str
    agent_b_icon: str
    win: float
    lose: float
    draw: float
    episodes: int

    def to_dict(self) -> Dict[str, Any]:
        """
        dict 形式に変換して返す。

        Returns:
            Dict[str, Any]: 辞書化したペイロード。
        """

        return asdict(self)

import random
from typing import List, Dict


class ReplayBuffer:
    """
    強化学習における遷移情報を格納・サンプリングするバッファ。
    DQNなどのオフポリシー学習アルゴリズムで使用される。
    各遷移は dict 形式で管理され、XAIや多チーム対応に適した柔軟な設計。
    """

    def __init__(self, capacity: int):
        """
        ReplayBuffer の初期化。

        Args:
            capacity (int): バッファに保持する最大エントリ数
        """
        self.capacity = capacity
        self.buffer: List[Dict] = []
        self.position = 0

    def append(self, transition: Dict) -> None:
        """
        遷移情報（状態・行動・報酬など）をバッファに追加。

        Args:
            transition (dict): 1ステップ分の遷移情報を表す辞書
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Dict]:
        """
        ランダムに指定数の遷移情報をサンプリングして返す。

        Args:
            batch_size (int): サンプル数

        Returns:
            List[Dict]: サンプリングされた遷移情報のリスト
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """
        バッファに現在格納されている遷移数を返す。
        """
        return len(self.buffer)

    def clear(self) -> None:
        """
        バッファを空にリセットし、書き込み位置を先頭に戻す。
        """
        self.buffer.clear()
        self.position = 0

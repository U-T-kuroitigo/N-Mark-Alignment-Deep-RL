import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    盤面状態とチーム値を入力として、各マスに対応するQ値を出力するネットワーク。
    盤面状態をCNNで特徴抽出し、チーム値と統合してMLPでQ値を出力する。
    """

    def __init__(self, board_side: int, num_team_values: int, embedding_dim: int = 16):
        """
        Qネットワークの初期化。

        Args:
            board_side (int): 盤面の一辺の長さ（例：9なら9x9）
            num_team_values (int): チーム値の種類数（例：2チームなら2）
            embedding_dim (int): チーム値の埋め込み次元
        """
        super().__init__()
        self.board_side = board_side
        self.embedding_dim = embedding_dim

        # 盤面入力（1チャネル）をCNNで処理
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # チーム値を埋め込みベクトルに変換
        self.team_embedding = nn.Embedding(num_team_values + 1, embedding_dim)

        # 特徴量ベクトルの次元数
        conv_output_dim = 64 * board_side * board_side
        total_feature_dim = conv_output_dim + embedding_dim

        # Q値出力層（各マスに対応するQ値）
        self.fc1 = nn.Linear(total_feature_dim, 512)
        self.fc2 = nn.Linear(512, board_side * board_side)

    def forward(
        self, board_tensor: torch.Tensor, team_value_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Qネットワークの順伝播処理。

        この関数は、盤面状態テンソルとチーム値を入力として受け取り、
        各マスに対応するQ値のベクトルを出力する。
        - 盤面状態はCNNで空間特徴を抽出
        - チーム値は埋め込みベクトルに変換
        - 両者を統合してMLPで最終的なQ値を出力

        Args:
            board_tensor (Tensor): 盤面状態のテンソル（B, 1, S, S）
            team_value_tensor (Tensor): チーム値のテンソル（B,）

        Returns:
            Tensor: 各マスのQ値ベクトル（B, S*S）
        """
        # --- CNN部（盤面特徴抽出） ---
        # conv1: 入力1チャネル → 32チャネルの特徴マップ
        x = F.relu(self.conv1(board_tensor))  # (B, 32, S, S)
        # conv2: 32チャネル → 64チャネルの特徴マップ
        x = F.relu(self.conv2(x))  # (B, 64, S, S)

        # flatten: 畳み込み結果を全結合層用に1次元に変換
        x = x.view(x.size(0), -1)  # (B, 64 * S * S)

        # --- チーム値埋め込みベクトル ---
        # チーム値（整数）を意味的なベクトルに変換
        team_embed = self.team_embedding(team_value_tensor)  # (B, embedding_dim)

        # --- 特徴統合 & Q値出力部（MLP） ---
        # 盤面特徴とチーム情報を結合
        combined = torch.cat([x, team_embed], dim=1)  # (B, total_feature_dim)

        # 隠れ層を経由してQ値ベクトルを出力（各マスの行動価値）
        x = F.relu(self.fc1(combined))  # (B, 512)
        q_values = self.fc2(x)  # (B, S*S)

        return q_values


def set_network(
    board_side: int, num_team_values: int, device: torch.device
) -> tuple[QNetwork, QNetwork]:
    policy_net = QNetwork(board_side, num_team_values).to(device)
    target_net = QNetwork(board_side, num_team_values).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    return policy_net, target_net

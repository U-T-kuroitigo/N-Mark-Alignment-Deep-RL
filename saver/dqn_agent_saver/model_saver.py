import os
import torch
from agent.utils.model_version_utils import make_version_dir_and_filename
from saver.dqn_agent_saver.meta_info_saver import MetaInfoSaver
from saver.dqn_agent_saver.replay_buffer_saver import ReplayBufferSaver
from agent.buffer.replay_buffer import ReplayBuffer
from agent.dqn.dqn_agent import DQN_Agent
import csv
from datetime import datetime


class ModelSaver:
    """
    モデルの保存および関連情報（メタデータ、リプレイバッファなど）の保存を管理するクラス。
    各モデルタイプ毎にバージョンディレクトリを生成し、
    重みファイル、メタ情報、リプレイバッファを一貫して管理します。
    """

    def __init__(self, save_dir: str = "agent_model"):
        """
        保存先のルートディレクトリを初期化する。

        Args:
            save_dir (str): モデルを保存するルートディレクトリのパス
        """
        self.save_dir = save_dir
        self.meta_saver = MetaInfoSaver()
        self.replay_saver = ReplayBufferSaver()

    def save(
        self,
        agent: DQN_Agent,
    ) -> str:
        """
        DQN_Agentインスタンスからモデル、メタデータ、リプレイバッファを保存する。
        モデルタイプのディレクトリ配下にバージョンディレクトリを作成し、
        ファイルを格納します。

        Args:
            agent (DQN_Agent): 保存対象のエージェントインスタンス

        Returns:
            str: 保存されたモデルファイルへのパス
        """
        # エージェントから必要な情報を取得
        policy_net = agent.policy_net
        metadata = agent.get_metadata()
        replay_buffer = metadata.get("learning_count", None)

        timestamp = datetime.now().strftime("%Y%m%d%H%M")

        # バージョンディレクトリおよびファイル名を生成
        model_type_dir, version_dir, base_filename = make_version_dir_and_filename(
            metadata, self.save_dir, timestamp
        )
        os.makedirs(version_dir, exist_ok=True)

        # モデル重み (.pt) の保存
        model_path = os.path.join(version_dir, f"{base_filename}.pt")
        torch.save(policy_net.state_dict(), model_path)

        # ランダムNPCとの評価結果を保存（勝率, 負率, 引き分け率）
        win, lose, draw = agent.get_rate()
        eval_result_path = os.path.join(version_dir, f"{base_filename}_eval_result.txt")
        with open(eval_result_path, "w", encoding="utf-8") as f:
            f.write(
                f"ランダムNPC評価結果\n"
                f"勝率: {win:.1f}%\n"
                f"負率: {lose:.1f}%\n"
                f"引き分け率: {draw:.1f}%\n"
            )

        # 評価履歴（CSV）をバージョンディレクトリに追記保存

        eval_history_path = os.path.join(model_type_dir, "evaluation_history.csv")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 学習回数を取得（agent側で保持していることを想定）
        learning_count = getattr(agent, "learning_count", -1)

        write_header = not os.path.exists(eval_history_path)
        with open(eval_history_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "filename",
                    "episode",
                    "win_rate",
                    "lose_rate",
                    "draw_rate",
                    "timestamp",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "filename": base_filename,
                    "episode": learning_count,
                    "win_rate": f"{win:.1f}",
                    "lose_rate": f"{lose:.1f}",
                    "draw_rate": f"{draw:.1f}",
                    "timestamp": timestamp,
                }
            )

        # メタ情報の保存
        self.meta_saver.save(agent, self.save_dir)

        # ReplayBuffer の保存があれば実行
        if replay_buffer is not None:
            self.replay_saver.save(agent, self.save_dir)

        return model_path

    def load(
        self,
        model_path: str,
        agent: DQN_Agent,
        load_replay: bool = False,
    ) -> DQN_Agent:
        """
        保存されたモデルとメタデータを読み込み、
        DQN_Agentインスタンスに復元する。

        Args:
            model_path (str): 保存されたモデルファイルのパス
            agent (DQN_Agent): 学習済みモデルを読み込む DQN_Agent インスタンス
            load_replay (bool, optional): True の場合、リプレイバッファを復元する

        Returns:
            DQN_Agent: モデルおよび関連情報を復元したエージェントインスタンス
        """
        # ネットワーク重みを読み込む
        agent.policy_net.load_state_dict(torch.load(model_path))

        # バージョンディレクトリと base_filename を取得
        version_dir = os.path.dirname(model_path)
        base_filename = os.path.splitext(os.path.basename(model_path))[0]

        # メタデータを読み込み、エージェントに反映
        self.meta_saver.load(agent, version_dir, base_filename)

        # ReplayBufferを必要に応じて復元
        if load_replay:
            agent.replay_buffer = self.replay_saver.load(
                agent, version_dir, base_filename
            )

        return agent

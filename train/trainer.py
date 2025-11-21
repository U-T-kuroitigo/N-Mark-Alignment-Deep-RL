"""
trainer.py

DQNエージェントをN目並べ環境で訓練する学習ループを管理するクラス
"""

import logging
import time
from typing import Callable, Dict, List, Optional

# 環境、エージェント、モデル保存用クラスを型指定
from N_Mark_Alignment_env import N_Mark_Alignment_Env
from agent.dqn.dqn_agent import DQN_Agent
from saver.dqn_agent_saver.model_saver import ModelSaver
from agent.N_Mark_Alignment_random_npc import N_Mark_Alignment_random_npc as Random_NPC
from utils.logging_utils import LoggingConfig, build_logger

# 定数定義 - ハイパーパラメータ等のデフォルト値
DEFAULT_TOTAL_EPISODES: int = 1000  # 総エピソード数デフォルト
DEFAULT_LEARN_ITERATIONS_PER_EPISODE: int = 1  # 学習反復回数デフォルト
DEFAULT_SAVE_FREQUENCY: int = 10  # モデル保存頻度デフォルト
DEFAULT_LOG_FREQUENCY: int = 10  # ログ出力頻度デフォルト


class Trainer:
    """
    DQNエージェントをN目並べ環境で訓練する学習ループを管理するクラス
    """

    def __init__(
        self,
        env: N_Mark_Alignment_Env,
        agent: DQN_Agent,
        model_saver: ModelSaver,
        config: Optional[Dict[str, int]] = None,
        eval_interval: int = 100,
        eval_episodes: int = 20,
        logger: Optional[logging.Logger] = None,
        episode_hooks: Optional[List[Callable[[Dict[str, int]], None]]] = None,
    ) -> None:
        """
        Trainerの初期化。

        Args:
            env (N_Mark_Alignment_Env): ゲーム環境インスタンス。
            agent (DQN_Agent): DQNエージェントインスタンス。
            model_saver (ModelSaver): モデル保存処理を担うオブジェクト。
            config (Optional[Dict[str, int]]): 学習設定を含む辞書。
                total_episodes: 総エピソード数。
                learn_iterations_per_episode: 各エピソード後の学習反復回数。
                save_frequency: モデル保存頻度。
                log_frequency: ログ出力頻度。
            eval_interval (int): ランダムNPC評価を行うエピソード間隔。
            eval_episodes (int): 評価時に実施するテストゲーム数。
            logger (Optional[logging.Logger]): 進捗ログを出力するロガー。
            episode_hooks (Optional[List[Callable[[Dict[str, int]], None]]]): エピソード完了時に呼び出されるフック。
        """
        cfg: Dict[str, int] = config or {}

        # 定期評価用設定
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes

        # 既存メンバの初期化
        self.env = env
        self.agent = agent
        self.model_saver = model_saver
        self.total_episodes = cfg.get("total_episodes", DEFAULT_TOTAL_EPISODES)
        self.learn_iterations_per_episode = cfg.get(
            "learn_iterations_per_episode", DEFAULT_LEARN_ITERATIONS_PER_EPISODE
        )
        self.save_frequency = cfg.get("save_frequency", DEFAULT_SAVE_FREQUENCY)
        self.log_frequency = cfg.get("log_frequency", DEFAULT_LOG_FREQUENCY)

        # ログ関連の初期化
        self.logger = logger or build_logger(
            LoggingConfig(name=__name__ + ".trainer", level="INFO")
        )
        self.episode_hooks = episode_hooks or []

    def train(self) -> None:
        """
        全エピソードにわたる学習ループを実行。
        """
        start_time = time.time()
        for episode in range(1, self.total_episodes + 1):
            self.train_episode()
            elapsed = time.time() - start_time

            # ログ出力
            if episode % self.log_frequency == 0:
                hours, rem = divmod(elapsed, 3600)
                minutes, seconds = divmod(rem, 60)
                message = {
                    "episode": episode,
                    "total_episodes": self.total_episodes,
                    "elapsed_seconds": int(elapsed),
                    "elapsed_h": int(hours),
                    "elapsed_m": int(minutes),
                    "elapsed_s": int(seconds),
                }
                self.logger.info(
                    "[Episode %(episode)d/%(total_episodes)d] 経過時間: "
                    "%(elapsed_h)d時間%(elapsed_m)d分%(elapsed_s)d秒",
                    message,
                )

            # モデル保存
            if episode % self.save_frequency == 0:
                # ランダムNPCとの定期評価
                # 評価モード：学習OFF
                self.agent.set_learning(False)
                win, lose, draw = self.evaluate_random_opponent(self.eval_episodes)
                # 評価結果をロギング
                eval_context = {
                    "episode": episode,
                    "evaluation_games": self.eval_episodes,
                    "win_rate": win,
                    "lose_rate": lose,
                    "draw_rate": draw,
                }
                self.logger.info(
                    "ランダムNPC評価: 勝率 %(win_rate).1f%%, "
                    "負率 %(lose_rate).1f%%, 引き分け率 %(draw_rate).1f%%",
                    eval_context,
                )
                for hook in self.episode_hooks:
                    hook(eval_context)
                # 学習再開
                self.agent.set_learning(True)

                self.model_saver.save(self.agent)
                self.agent.epsilon_reset()

    def train_episode(self) -> float:
        """
        単一エピソードの学習を実行。

        Returns:
            float: エピソード合計報酬。
        """
        # 環境リセット
        state = self.env.reset()
        done = False

        # step() を使った1手単位のループ
        while not done:
            (
                action,
                prev_board,
                next_board,
                actor_team_value,
                next_team_value,
                done,
                result_value,
            ) = self.env.step()

            # 中間報酬・遷移情報をエージェントに登録
            self.agent.append_continue_result(
                action,
                prev_board,
                actor_team_value,
                next_team_value,
            )

            # 状態更新
            state = next_board

        # 終局時の報酬合成・バッファ登録
        self.agent.append_finish_result(action, state, result_value)

        # ネットワーク更新
        for _ in range(self.learn_iterations_per_episode):
            self.agent.learn()

    def evaluate_random_opponent(self, num_games: int) -> tuple[float, float, float]:
        """
        ランダムNPC と対戦して勝率・敗率・引き分け率を計測する。

        Args:
            num_games (int): 対戦ゲーム数

        Returns:
            tuple[float, float, float]: (win_rate, lose_rate, draw_rate)
        """
        # ランダムNPCを生成（学習モードOFF）
        random_npc = Random_NPC("R", 1)
        player_list = [self.agent, random_npc]

        # テスト環境を再構築
        test_env = N_Mark_Alignment_Env(
            board_side=self.env.BOARD_SIDE,
            reward_line=self.env.REWARD_LINE,
            player_list=player_list,
        )

        # 統計リセット
        self.agent.reset_rate()
        test_env.reset_game_rate()

        # 自動対戦実行
        for i in range(num_games):
            _, board_icon = test_env.auto_play()
            # ゲーム数の5分の1ごとに盤面を表示
            # if (i + 1) % (num_games // 5) == 0:
            #     print(f"Game {i + 1}/{num_games} の結果:")
            #     test_env.print_board(board_icon)

        # 結果取得
        win, lose, draw = self.agent.get_rate()
        return win, lose, draw

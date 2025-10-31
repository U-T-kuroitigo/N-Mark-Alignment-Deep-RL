import random
import copy
from typing import Any
import agent.model.N_Mark_Alignment_agent_model as model
import N_Mark_Alignment_env as ENV


class PlayableAgent(model.Agent_Model):
    def __init__(self, player_icon: Any, player_value: int) -> None:
        super().__init__(player_icon, player_value)

    # 学習するかどうかを設定する関数
    def set_learning(self, learning):
        # 人間は学習しない
        pass

    def get_action(self, env: ENV.N_Mark_Alignment_Env) -> int:
        """
        人間プレイヤーに現在の盤面を表示し、次のアクションを選ばせる関数。

        盤面は見やすいフォーマットで出力され、空いているマスの番号を使って入力を受け付ける。
        不正な入力やすでに置かれているマスへの入力は再入力を求める。

        Args:
            env (N_Mark_Alignment_Env): 現在の環境インスタンス（盤面情報等の取得に使用）

        Returns:
            int: プレイヤーが選んだ次のアクション（マスのインデックス）
        """

        def decide_action(board_data: list[list[str]]) -> int:
            # 見やすく整形した盤面を出力
            for row in board_data:
                print(" | ".join(row))
            try:
                return int(
                    input(f"どこに置きますか？ (0〜{env.get_board_size() - 1}): ")
                )
            except ValueError:
                return -1

        def check_next_action(state: list[int], next_action: int) -> bool:
            return (
                0 <= next_action < len(state) and state[next_action] == self.empty_value
            )

        state = env.get_board()
        board_data = env.get_rendered_board_data()

        while True:
            next_action = decide_action(board_data)
            if not check_next_action(state, next_action):
                print("そのマスには置けません。別のマスを選んでください。")
            else:
                break

        self.prev_action = next_action
        return next_action

    # 渡されたaction, result_value, stateをもとに結果を追加する関数
    def append_continue_result(
        self, action: int, state: list[int], actor_team_value: int, next_team_value: int
    ) -> None:
        # 特に何もしない
        pass

    # 渡されたresult_valueをもとに結果を追加する関数
    def append_finish_result(
        self, action: int, state: list[int], result_value: int
    ) -> None:
        self.game_count += 1
        if result_value == self.my_team_value:
            print("あなたの勝ちです！")
            self.win += 1
        elif result_value == self.empty_value:
            print("引き分けです。")
            self.draw += 1
        else:
            print("あなたの負けです。")
            self.lose += 1

    # エージェントを保存する関数
    def save_agent(self):
        # npcは保存しない
        pass

    # 学習回数を取得する関数
    def get_learning_count(self):
        # npcは学習できない
        return 0

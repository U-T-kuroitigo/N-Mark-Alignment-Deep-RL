"""
DQNベースのエージェントクラス。
中間報酬をエピソード終了時に正規化して学習する構成。
状態・報酬評価はユーティリティモジュールに分離し、責務を明確化。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from agent.buffer.replay_buffer import ReplayBuffer
from agent.utils.state_utils import make_tensor, make_state_tensor
from agent.utils.reward_utils import (
    is_win,
    is_draw,
    is_loss,
    calculate_intermediate_reward,
    normalize_intermediate_rewards,
)
import agent.model.N_Mark_Alignment_agent_model as model
from typing import Dict, Any
from collections import defaultdict


class DQN_Agent(model.Agent_Model):
    # 型注釈による属性宣言
    policy_net: nn.Module
    target_net: nn.Module
    device: torch.device

    """
    Deep Q-Network による五目並べエージェント。
    状態とチーム値を入力とし、Q値ベクトルを出力するネットワークにより行動を決定。
    勝敗に応じた報酬を与え、Replay Buffer に蓄積して学習。
    """

    AGENT_NAME = "DQN-Agent"
    WIN_POINT = 1.0
    DRAW_POINT = 0.2
    LOSE_POINT = -2
    GAMMA = 0.9
    LEARNING_RATE = 1e-4
    EPSILON_START = 0.5
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.99965
    BATCH_SIZE = 32
    BUFFER_SIZE = 50000

    # 中間報酬の係数（必要に応じて外部化可能）
    REACH_CREATED_REWARD = 0.2
    REACH_BLOCKED_REWARD = 0.8

    def __init__(
        self, player_icon, player_value, learning, policy_net, target_net, device
    ):
        """
        DQNエージェントの初期化処理。

        Args:
            player_icon (str): プレイヤーの表示記号
            player_value (int): プレイヤーを一意に表す値
            learning (bool): 学習モードかどうか
            policy_net (nn.Module): メインのQネットワーク
            target_net (nn.Module): ターゲットネットワーク
            device (torch.device): 使用するデバイス
        """
        super().__init__(player_icon, player_value)
        self.set_learning(learning)
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE)
        self.epsilon_reset()

        # エピソード中の一時的な記録用バッファ
        self.episode_buffer = []
        # 経験を蓄積するためのリプレイバッファ
        self.replay_buffer = ReplayBuffer(self.BUFFER_SIZE)
        # 学習回数カウント
        self.learning_count = 0

    def epsilon_reset(self):
        """
        ε-greedy法のε値を初期化。
        学習開始時に呼び出すことで、探索率をリセットする。
        """
        self.epsilon = self.EPSILON_START

    def set_learning(self, learning):
        """
        学習の有効/無効を設定する。

        Args:
            learning (bool): Trueなら学習モード、Falseなら推論のみ
        """
        self.learning = learning

    def game_init(self):
        """
        ゲーム開始時に呼ばれる初期化処理。
        通常のrate初期化に加えて、エピソードバッファも初期化。
        """
        super().game_init()
        # 1試合分の行動記録バッファを初期化
        self.episode_buffer = []

    def get_action(self, env):
        """
        環境から現在の状態を取得し、Qネットワークを用いて行動を決定する。
        ε-greedy法によりランダム行動も選択する。

        Args:
            env: 現在の環境オブジェクト

        Returns:
            int: 選択された行動（マスのインデックス）
        """
        # 状態テンソルを生成
        state = make_state_tensor(env, self.board_side, self.device)
        team_value_tensor = torch.tensor([self.my_team_value], dtype=torch.long).to(
            self.device
        )

        # ε-greedyによるランダム行動選択
        if self.learning and random.random() < self.epsilon:
            return random.randint(0, self.board_side**2 - 1)

        # ネットワークによりQ値を予測
        with torch.no_grad():
            q_values = self.policy_net(state.unsqueeze(0), team_value_tensor).squeeze()

        # 空いているマスだけを対象に最大Q値の行動を選択
        board = env.get_board()
        valid_actions = [i for i, v in enumerate(board) if v == self.empty_value]
        q_values[[i for i in range(len(q_values)) if i not in valid_actions]] = -float(
            "inf"
        )
        action = torch.argmax(q_values).item()
        return action

    def append_continue_result(self, action, state, actor_team_value, next_team_value):
        """
        各ターンでの中間結果をepisode_bufferに記録。
        全プレイヤーの行動を保存するが、学習は自身の行動のみ対象とする。

        次の手番者のチーム値は引数next_team_valueで受け取り、終了フラグは常にFalse。

        Args:
            action (int): 実行された行動
            state (List[int]): 行動前の状態
            actor_team_value (int): 行動したプレイヤーのチーム値
            next_team_value (int): 次に行動するプレイヤーのチーム値
        """
        # 終了フラグは常にFalse
        done = False

        # 次状態をコピーして更新
        next_state = state.copy()
        next_state[action] = actor_team_value

        # 中間報酬を計算
        reward = calculate_intermediate_reward(
            state_before=state,
            action=action,
            team_value=actor_team_value,
            board_side=self.board_side,
            reward_line=self.reward_line,
            reach_created_reward=self.REACH_CREATED_REWARD,
            reach_blocked_reward=self.REACH_BLOCKED_REWARD,
        )

        state_tensor = make_tensor(state, self.board_side)
        next_state_tensor = make_tensor(next_state, self.board_side)

        # エピソードバッファに辞書形式で保存
        self.episode_buffer.append(
            {
                "board_state": state_tensor,
                "actor_team_value": actor_team_value,
                "action": action,
                "reward": reward,
                "next_board_state": next_state_tensor,
                "next_team_value": next_team_value,
                "done": done,
            }
        )

    def append_finish_result(self, action, state, result_value):
        """
        試合終了時に呼ばれる関数。
        エピソード中に集めた中間報酬を正規化し、全視点分の終局報酬を割引付きで加算。
        各視点ごとにReplayBufferに追加し、その都度学習を実行する。

        Args:
            action (int): 最後に実行された行動
            state (List[int]): 最終状態（盤面のリスト）
            result_value (int): 勝者のチーム値、または引き分けを示す値
        """
        # 最終状態テンソルを作成
        final_state_tensor = make_tensor(state, self.board_side)
        done = True

        # 最終ステップも一旦 episode_buffer に追加（reward=0.0）
        self.episode_buffer.append(
            {
                "board_state": final_state_tensor,
                "actor_team_value": self.my_team_value,
                "action": action,
                "reward": 0.0,
                "next_board_state": None,
                "next_team_value": None,
                "done": done,
            }
        )

        # 中間報酬をチームごとに個別正規化

        # チームごとにトランジションを分ける
        team_transitions = defaultdict(list)
        for trans in self.episode_buffer:
            team_transitions[trans["actor_team_value"]].append(trans)
        # 各チームの中間報酬を正規化して書き戻し
        for team, trans_list in team_transitions.items():
            rewards = [t["reward"] for t in trans_list]
            normalized_rewards = normalize_intermediate_rewards(rewards)
            for t, r in zip(trans_list, normalized_rewards):
                t["reward"] = r

        # 1試合分の全ステップ情報を保持
        episode = list(self.episode_buffer)
        self.episode_buffer.clear()

        # 終局報酬をチームごとに計算
        final_rewards = {}
        for team in self.team_value_list:
            if is_win(result_value, team):
                final_rewards[team] = self.WIN_POINT
            elif is_draw(result_value, self.empty_value):
                final_rewards[team] = self.DRAW_POINT
            else:
                final_rewards[team] = self.LOSE_POINT

        # 各視点ごとにReplayBufferへ追加し、その後学習
        T = len(episode)
        for team in self.team_value_list:
            # その視点の終局報酬
            final_reward = final_rewards[team]
            for idx, trans in enumerate(episode):
                if trans["actor_team_value"] != team:
                    continue
                discount = self.GAMMA ** (T - idx - 1)
                total_reward = trans["reward"] + discount * final_reward
                self.replay_buffer.append(
                    {
                        "board_state": trans["board_state"],
                        "actor_team_value": team,
                        "action": trans["action"],
                        "reward": total_reward,
                        "next_board_state": trans["next_board_state"],
                        "next_team_value": trans["next_team_value"],
                        "done": trans["done"],
                    }
                )

        if self.learning:
            self.learn()
            # 学習カウントは試合単位で加算
            self.learning_count += 1
            self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)

        # 統計情報の更新
        self.game_count += 1
        if is_win(result_value, self.my_team_value):
            self.win += 1
        elif is_draw(result_value, self.empty_value):
            self.draw += 1
        else:
            self.lose += 1

    def learn(self):
        """
        ReplayBufferからサンプリングしてQ学習を行う。
        Double DQN形式に基づいた更新式を使用。
        """
        # 各チームごとに学習を実行
        for team_value in self.team_value_list:
            # バッファから当該チームの遷移のみを抽出
            team_batch = [
                t
                for t in self.replay_buffer.buffer
                if t["actor_team_value"] == team_value
            ]
            # サンプル数がバッチサイズに満たない場合はスキップ（学習の安定化のため）
            if len(team_batch) < self.BATCH_SIZE:
                continue
            # 経験の多様性を保つためランダムサンプリング
            sampled = random.sample(team_batch, self.BATCH_SIZE)

            # サンプルから各種テンソルを生成
            states = torch.stack([t["board_state"] for t in sampled]).to(self.device)
            team_values = torch.tensor(
                [t["actor_team_value"] for t in sampled], dtype=torch.long
            ).to(self.device)
            actions = torch.tensor([t["action"] for t in sampled]).to(self.device)
            rewards = torch.tensor([t["reward"] for t in sampled]).to(self.device)
            # 終端状態を除いて次状態をスタック
            next_states = torch.stack(
                [
                    t["next_board_state"]
                    for t in sampled
                    if t["next_board_state"] is not None
                ]
            ).to(self.device)
            # 次ステップのチームID（Noneの場合はダミー値0）
            next_team_values = torch.tensor(
                [
                    t["next_team_value"] if t["next_team_value"] is not None else 0
                    for t in sampled
                ],
                dtype=torch.long,
            ).to(self.device)
            # エピソード終了フラグをfloatで取得（1:終端, 0:非終端）
            dones = torch.tensor([t["done"] for t in sampled], dtype=torch.float32).to(
                self.device
            )

            # ポリシーネットワークによる現在Q値の推定
            q_values = self.policy_net(states, team_values)
            # 実際に選択された行動のQ値のみ抽出
            q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # 非終端サンプルのマスクを作成
            non_final_mask = torch.tensor(
                [t["next_board_state"] is not None for t in sampled],
                dtype=torch.bool,
                device=self.device,
            )
            # 次状態Q値ベクトル初期化
            next_q_values = torch.zeros(len(sampled), device=self.device)
            if non_final_mask.any():
                # マスクで非終端サンプルのインデックスを抽出
                non_final_indices = non_final_mask.nonzero(as_tuple=False).squeeze()
                non_final_team_vals = next_team_values[non_final_indices]
                # ターゲットネットワークで次状態のQ値を推定し最大値を取得
                next_q_non_final = (
                    self.target_net(next_states, non_final_team_vals).max(1)[0].detach()
                )
                # 該当インデックスに次状態Q値を設定
                next_q_values[non_final_indices] = next_q_non_final

            # Double DQNの期待Q値: 報酬 + 割引率 * 次状態Q値
            expected_q = rewards + (1 - dones) * self.GAMMA * next_q_values

            # 損失計算 (MSE) と逆伝播/パラメータ更新
            loss = nn.MSELoss()(q_a, expected_q)  # Q値回帰の損失関数
            self.optimizer.zero_grad()  # 勾配をゼロクリア
            loss.backward()  # 逆伝播で勾配計算
            self.optimizer.step()  # パラメータ更新

    def get_learning_count(self):
        """
        学習回数を返す。

        Returns:
            int: 累積の学習回数
        """
        return self.learning_count

    def get_metadata(self) -> Dict[str, Any]:
        """
        モデル保存時に含めるメタ情報を辞書形式で返す。

        Returns:
            Dict[str, Any]: メタデータ（学習率や報酬係数など）
        """
        return {
            "agent_name": self.AGENT_NAME,
            "board_side": self.board_side,
            "reward_line": self.reward_line,
            "team_count": len(self.team_value_list),
            "learning_count": self.learning_count,
            "gamma": self.GAMMA,
            "learning_rate": self.LEARNING_RATE,
            "epsilon_start": self.EPSILON_START,
            "epsilon_min": self.EPSILON_MIN,
            "epsilon_decay": self.EPSILON_DECAY,
            "batch_size": self.BATCH_SIZE,
            "buffer_size": self.BUFFER_SIZE,
            "reach_created_reward": self.REACH_CREATED_REWARD,
            "reach_blocked_reward": self.REACH_BLOCKED_REWARD,
        }

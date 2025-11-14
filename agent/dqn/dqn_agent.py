"""
DQNベースのエージェントクラス。
中間報酬をエピソード終了時に正規化して学習する構成。
状態・報酬評価はユーティリティモジュールに分離し、責務を明確化。
"""

import logging
import random
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import agent.model.N_Mark_Alignment_agent_model as model
from agent.buffer.replay_buffer import ReplayBuffer
from agent.utils.state_utils import build_state_from_env, make_tensor
from agent.utils.reward_utils import (
    is_win,
    is_draw,
    is_loss,
    calculate_intermediate_reward,
    normalize_intermediate_rewards,
)

logger = logging.getLogger(__name__)

ACTION_SELECTED_HOOK = "action_selected"
TRANSITION_RECORDED_HOOK = "transition_recorded"
EPISODE_FINISHED_HOOK = "episode_finished"


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
        self._hooks: Dict[str, list[Callable[[Dict[str, Any]], None]]] = defaultdict(
            list
        )
        self.last_action_context: Optional[Dict[str, Any]] = None
        self.last_transition: Optional[Dict[str, Any]] = None

    def epsilon_reset(self):
        """
        ε-greedy法のε値を初期化。
        学習開始時に呼び出すことで、探索率をリセットする。
        """
        self.epsilon = self.EPSILON_START

    def register_hook(
        self, event: str, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        XAI など外部処理で利用するフックを登録する。
        指定イベントが発火すると callback(payload) が呼び出される。

        Args:
            event: イベント名（例: ACTION_SELECTED_HOOK）。
            callback: コンテキスト辞書を受け取るコールバック。
        """

        if not isinstance(event, str):
            raise TypeError("event must be a str")
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._hooks[event].append(callback)

    def remove_hook(
        self, event: str, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        登録済みフックを解除する。存在しない場合は何もしない。

        Args:
            event: イベント名。
            callback: 解除したいコールバック。
        """

        callbacks = self._hooks.get(event)
        if not callbacks:
            return
        try:
            callbacks.remove(callback)
        except ValueError:
            return
        if not callbacks:
            self._hooks.pop(event, None)

    def clear_hooks(self, event: Optional[str] = None) -> None:
        """
        登録済みフックを削除する。

        Args:
            event: 指定するとそのイベントのみ削除。None の場合は全イベントを削除。
        """

        if event is None:
            self._hooks.clear()
        else:
            self._hooks.pop(event, None)

    def _emit_hook(self, event: str, payload: Dict[str, Any]) -> None:
        """
        登録済みフックを呼び出す内部ユーティリティ。例外はログのみ記録する。

        Args:
            event: イベント名。
            payload: コールバックに渡すコンテキスト辞書。
        """

        for callback in list(self._hooks.get(event, [])):
            try:
                callback(payload)
            except Exception:  # pragma: no cover
                logger.exception("hook '%s' raised an exception", event)

    def _record_transition(self, transition: Dict[str, Any]) -> None:
        """
        エピソード内の遷移をバッファへ記録し、必要に応じてフックへ通知する。

        Args:
            transition: エピソード中に記録した遷移辞書。
        """

        self.episode_buffer.append(transition)
        self.last_transition = transition
        self._emit_hook(TRANSITION_RECORDED_HOOK, transition)

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
        self.last_action_context = None
        self.last_transition = None

    def get_action(self, env):
        """
        環境から現在の盤面を取得し、Qネットワークと ε-greedy 法で行動を選択する。
        選択内容はフックに通知され、XAI から参照できるよう情報を保持する。

        Args:
            env: 現在の環境オブジェクト

        Returns:
            int: 選択された行動（マスのインデックス）
        """

        # 盤面テンソルと手番情報、有効手マスクをまとめて生成し、推論・ログの双方で再利用する。
        state_info = build_state_from_env(
            env,
            self.board_side,
            self.device,
            empty_value=self.empty_value,
            team_value=self.my_team_value,
        )
        state_tensor = state_info.board_tensor
        state_batch = state_tensor.unsqueeze(0)
        team_value_tensor = state_info.team_tensor
        if team_value_tensor is None:
            team_value_tensor = torch.tensor(
                [self.my_team_value], dtype=torch.long, device=self.device
            )

        valid_actions = state_info.valid_actions

        with torch.no_grad():  # Q 値推論
            raw_q_values = self.policy_net(state_batch, team_value_tensor).squeeze(0)

        valid_action_mask = state_info.valid_action_mask.reshape_as(raw_q_values)
        masked_q_values = raw_q_values.masked_fill(
            ~valid_action_mask.bool(), -float("inf")
        )

        epsilon_sample = random.random() if self.learning else 1.0
        greedy_action = int(torch.argmax(masked_q_values).item())

        if self.learning and valid_actions and epsilon_sample < self.epsilon:
            action = int(random.choice(valid_actions))
            selection_mode = "exploration"
        else:
            action = greedy_action
            if action not in valid_actions and valid_actions:
                action = valid_actions[0]
            selection_mode = "policy" if valid_actions else "fallback"

        context = {
            "state_tensor": state_tensor.detach().cpu(),
            "team_value": self.my_team_value,
            "raw_q_values": raw_q_values.detach().cpu(),
            "masked_q_values": masked_q_values.detach().cpu(),
            "valid_actions": valid_actions,
            "valid_action_mask": valid_action_mask.detach().cpu(),
            "selected_action": action,
            "greedy_action": greedy_action,
            "selection_mode": selection_mode,
            "epsilon": self.epsilon,
            "epsilon_sample": epsilon_sample,
        }
        self.last_action_context = context
        self._emit_hook(ACTION_SELECTED_HOOK, context)

        self.prev_action = action
        return action

    def append_continue_result(self, action, state, actor_team_value, next_team_value):
        """
        ゲーム継続中の遷移を episode_buffer に記録する。

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

        transition = {
            "board_state": state_tensor,
            "actor_team_value": actor_team_value,
            "action": action,
            "reward": reward,
            "next_board_state": next_state_tensor,
            "next_team_value": next_team_value,
            "done": done,
            "state": state.copy(),
            "next_state": next_state,
        }
        self._record_transition(transition)

        if (
            self.last_action_context
            and self.last_action_context.get("selected_action") == action
        ):
            self.last_action_context["reward"] = reward
            self.last_action_context["next_state_tensor"] = (
                next_state_tensor.detach().cpu()
            )

    def append_finish_result(self, action, state, result_value):
        """
        エピソード終了時の処理を行い、報酬計算とリプレイバッファ更新を実施する。

        Args:
            action (int): 最後に実行された行動
            state (List[int]): 最終状態（盤面のリスト）
            result_value (int): 勝者のチーム値、または引き分けを示す値
        """

        final_state_tensor = make_tensor(state, self.board_side)
        terminal_transition = {
            "board_state": final_state_tensor,
            "actor_team_value": self.my_team_value,
            "action": action,
            "reward": 0.0,
            "next_board_state": None,
            "next_team_value": None,
            "done": True,
            "state": state.copy(),
            "next_state": None,
        }
        self._record_transition(terminal_transition)

        episode = list(self.episode_buffer)
        self.episode_buffer.clear()

        team_transitions = defaultdict(list)
        for trans in episode:
            team_transitions[trans["actor_team_value"]].append(trans)
        for trans_list in team_transitions.values():
            rewards = [t["reward"] for t in trans_list]
            normalized = normalize_intermediate_rewards(rewards)
            for t, r in zip(trans_list, normalized):
                t["reward"] = r

        final_rewards: Dict[int, float] = {}
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

        self._emit_hook(
            EPISODE_FINISHED_HOOK,
            {
                "result_value": result_value,
                "final_rewards": final_rewards,
                "episode_length": T,
                "episode_transitions": episode,
            },
        )

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

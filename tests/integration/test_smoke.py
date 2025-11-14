"""
現在の実装に対するスモークテスト。

主要コンポーネントを最小構成で動かし、リファクタリング前後で挙動が変わっていないか
検出することを目的とする。
"""

import random
import sys
from pathlib import Path

import pytest

# プロジェクトルートを import パスに追加してローカルモジュールを解決
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

import N_Mark_Alignment_env as env_module
from agent.N_Mark_Alignment_random_npc import N_Mark_Alignment_random_npc
from agent.dqn.dqn_agent import DQN_Agent
from agent.network.q_network import set_network
from agent.utils import reward_utils
from agent.model.N_Mark_Alignment_agent_model import Agent_Model
from evaluate.round_robin_match_runner import RoundRobinMatchRunner
from saver.dqn_agent_saver.model_saver import ModelSaver
from train.trainer import Trainer


class DummyAgent(Agent_Model):
    """シンプルな固定手で着手するテスト専用エージェント。"""

    def __init__(
        self, player_icon: str, player_value: int, moves: list[int] | None = None
    ) -> None:
        """
        Args:
            player_icon: 盤面表示用のアイコン。
            player_value: チーム ID。
            moves: あらかじめ打つマスのリスト。足りなくなった場合は空きマスを探索する。
        """
        super().__init__(player_icon, player_value)
        self.moves = moves or []
        self.move_idx = 0
        self.learning = False
        self.player_name = f"Dummy-{player_icon}"

    def set_learning(self, learning: bool) -> None:
        """ダミーエージェントは学習しないため、フラグのみ保持。"""
        self.learning = learning

    def get_action(self, env: env_module.N_Mark_Alignment_Env) -> int:
        """事前に指定された手、もしくは最初の空きマスを選んで着手する。"""
        if self.move_idx < len(self.moves):
            action = self.moves[self.move_idx]
            self.move_idx += 1
        else:
            # 予備として最初の空いているマスを選択
            board = env.get_board()
            action = next(i for i, v in enumerate(board) if v == self.empty_value)
        self.prev_action = action
        return action

    def append_continue_result(
        self,
        action: int,
        state: list[int],
        actor_team_value: int,
        next_team_value: int,
    ) -> None:
        """継続結果は管理しない（テスト用途のため no-op）。"""

    def append_finish_result(
        self, action: int, state: list[int], result_value: int
    ) -> None:
        """ゲーム完了時に試合数だけ加算する。"""
        self.game_count += 1

    def get_learning_count(self) -> int:
        """学習回数を持たないテスト用エージェントのため常に 0 を返す。"""
        return 0


def test_environment_step_updates_board() -> None:
    """環境の step() が盤面を更新し、ターン情報を返すことを確認。"""
    agent_a = DummyAgent("A", 0, moves=[0])
    agent_b = DummyAgent("B", 1, moves=[1])

    env = env_module.N_Mark_Alignment_Env(
        board_side=3, reward_line=3, player_list=[agent_a, agent_b]
    )
    env.player_turn_list = [agent_a, agent_b]
    env.this_turn = 0
    env.reset()

    action, prev_board, next_board, actor_team_value, _, _, _ = env.step()

    assert action == 0
    assert prev_board[action] == env.EMPTY
    assert next_board[action] == actor_team_value


def test_dqn_network_forward() -> None:
    """Q ネットワークがダミー入力に対して有限値のベクトルを返すことを確認。"""
    device = torch.device("cpu")
    policy_net, _ = set_network(board_side=3, num_team_values=2, device=device)
    board = torch.randn(1, 1, 3, 3, device=device)
    team = torch.zeros(1, dtype=torch.long, device=device)

    output = policy_net(board, team)

    assert output.shape == (1, 9)
    assert torch.isfinite(output).all()


def test_reward_utils_behaviour() -> None:
    """reward_utils の代表的な関数が期待通りの判定・値を返すことを確認。"""
    board_side = 3
    reward_line = 3
    team = 0

    # reach を作るケース
    pre_state_create = [team, -1, -1, -1, -1, -1, -1, -1, -1]
    action_create = 1
    assert reward_utils.is_reach_created(
        pre_state_create, action_create, team, board_side, reward_line
    )

    reward_create = reward_utils.calculate_intermediate_reward(
        pre_state_create,
        action_create,
        team,
        board_side,
        reward_line,
        reach_created_reward=0.2,
        reach_blocked_reward=0.3,
    )
    assert pytest.approx(reward_create) == 0.2

    # 相手の reach をブロックするケース
    opponent_state = [1, 1, -1, -1, -1, -1, -1, -1, -1]
    action_block = 2
    assert reward_utils.is_reach_blocked(
        opponent_state, action_block, team, board_side, reward_line
    )

    reward_block = reward_utils.calculate_intermediate_reward(
        opponent_state,
        action_block,
        team,
        board_side,
        reward_line,
        reach_created_reward=0.2,
        reach_blocked_reward=0.3,
    )
    assert pytest.approx(reward_block) == 0.3


def _create_dqn_agent(board_side: int = 3) -> DQN_Agent:
    """テストで使う最小構成の DQN_Agent を生成。"""
    device = torch.device("cpu")
    policy_net, target_net = set_network(
        board_side=board_side, num_team_values=2, device=device
    )
    agent = DQN_Agent("A", 0, True, policy_net, target_net, device)
    agent.player_name = "DQN-A"
    return agent


def test_trainer_smoke(tmp_path: Path) -> None:
    """小規模設定で Trainer.train() が例外なく完走するかを確認。"""
    random.seed(0)
    board_side = 3
    agent = _create_dqn_agent(board_side)
    npc = N_Mark_Alignment_random_npc("B", 1)

    env = env_module.N_Mark_Alignment_Env(
        board_side=board_side, reward_line=3, player_list=[agent, npc]
    )

    saver = ModelSaver(save_dir=str(tmp_path / "models"))
    trainer = Trainer(
        env=env,
        agent=agent,
        model_saver=saver,
        config={
            "total_episodes": 2,
            "learn_iterations_per_episode": 1,
            "save_frequency": 999,
            "log_frequency": 999,
        },
        eval_interval=999,
        eval_episodes=1,
    )

    trainer.train()


def test_model_saver_roundtrip(tmp_path: Path) -> None:
    """ModelSaver で保存したモデルが再ロード可能かを確認。"""
    device = torch.device("cpu")
    policy_net, target_net = set_network(board_side=3, num_team_values=2, device=device)
    agent = DQN_Agent("A", 0, True, policy_net, target_net, device)
    agent.board_side = 3
    agent.reward_line = 3
    agent.team_value_list = [0, 1]
    agent.learning_count = 5

    saver = ModelSaver(save_dir=str(tmp_path / "models"))
    artifacts = saver.save(agent)

    policy_net_2, target_net_2 = set_network(
        board_side=3, num_team_values=2, device=device
    )
    loaded_agent = DQN_Agent("A", 0, True, policy_net_2, target_net_2, device)
    loaded_agent.board_side = 3
    loaded_agent.reward_line = 3
    loaded_agent.team_value_list = [0, 1]

    saver.load(artifacts.model_path, loaded_agent)

    assert loaded_agent.board_side == 3
    assert all(
        torch.equal(p1, p2)
        for p1, p2 in zip(
            agent.policy_net.state_dict().values(),
            loaded_agent.policy_net.state_dict().values(),
        )
    )


def test_round_robin_runner(monkeypatch, tmp_path: Path) -> None:
    """RoundRobinMatchRunner が結果サマリを生成できるかのスモークテスト。"""
    agent_a = DummyAgent("A", 0, moves=[0, 4, 8])
    agent_b = DummyAgent("B", 1, moves=[1, 2, 5])

    env = env_module.N_Mark_Alignment_Env(
        board_side=3, reward_line=3, player_list=[agent_a, agent_b]
    )
    runner = RoundRobinMatchRunner(env, eval_episodes=1)

    monkeypatch.chdir(tmp_path)
    summary = runner.evaluate([agent_a, agent_b])

    assert len(summary) == 2
    assert {entry["agent_name"] for entry in summary} == {
        agent_a.player_name,
        agent_b.player_name,
    }

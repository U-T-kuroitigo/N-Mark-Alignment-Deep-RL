"""
ModelSaver / MetaInfoSaver / ReplayBufferSaver の往復を検証するテスト。

ダミーの DQN エージェントを保存→再ロードし、主要な情報が復元されることを確認する。
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from agent.dqn.dqn_agent import DQN_Agent  # noqa: E402
from agent.network.q_network import set_network  # noqa: E402
from saver.dqn_agent_saver.model_saver import ModelSaver  # noqa: E402


def _create_agent_with_replay(board_side: int = 3) -> DQN_Agent:
    """保存テスト用の DQN エージェントを生成し、リプレイバッファにダミー遷移を入れておく。"""
    device = torch.device("cpu")
    policy_net, target_net = set_network(
        board_side=board_side, num_team_values=2, device=device
    )
    agent = DQN_Agent("A", 0, True, policy_net, target_net, device)
    agent.player_name = "DQN-A"
    agent.board_side = board_side
    agent.reward_line = 3
    agent.team_value_list = [0, 1]
    agent.learning_count = 5

    # リプレイバッファにダミー遷移を追加
    agent.replay_buffer.append(
        {
            "board_state": torch.zeros(1, board_side, board_side),
            "actor_team_value": 0,
            "action": 0,
            "reward": 1.0,
            "next_board_state": torch.ones(1, board_side, board_side),
            "next_team_value": 1,
            "done": False,
        }
    )
    return agent


def _create_fresh_agent(board_side: int = 3) -> DQN_Agent:
    """ロード先に使用する新しいエージェントを用意する。"""
    device = torch.device("cpu")
    policy_net, target_net = set_network(
        board_side=board_side, num_team_values=2, device=device
    )
    return DQN_Agent("A", 0, True, policy_net, target_net, device)


def test_model_saver_roundtrip(tmp_path: Path) -> None:
    """
    ModelSaver.save → ModelSaver.load を行い、
    パラメータ・メタ情報・リプレイバッファが復元されることを確認する。
    """
    agent = _create_agent_with_replay()
    save_dir = tmp_path / "models"
    saver = ModelSaver(save_dir=str(save_dir))

    artifacts = saver.save(agent)

    # 保存物の存在を確認
    model_path = Path(artifacts.model_path)
    assert model_path.exists()
    meta_path = Path(artifacts.metadata_path)
    replay_path = Path(artifacts.replay_buffer_path)
    history_path = Path(artifacts.eval_history_path)
    assert meta_path.exists()
    assert replay_path.exists()
    assert history_path.exists()
    assert artifacts.version_paths.version_dir == str(model_path.parent)

    # 新しいエージェントに読み込み
    loaded_agent = _create_fresh_agent()
    saver.load(str(model_path), loaded_agent, load_replay=True)

    # メタ情報が復元されているか
    assert loaded_agent.board_side == agent.board_side
    assert loaded_agent.reward_line == agent.reward_line
    assert loaded_agent.learning_count == agent.learning_count

    # ネットワークパラメータが一致
    for original, loaded in zip(
        agent.policy_net.state_dict().values(),
        loaded_agent.policy_net.state_dict().values(),
    ):
        assert torch.equal(original, loaded)

    # リプレイバッファが復元されているか（保存時 1 件のみ）
    assert (
        len(loaded_agent.replay_buffer.buffer) == len(agent.replay_buffer.buffer) == 1
    )
    loaded_transition = loaded_agent.replay_buffer.buffer[0]
    assert loaded_transition["reward"] == agent.replay_buffer.buffer[0]["reward"]

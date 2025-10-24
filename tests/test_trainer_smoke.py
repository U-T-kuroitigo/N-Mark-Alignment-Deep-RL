"""
Trainer の最小構成スモークテスト。

極小の設定で `Trainer.train()` が例外なく完走し、モデル保存が行われるか確認する。
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from agent.N_Mark_Alignment_random_npc import N_Mark_Alignment_random_npc  # noqa: E402
from agent.dqn.dqn_agent import DQN_Agent  # noqa: E402
from agent.network.q_network import set_network  # noqa: E402
from saver.dqn_agent_saver.model_saver import ModelSaver  # noqa: E402
from train.trainer import Trainer  # noqa: E402
import N_Mark_Alignment_env as env_module  # noqa: E402


def _create_dqn_agent(board_side: int = 3) -> DQN_Agent:
    """テストで使用する最小構成の DQN エージェントを生成する。"""
    device = torch.device("cpu")
    policy_net, target_net = set_network(board_side=board_side, num_team_values=2, device=device)
    agent = DQN_Agent("A", 0, True, policy_net, target_net, device)
    agent.player_name = "DQN-A"
    return agent


def test_trainer_runs_and_saves_model(tmp_path: Path) -> None:
    """
    3×3 盤・エピソード数 3 の設定で Trainer が例外なく完走し、
    ModelSaver が保存ディレクトリを生成することを確認する。
    """
    board_side = 3
    agent = _create_dqn_agent(board_side)
    npc = N_Mark_Alignment_random_npc("B", 1)

    env = env_module.N_Mark_Alignment_Env(board_side=board_side, reward_line=3, player_list=[agent, npc])

    save_dir = tmp_path / "models"
    saver = ModelSaver(save_dir=str(save_dir))

    trainer = Trainer(
        env=env,
        agent=agent,
        model_saver=saver,
        config={
            "total_episodes": 3,
            "learn_iterations_per_episode": 1,
            "save_frequency": 2,
            "log_frequency": 999,
        },
        eval_interval=999,
        eval_episodes=1,
    )

    trainer.train()

    # save_frequency=2 のため最低 1 回はモデルが保存されているはず
    assert any(save_dir.rglob("*.pt")), "ModelSaver が .pt ファイルを出力していません。"

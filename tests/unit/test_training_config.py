"""
TrainingConfig の読み込みと辞書変換を確認するテスト。
"""

from pathlib import Path

from train.training_config import TrainingConfig


def test_training_config_from_yaml(tmp_path: Path) -> None:
    """YAML からの読み込みで値が反映されることを確認する。"""
    config_text = """
board_side: 4
reward_line: 3
num_team_values: 2
total_episodes: 200
learn_iterations_per_episode: 2
save_frequency: 10
log_frequency: 20
eval_episodes: 30
log_level: DEBUG
"""
    config_path = tmp_path / "train.yaml"
    config_path.write_text(config_text.strip(), encoding="utf-8")

    loaded = TrainingConfig.from_yaml(config_path)
    assert loaded.board_side == 4
    assert loaded.total_episodes == 200
    assert loaded.eval_episodes == 30
    assert loaded.log_level == "DEBUG"


def test_training_config_to_trainer_dict() -> None:
    """Trainer に渡す辞書が期待通りのキーを持つことを確認する。"""
    cfg = TrainingConfig(
        total_episodes=150,
        learn_iterations_per_episode=3,
        save_frequency=15,
        log_frequency=5,
    )
    trainer_dict = cfg.to_trainer_dict()
    assert trainer_dict["total_episodes"] == 150
    assert trainer_dict["learn_iterations_per_episode"] == 3
    assert trainer_dict["save_frequency"] == 15
    assert trainer_dict["log_frequency"] == 5

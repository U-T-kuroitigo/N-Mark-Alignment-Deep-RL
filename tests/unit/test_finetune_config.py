"""
FinetuneConfig の読み込みと補助メソッドを検証するテスト。
"""

from pathlib import Path

from train.finetune_config import FinetuneConfig


def test_finetune_config_from_yaml(tmp_path: Path) -> None:
    """YAML から設定を読み込み、値が反映されることを確認する。"""
    config_text = """
base_model_path: models/base.pt
player_settings:
  - type: base
    icon: A
    learning: true
  - type: npc
    icon: B
learning_count: 120
log_level: DEBUG
"""
    config_path = tmp_path / "finetune.yaml"
    config_path.write_text(config_text.strip(), encoding="utf-8")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    target_file = models_dir / "base.pt"
    target_file.write_text("", encoding="utf-8")

    cfg = FinetuneConfig.from_yaml(config_path)
    assert cfg.log_level == "DEBUG"
    assert cfg.learning_count == 120
    assert cfg.player_settings[0].type == "base"
    resolved = cfg.resolve_path(None)
    assert resolved == target_file.resolve()


def test_finetune_resolve_path_fallback(tmp_path: Path) -> None:
    """設定先にファイルが無い場合は CWD を基準に解決することを確認する。"""
    config_text = """
base_model_path: models/base.pt
player_settings:
  - type: base
    icon: A
    learning: true
"""
    config_path = tmp_path / "finetune.yaml"
    config_path.write_text(config_text.strip(), encoding="utf-8")

    cfg = FinetuneConfig.from_yaml(config_path)
    resolved = cfg.resolve_path(None)
    assert resolved == (Path.cwd() / "models/base.pt").resolve()


def test_finetune_trainer_config_defaults() -> None:
    """Trainer 設定辞書がデフォルト値で生成されることを確認する。"""
    cfg = FinetuneConfig()
    trainer_dict = cfg.trainer_config()
    assert trainer_dict["total_episodes"] == cfg.learning_count
    assert trainer_dict["save_frequency"] == max(1, cfg.learning_count // 20)

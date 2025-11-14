"""
pytest 用の共通フィクスチャ・マーカー設定。
テストディレクトリごとに unit / integration マーカーを自動付与する。
"""

from pathlib import Path


def pytest_collection_modifyitems(config, items):
    """
    収集されたテストのファイルパスを確認し、unit/integration マーカーを付与する。

    Args:
        config: pytest の設定オブジェクト。
        items: 収集済み TestItem のリスト。
    """

    project_root = Path(config.rootpath).resolve()
    unit_dir = project_root / "tests" / "unit"
    integration_dir = project_root / "tests" / "integration"

    for item in items:
        path = Path(item.fspath).resolve()
        if unit_dir in path.parents:
            item.add_marker("unit")
        elif integration_dir in path.parents:
            item.add_marker("integration")

#!/usr/bin/env python
"""
プロジェクト内の主要ディレクトリ（utils/agent/saver/train/evaluate）における
import 依存を可視化する簡易スクリプト。

Rule:
    下位レイヤー（数値が小さい）へは依存可、上位レイヤーへの依存は違反とする。
    例: evaluate(4) -> agent(1) は OK、agent(1) -> evaluate(4) は NG。

使い方:
    python tools/module_dependency_report.py
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# レイヤー定義（値が小さいほど下層）
LAYER_ORDER = {
    "utils": 0,
    "agent": 1,
    "env": 1,
    "saver": 2,
    "train": 3,
    "evaluate": 4,
}

# top-level ファイル名をレイヤーへ紐付けるための補助
SPECIAL_FILES = {
    "N_Mark_Alignment_env.py": "env",
}

TARGET_DIRS = {"agent", "utils", "saver", "train", "evaluate"}


def _layer_from_path(path: Path) -> str | None:
    rel = path.relative_to(PROJECT_ROOT)
    if rel.parts[0] in TARGET_DIRS:
        return rel.parts[0]
    if rel.parts[0] == "tests":
        return None
    if len(rel.parts) == 1 and rel.name in SPECIAL_FILES:
        return SPECIAL_FILES[rel.name]
    return None


def _layer_from_module(module: str | None) -> str | None:
    if not module:
        return None
    first = module.split(".")[0]
    if first in LAYER_ORDER:
        return first
    if module in SPECIAL_FILES:
        return SPECIAL_FILES[module]
    return None


def _iter_python_files() -> Iterable[Path]:
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part.startswith(".") for part in path.parts):
            continue
        if path.match("tests/tools/*"):
            continue
        if path.parts[0] == "tests":
            # テストコードは依存制約の対象外
            continue
        yield path


def analyze_dependencies() -> Tuple[Dict[str, Dict[str, int]], List[Tuple[str, str, Path]]]:
    adjacency: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    violations: List[Tuple[str, str, Path]] = []

    for py_file in _iter_python_files():
        importer_layer = _layer_from_path(py_file)
        if importer_layer is None:
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"))

        imports: List[str | None] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)

        for module in imports:
            importee_layer = _layer_from_module(module)
            if importee_layer is None or importee_layer == importer_layer:
                continue
            adjacency[importer_layer][importee_layer] += 1
            if LAYER_ORDER[importer_layer] < LAYER_ORDER[importee_layer]:
                violations.append((importer_layer, importee_layer, py_file.relative_to(PROJECT_ROOT)))

    return adjacency, violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Display module dependency matrix.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="結果を JSON 形式で出力する（CI 連携など向け）。",
    )
    args = parser.parse_args()

    adjacency, violations = analyze_dependencies()
    if args.json:
        import json

        print(
            json.dumps(
                {
                    "adjacency": adjacency,
                    "violations": [
                        {"importer": imp, "importee": dep, "file": str(path)}
                        for imp, dep, path in violations
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    print("=== 依存一覧 ===")
    for importer in sorted(adjacency, key=lambda k: LAYER_ORDER[k]):
        for importee, count in sorted(adjacency[importer].items(), key=lambda kv: LAYER_ORDER[kv[0]]):
            print(f"{importer:>10} -> {importee:<10} : {count} imports")

    if violations:
        print("\n=== 違反（下層→上層の依存）===")
        for importer, importee, path in violations:
            print(f"- {path}: {importer} -> {importee}")
    else:
        print("\n依存違反は検出されませんでした。")


if __name__ == "__main__":
    main()

"""
logging_utils.py

ロギング設定を集中管理するためのユーティリティ。
Trainer や評価スクリプトから共通の初期化処理を呼び出せるようにする。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LoggingConfig:
    """
    ロギング設定を表すデータクラス。

    Attributes:
        name (str): 生成するロガー名。
        level (str): ログレベル文字列（例: "INFO"）。
        log_file (Optional[Path]): ファイル出力先。None の場合は標準出力のみ。
        json_format (bool): True の場合、JSON 形式でログを整形する。
        propagate (bool): 親ロガーへ伝播させるかどうか。
    """

    name: str
    level: str = "INFO"
    log_file: Optional[Path] = None
    json_format: bool = False
    propagate: bool = False


class _JsonFormatter(logging.Formatter):
    """
    ログレコードを JSON 文字列へ変換するフォーマッタ。
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def build_logger(config: LoggingConfig) -> logging.Logger:
    """
    指定された設定を基にロガーを構築する。

    Args:
        config (LoggingConfig): ロガー構築用の設定。

    Returns:
        logging.Logger: 初期化済みロガー。
    """
    logger = logging.getLogger(config.name)
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    logger.propagate = config.propagate

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter: logging.Formatter
        if config.json_format:
            formatter = _JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if config.log_file:
            log_file_path = Path(config.log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

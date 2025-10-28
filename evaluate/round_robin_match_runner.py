from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
import logging
import csv

from agent.model.N_Mark_Alignment_agent_model import Agent_Model
from N_Mark_Alignment_env import N_Mark_Alignment_Env


class RoundRobinMatchRunner:
    def __init__(
        self,
        env: N_Mark_Alignment_Env,
        eval_episodes: int = 50,
        logger: Optional[logging.Logger] = None,
        result_dir: Optional[Path] = None,
        record_boards: bool = True,
        write_summary: bool = True,
        result_hooks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
    ):
        """
        総当たり戦形式でエージェント同士を評価するためのランナー。
        環境は1つを使いまわし、各試合でset_player()によりプレイヤーを差し替える。

        Args:
            env (N_Mark_Alignment_Env): 対戦に使用する環境（1インスタンスを使い回す）
            eval_episodes (int): 各対戦カードごとの試合回数
            logger (Optional[logging.Logger]): 進行状況を出力するロガー。
            result_dir (Optional[Path]): 評価結果を保存するディレクトリ（未指定時は既定パス）。
            record_boards (bool): 各試合の盤面ログを保存するかどうか。
            write_summary (bool): 集計結果を CSV に書き出すかどうか。
            result_hooks (Optional[List[Callable[[Dict[str, Any]], None]]]): 各試合結果を受け取るフック。
        """
        self.env = env
        self.eval_episodes = eval_episodes
        self.match_results: List[Dict[str, Any]] = []
        self.logger = logger or logging.getLogger(__name__ + ".round_robin")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.result_dir = Path(result_dir) if result_dir else None
        self.record_boards = record_boards
        self.write_summary = write_summary
        self.result_hooks = result_hooks or []

    def run_match(self, agent_a: Agent_Model, agent_b: Agent_Model) -> Dict[str, Any]:
        """
        agent_a vs agent_b の対戦を指定回数行い、結果を記録する。

        Args:
            agent_a (Agent_Model): プレイヤー1
            agent_b (Agent_Model): プレイヤー2

        Returns:
            dict: agent_a視点での win/lose/draw の統計結果
        """
        self.env.set_player([agent_a, agent_b])
        agent_a.reset_rate()
        agent_b.reset_rate()
        self.env.reset_game_rate()

        board_records = [] if self.record_boards else None

        for _ in range(self.eval_episodes):
            self.env.auto_play()
            if board_records is not None:
                board = self.env.get_board()
                board_records.append(board.copy())

        if board_records is not None:
            self._write_board_logs(agent_a, agent_b, board_records)

        win, lose, draw = agent_a.get_rate()

        result = {
            "agent_a_id": str(agent_a.get_agent_id()),
            "agent_b_id": str(agent_b.get_agent_id()),
            "agent_a_icon": agent_a.get_player_icon(),
            "agent_b_icon": agent_b.get_player_icon(),
            "agent_a_name": agent_a.player_name,
            "agent_b_name": agent_b.player_name,
            # "agent_a_learning_count": agent_a.get_learning_count(),
            # "agent_b_learning_count": agent_b.get_learning_count(),
            "win": win,
            "lose": lose,
            "draw": draw,
            "episodes": self.eval_episodes,
        }
        self.match_results.append(result)
        self.logger.info(
            "%s(%s) vs %s(%s): win=%.1f%% lose=%.1f%% draw=%.1f%%",
            agent_a.player_name,
            agent_a.get_player_icon(),
            agent_b.player_name,
            agent_b.get_player_icon(),
            win,
            lose,
            draw,
        )
        self._emit_result(result)
        return result

    def evaluate(self, agent_list: List[Agent_Model]) -> List[Dict[str, Any]]:
        """
        与えられたエージェント一覧で総当たり戦を実施。

        Args:
            agent_list (list): 評価対象のAgent_Modelインスタンス一覧

        Returns:
            list: 各対戦カードごとの結果のリスト
        """
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                agent_a = agent_list[i]
                agent_b = agent_list[j]
                self.logger.info(
                    "=== %s(%s) vs %s(%s) ===",
                    agent_a.player_name,
                    agent_a.get_player_icon(),
                    agent_b.player_name,
                    agent_b.get_player_icon(),
                )
                result = self.run_match(agent_a, agent_b)

        # 結果出力
        self.logger.info("=== 最終結果 ===")
        for r in self.match_results:
            self.logger.info(
                "%s(%s) vs %s(%s) → %s:win=%.1f%%, %s:win=%.1f%%, draw=%.1f%%",
                r["agent_a_name"],
                r["agent_a_icon"],
                r["agent_b_name"],
                r["agent_b_icon"],
                r["agent_a_name"],
                r["win"],
                r["agent_b_name"],
                r["lose"],
                r["draw"],
            )

        # 2) 統計集計（AvsB, AvsC… の両者分を足し合わせ）
        stats: Dict[str, Dict[str, float]] = {}
        match_counts: Dict[str, int] = {}

        for r in self.match_results:
            a = r["agent_a_name"]
            b = r["agent_b_name"]

            # a 側の成績加算
            stats.setdefault(a, {"win": 0.0, "lose": 0.0, "draw": 0.0})
            match_counts.setdefault(a, 0)
            stats[a]["win"] += r["win"]
            stats[a]["lose"] += r["lose"]
            stats[a]["draw"] += r["draw"]
            match_counts[a] += 1

            # b 側の成績加算（役割反転）
            stats.setdefault(b, {"win": 0.0, "lose": 0.0, "draw": 0.0})
            match_counts.setdefault(b, 0)
            stats[b]["win"] += r["lose"]
            stats[b]["lose"] += r["win"]
            stats[b]["draw"] += r["draw"]
            match_counts[b] += 1

        # 3) 統計リスト生成
        summary: List[Dict[str, Any]] = []
        for agent in agent_list:
            name = agent.player_name
            # マッチ数取得
            count = match_counts.get(name, 1)
            win_rate = stats[name]["win"] / count
            lose_rate = stats[name]["lose"] / count
            draw_rate = stats[name]["draw"] / count

            summary.append(
                {
                    "agent_name": name,
                    "agent_icon": agent.get_player_icon(),
                    "episode": agent.get_learning_count(),
                    "win_rate": round(win_rate, 3),
                    "lose_rate": round(lose_rate, 3),
                    "draw_rate": round(draw_rate, 3),
                }
            )

        if self.write_summary:
            self._write_summary_csv(summary)

        return summary

    def _emit_result(self, payload: Dict[str, Any]) -> None:
        """
        登録されたフックに対して試合結果を通知する。

        Args:
            payload (Dict[str, Any]): 試合結果を格納した辞書。
        """
        for hook in self.result_hooks:
            try:
                hook(payload)
            except Exception as exc:  # pragma: no cover - hook失敗はログのみ
                self.logger.exception("Result hook raised an exception: %s", exc)

    def _resolve_result_dir(self, subdir: Optional[str] = None) -> Path:
        """
        保存先ディレクトリを生成・返却する。

        Args:
            subdir (Optional[str]): 追加で連結するサブディレクトリ名。

        Returns:
            Path: 実際に使用するディレクトリパス。
        """
        player_count = max(len(self.env.get_player_list()), 1)
        base = self.result_dir or Path("evaluate") / "result"
        base = (
            Path(base) / f"{self.env.BOARD_SIDE}_{self.env.REWARD_LINE}_{player_count}"
        )
        if subdir:
            base = base / subdir
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _write_board_logs(
        self,
        agent_a: Agent_Model,
        agent_b: Agent_Model,
        board_records: List[List[int]],
    ) -> None:
        """
        対戦ごとの盤面推移をテキストファイルとして記録する。

        Args:
            agent_a (Agent_Model): 対戦者A。
            agent_b (Agent_Model): 対戦者B。
            board_records (List[List[int]]): 各対戦終了時の盤面履歴。
        """
        dir_path = self._resolve_result_dir("match")
        filename = (
            f"{agent_a.player_name}({agent_a.get_player_icon()})_vs_"
            f"{agent_b.player_name}({agent_b.get_player_icon()}).txt"
        )
        file_path = dir_path / filename
        with file_path.open("w", encoding="utf-8") as f:
            f.write(
                f"{agent_a.get_player_icon()} = {agent_a.get_my_team_value()}, "
                f"{agent_b.get_player_icon()} = {agent_b.get_my_team_value()}, "
                f"空白 = {self.env.EMPTY}\n\n"
            )
            for idx, board in enumerate(board_records):
                f.write(f"Game {idx + 1}\n")
                for y in range(self.env.BOARD_SIDE):
                    row = board[y * self.env.BOARD_SIDE : (y + 1) * self.env.BOARD_SIDE]
                    row_str = " ".join(
                        (
                            agent_a.get_player_icon()
                            if v == agent_a.get_my_team_value()
                            else (
                                agent_b.get_player_icon()
                                if v == agent_b.get_my_team_value()
                                else " "
                                if v == self.env.EMPTY
                                else "?"
                            )
                        )
                        for v in row
                    )
                    f.write(row_str + "\n")
                f.write("\n")

    def _write_summary_csv(self, summary: List[Dict[str, Any]]) -> None:
        """
        評価サマリを CSV 形式で保存する。

        Args:
            summary (List[Dict[str, Any]]): 各エージェントの統計情報リスト。
        """
        dir_path = self._resolve_result_dir()
        csv_path = dir_path / "evaluation_history.csv"
        with csv_path.open(mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["agent_name", "episode", "win_rate", "lose_rate", "draw_rate"]
            )
            for entry in summary:
                writer.writerow(
                    [
                        entry["agent_name"],
                        entry["episode"],
                        f"{entry['win_rate']:.1f}",
                        f"{entry['lose_rate']:.1f}",
                        f"{entry['draw_rate']:.1f}",
                    ]
                )

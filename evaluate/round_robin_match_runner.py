from typing import List, Dict, Any
from agent.model.N_Mark_Alignment_agent_model import Agent_Model
from N_Mark_Alignment_env import N_Mark_Alignment_Env
import os
import csv


class RoundRobinMatchRunner:
    def __init__(self, env: N_Mark_Alignment_Env, eval_episodes: int = 50):
        """
        総当たり戦形式でエージェント同士を評価するためのランナー。
        環境は1つを使いまわし、各試合でset_player()によりプレイヤーを差し替える。

        Args:
            env (N_Mark_Alignment_Env): 対戦に使用する環境（1インスタンスを使い回す）
            eval_episodes (int): 各対戦カードごとの試合回数
        """
        self.env = env
        self.eval_episodes = eval_episodes
        self.match_results: List[Dict[str, Any]] = []

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

        board_records = []

        for _ in range(self.eval_episodes):
            self.env.auto_play()
            board = self.env.get_board()
            board_records.append(board.copy())

        # 保存処理（アイコン表示対応）
        dir_path = f"evaluate/result/{self.env.BOARD_SIDE}_{self.env.REWARD_LINE}_{len(self.env.get_player_list())}/match"
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{dir_path}/{agent_a.player_name}({agent_a.get_player_icon()})_vs_{agent_b.player_name}({agent_b.get_player_icon()}).txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(
                f"{agent_a.get_player_icon()} = {agent_a.get_my_team_value()}, {agent_b.get_player_icon()} = {agent_b.get_my_team_value()}, 空白 = {self.env.EMPTY}\n\n"
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
                print(
                    f"\n== {agent_a.get_player_icon()} vs {agent_b.get_player_icon()} =="
                )
                result = self.run_match(agent_a, agent_b)
                print(
                    f"試合結果: {agent_a.get_player_icon()}:win={result['win']:.1f}%, {agent_b.get_player_icon()}:win={result['lose']:.1f}%, draw={result['draw']:.1f}%"
                )

        # 結果出力
        print("\n=== 最終結果 ===")
        for r in self.match_results:
            print(
                f"{r['agent_a_name']}({r['agent_a_icon']}) vs {r['agent_b_name']}({r['agent_b_icon']}) → {r['agent_a_name']}:win={r['win']:.1f}%, "
                f"{r['agent_b_name']}:win={r['lose']:.1f}%, draw={r['draw']:.1f}%"
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

        # ─── 全モデルの学習回数と勝率等をまとめてCSV保存 ───
        dir_path = f"evaluate/result/{self.env.BOARD_SIDE}_{self.env.REWARD_LINE}_{len(self.env.get_player_list())}"
        os.makedirs(dir_path, exist_ok=True)
        csv_path = os.path.join(dir_path, "evaluation_history.csv")

        # CSV モジュールを使用して保存
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            # ヘッダー
            writer.writerow(["episode", "win_rate", "lose_rate", "draw_rate"])
            for entry in summary:
                writer.writerow(
                    [
                        entry["episode"],
                        f"{entry['win_rate']:.1f}",
                        f"{entry['lose_rate']:.1f}",
                        f"{entry['draw_rate']:.1f}",
                    ]
                )

        return summary

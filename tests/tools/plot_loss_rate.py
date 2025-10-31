"""
評価履歴 CSV から負け率の推移を描画するスクリプト。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    board_side: int = 3
    reward_line: int = 3
    player_count: int = 2

    filename_prefix = f"DQN-Agent_{board_side}_{reward_line}_{player_count}"
    csv_path = Path("agent_model") / filename_prefix / "evaluation_history.csv"

    df = pd.read_csv(csv_path)

    plt.rcParams["font.family"] = "MS Gothic"

    title_str = (
        f"{board_side}×{board_side}盤 {reward_line}列揃え ({player_count}人対戦)"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(df["episode"], df["lose_rate"], marker="o", label="Lose Rate")

    plt.xlabel("Episode（学習数）")
    plt.ylabel("Lose Rate (%)")
    plt.title(f"学習回数と敗北率の推移\n{title_str}")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

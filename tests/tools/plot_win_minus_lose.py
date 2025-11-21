"""
評価結果から勝率と敗北率の差を描画するスクリプト。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    board_side: int = 9
    reward_line: int = 5
    player_count: int = 2

    filename_prefix = f"{board_side}_{reward_line}_{player_count}"
    csv_path = Path("evaluate") / "result" / filename_prefix / "evaluation_history.csv"

    df = pd.read_csv(csv_path)
    df["win_minus_lose"] = df["win_rate"] - df["lose_rate"]

    plt.rcParams["font.family"] = "MS Gothic"

    title_str = (
        f"{board_side}×{board_side}盤 {reward_line}列揃え ({player_count}人対戦)"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        df["episode"],
        df["win_minus_lose"],
        marker="o",
        label="勝率 - 敗北率",
    )

    plt.xlabel("Episode（学習数）")
    plt.ylabel("勝率 - 敗北率 (%)")
    plt.title(f"学習回数と（勝率 - 敗北率）の推移\n{title_str}")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

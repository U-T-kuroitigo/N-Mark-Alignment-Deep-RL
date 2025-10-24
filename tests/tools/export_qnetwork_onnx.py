"""
QNetwork を ONNX 形式でエクスポートするユーティリティ。
学習済みモデルを読み込んでいない初期状態のネットワークを出力する。
"""

from pathlib import Path

import torch

from agent.network.q_network import set_network


def main() -> None:
    board_side = 9
    num_team_values = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net, _ = set_network(
        board_side=board_side,
        num_team_values=num_team_values,
        device=device,
    )
    policy_net.eval()

    dummy_board = torch.randn(
        1, 1, board_side, board_side, device=device
    )  # (B, C=1, S, S)
    dummy_team = torch.zeros(1, dtype=torch.long, device=device)  # (B,)

    onnx_output_path = Path(f"qnetwork_{board_side}x{board_side}.onnx")

    torch.onnx.export(
        model=policy_net,
        args=(dummy_board, dummy_team),
        f=onnx_output_path.as_posix(),
        input_names=["board", "team"],
        output_names=["q_values"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "team": {0: "batch_size"},
            "q_values": {0: "batch_size"},
        },
        opset_version=11,
        do_constant_folding=True,
    )

    print(f"ONNX モデルを書き出しました: {onnx_output_path}")


if __name__ == "__main__":
    main()

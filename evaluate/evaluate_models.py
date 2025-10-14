"""
evaluate_models.py

保存済みモデル同士を総当たりで対戦させ、評価結果を表示・保存するスクリプト。
"""

# ========= 定数定義 ========= #
BOARD_SIDE: int = 3  # 盤面サイズ
REWARD_LINE: int = 3  # N目並べのN
EVAL_EPISODES: int = 200  # 各対戦の試行回数

# ========= モジュールインポート ========= #
import torch
from agent.dqn.dqn_agent import DQN_Agent
from saver.dqn_agent_saver.model_saver import ModelSaver
from agent.model.N_Mark_Alignment_agent_model import Agent_Model
from N_Mark_Alignment_env import N_Mark_Alignment_Env
from evaluate.round_robin_match_runner import RoundRobinMatchRunner
from agent.network.q_network import QNetwork
from agent.network.q_network import set_network  # ネットワーク初期化関数
from agent.playable_agent import PlayableAgent


# ========= 評価対象モデルの定義 ========= #
def get_model_list() -> list:
    """
    評価対象となるモデルファイルとプレイヤー記号を指定。
    """

    # DQN-Agent_3_3_2
    list_3_3_2 = [
        {
            "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_5000_202507191601\DQN-Agent_3_3_2_5000_202507191601.pt",
            "icon": "A",
            "player_name": "DQN3x3x2_5k",
        },
        {
            "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_10000_202507191606\DQN-Agent_3_3_2_10000_202507191606.pt",
            "icon": "B",
            "player_name": "DQN3x3x2_10k",
        },
        {
            "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_15000_202507191613\DQN-Agent_3_3_2_15000_202507191613.pt",
            "icon": "C",
            "player_name": "DQN3x3x2_15k",
        },
        {
            "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_20000_202507191620\DQN-Agent_3_3_2_20000_202507191620.pt",
            "icon": "D",
            "player_name": "DQN3x3x2_20k",
        },
        {
            "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_25000_202507191628\DQN-Agent_3_3_2_25000_202507191628.pt",
            "icon": "E",
            "player_name": "DQN3x3x2_25k",
        },
        {
            "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_30000_202507191636\DQN-Agent_3_3_2_30000_202507191636.pt",
            "icon": "F",
            "player_name": "DQN3x3x2_30k",
        },
        {
            "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_35000_202507191644\DQN-Agent_3_3_2_35000_202507191644.pt",
            "icon": "G",
            "player_name": "DQN3x3x2_35k",
        },
        {
            "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_40000_202507191652\DQN-Agent_3_3_2_40000_202507191652.pt",
            "icon": "H",
            "player_name": "DQN3x3x2_40k",
        },
        {
            "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_45000_202507191701\DQN-Agent_3_3_2_45000_202507191701.pt",
            "icon": "I",
            "player_name": "DQN3x3x2_45k",
        },
        {
            "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_50000_202507191710\DQN-Agent_3_3_2_50000_202507191710.pt",
            "icon": "J",
            "player_name": "DQN3x3x2_50k",
        },
    ]

    # DQN-Agent_5_3_2
    list_5_3_2 = [
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_5000_202507191857\DQN-Agent_5_3_2_5000_202507191857.pt",
            "icon": "A",
            "player_name": "DQN5x3x2_5k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_10000_202507191905\DQN-Agent_5_3_2_10000_202507191905.pt",
            "icon": "B",
            "player_name": "DQN5x3x2_10k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_15000_202507191913\DQN-Agent_5_3_2_15000_202507191913.pt",
            "icon": "C",
            "player_name": "DQN5x3x2_15k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_20000_202507191923\DQN-Agent_5_3_2_20000_202507191923.pt",
            "icon": "D",
            "player_name": "DQN5x3x2_20k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_25000_202507191931\DQN-Agent_5_3_2_25000_202507191931.pt",
            "icon": "E",
            "player_name": "DQN5x3x2_25k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_30000_202507191940\DQN-Agent_5_3_2_30000_202507191940.pt",
            "icon": "F",
            "player_name": "DQN5x3x2_30k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_35000_202507191950\DQN-Agent_5_3_2_35000_202507191950.pt",
            "icon": "G",
            "player_name": "DQN5x3x2_35k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_40000_202507191959\DQN-Agent_5_3_2_40000_202507191959.pt",
            "icon": "H",
            "player_name": "DQN5x3x2_40k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_45000_202507192008\DQN-Agent_5_3_2_45000_202507192008.pt",
            "icon": "I",
            "player_name": "DQN5x3x2_45k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_50000_202507192018\DQN-Agent_5_3_2_50000_202507192018.pt",
            "icon": "J",
            "player_name": "DQN5x3x2_50k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_55000_202507192028\DQN-Agent_5_3_2_55000_202507192028.pt",
            "icon": "K",
            "player_name": "DQN5x3x2_55k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_60000_202507192038\DQN-Agent_5_3_2_60000_202507192038.pt",
            "icon": "L",
            "player_name": "DQN5x3x2_60k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_65000_202507192048\DQN-Agent_5_3_2_65000_202507192048.pt",
            "icon": "M",
            "player_name": "DQN5x3x2_65k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_70000_202507192058\DQN-Agent_5_3_2_70000_202507192058.pt",
            "icon": "N",
            "player_name": "DQN5x3x2_70k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_75000_202507192109\DQN-Agent_5_3_2_75000_202507192109.pt",
            "icon": "O",
            "player_name": "DQN5x3x2_75k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_80000_202507192119\DQN-Agent_5_3_2_80000_202507192119.pt",
            "icon": "P",
            "player_name": "DQN5x3x2_80k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_85000_202507192130\DQN-Agent_5_3_2_85000_202507192130.pt",
            "icon": "Q",
            "player_name": "DQN5x3x2_85k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_90000_202507192140\DQN-Agent_5_3_2_90000_202507192140.pt",
            "icon": "R",
            "player_name": "DQN5x3x2_90k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_95000_202507192152\DQN-Agent_5_3_2_95000_202507192152.pt",
            "icon": "S",
            "player_name": "DQN5x3x2_95k",
        },
        {
            "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_100000_202507192203\DQN-Agent_5_3_2_100000_202507192203.pt",
            "icon": "T",
            "player_name": "DQN5x3x2_100k",
        },
    ]

    # DQN-Agent_9_5_2
    list_9_5_2 = [
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_5000_202507200416\DQN-Agent_9_5_2_5000_202507200416.pt",
            "icon": "A",
            "player_name": "DQN9x5x2_5k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_10000_202507200435\DQN-Agent_9_5_2_10000_202507200435.pt",
            "icon": "B",
            "player_name": "DQN9x5x2_10k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_15000_202507200456\DQN-Agent_9_5_2_15000_202507200456.pt",
            "icon": "C",
            "player_name": "DQN9x5x2_15k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_20000_202507200518\DQN-Agent_9_5_2_20000_202507200518.pt",
            "icon": "D",
            "player_name": "DQN9x5x2_20k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_25000_202507200539\DQN-Agent_9_5_2_25000_202507200539.pt",
            "icon": "E",
            "player_name": "DQN9x5x2_25k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_30000_202507200602\DQN-Agent_9_5_2_30000_202507200602.pt",
            "icon": "F",
            "player_name": "DQN9x5x2_30k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_35000_202507200623\DQN-Agent_9_5_2_35000_202507200623.pt",
            "icon": "G",
            "player_name": "DQN9x5x2_35k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_40000_202507200646\DQN-Agent_9_5_2_40000_202507200646.pt",
            "icon": "H",
            "player_name": "DQN9x5x2_40k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_45000_202507200710\DQN-Agent_9_5_2_45000_202507200710.pt",
            "icon": "I",
            "player_name": "DQN9x5x2_45k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_50000_202507200733\DQN-Agent_9_5_2_50000_202507200733.pt",
            "icon": "J",
            "player_name": "DQN9x5x2_50k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_55000_202507200757\DQN-Agent_9_5_2_55000_202507200757.pt",
            "icon": "K",
            "player_name": "DQN9x5x2_55k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_60000_202507200821\DQN-Agent_9_5_2_60000_202507200821.pt",
            "icon": "L",
            "player_name": "DQN9x5x2_60k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_65000_202507200845\DQN-Agent_9_5_2_65000_202507200845.pt",
            "icon": "M",
            "player_name": "DQN9x5x2_65k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_70000_202507200910\DQN-Agent_9_5_2_70000_202507200910.pt",
            "icon": "N",
            "player_name": "DQN9x5x2_70k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_75000_202507200935\DQN-Agent_9_5_2_75000_202507200935.pt",
            "icon": "O",
            "player_name": "DQN9x5x2_75k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_80000_202507201000\DQN-Agent_9_5_2_80000_202507201000.pt",
            "icon": "P",
            "player_name": "DQN9x5x2_80k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_85000_202507201026\DQN-Agent_9_5_2_85000_202507201026.pt",
            "icon": "Q",
            "player_name": "DQN9x5x2_85k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_90000_202507201051\DQN-Agent_9_5_2_90000_202507201051.pt",
            "icon": "R",
            "player_name": "DQN9x5x2_90k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_95000_202507201116\DQN-Agent_9_5_2_95000_202507201116.pt",
            "icon": "S",
            "player_name": "DQN9x5x2_95k",
        },
        {
            "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_100000_202507201141\DQN-Agent_9_5_2_100000_202507201141.pt",
            "icon": "T",
            "player_name": "DQN9x5x2_100k",
        },
    ]

    # 各一番性能が高かったモデル
    most_performing_model = [
        # DQN-Agent_3_3_2
        # {
        #     "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_45000_202507191701\DQN-Agent_3_3_2_45000_202507191701.pt",
        #     "icon": "I",
        #     "player_name": "DQN3x3x2_45k",
        # },
        {
            "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_45000_202507191701\DQN-Agent_3_3_2_45000_202507191701.pt",
            "icon": "IA",
            "player_name": "DQN3x3x2_45k_A",
        },
        {
            "path": "agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_45000_202507191701\DQN-Agent_3_3_2_45000_202507191701.pt",
            "icon": "IB",
            "player_name": "DQN3x3x2_45k_B",
        },
        #
        #
        #
        # DQN-Agent_5_3_2
        # {
        #     "path": "agent_model\DQN-Agent_5_3_2\DQN-Agent_5_3_2_60000_202507192038\DQN-Agent_5_3_2_60000_202507192038.pt",
        #     "icon": "L",
        #     "player_name": "DQN5x3x2_60k",
        # },
        #
        #
        # DQN-Agent_9_5_2
        # {
        #     "path": "agent_model\DQN-Agent_9_5_2\DQN-Agent_9_5_2_100000_202507201141\DQN-Agent_9_5_2_100000_202507201141.pt",
        #     "icon": "T",
        #     "player_name": "DQN9x5x2_100k",
        # },
    ]

    # return list_3_3_2
    # return list_5_3_2
    # return list_9_5_2
    return most_performing_model


# ========= モデル読込関数 ========= #
def load_agent_model(filepath: str) -> DQN_Agent:
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ネットワーク定義と初期化
    policy_net, target_net = set_network(BOARD_SIDE, device)

    dummy_agent = DQN_Agent(
        player_icon="?",
        player_value=0,
        learning=False,
        policy_net=policy_net,
        target_net=target_net,
        device=device,
    )

    # MetaSaver.load() に備えて最低限の属性を持たせる（仮値でOK）
    dummy_agent.reward_line = REWARD_LINE
    dummy_agent.board_side = BOARD_SIDE
    dummy_agent.team_value_list = [1, 2]  # 仮チームリスト

    agent = ModelSaver().load(filepath, dummy_agent, load_replay=False)
    agent.set_learning(False)
    return agent


# ========= メイン評価実行部 ========= #
def main() -> None:
    # モデルロード
    model_list = []
    for idx, info in enumerate(get_model_list()):
        agent = load_agent_model(info["path"])
        agent.set_player_icon(info["icon"])
        agent.set_player_value(idx)
        agent.player_name = info["player_name"]
        model_list.append(agent)

    # temp = model_list[0]
    # model_list.insert(0, model_list.pop(1))
    # model_list.insert(1, temp)

    # プレイアブルエージェントを追加
    # playable_agent = PlayableAgent("PL", 99)
    # playable_agent.player_name = "Player"
    # model_list.append(playable_agent)

    # 環境作成（使い回し）
    env = N_Mark_Alignment_Env(
        board_side=BOARD_SIDE,
        reward_line=REWARD_LINE,
        player_list=model_list[:2],  # 仮で2人渡す（run_match時に毎回上書きされる）
    )

    # ランナー生成
    runner = RoundRobinMatchRunner(env=env, eval_episodes=EVAL_EPISODES)

    # 総当たり評価開始
    summary = runner.evaluate(model_list)

    # ===== 正しいランキング集計（両者に反映） =====

    # ─── ランキング表示 ───
    print("\n=== ランキング ===")
    # 「勝率 - 敗率」が大きい順にソート
    ranking = sorted(
        summary, key=lambda x: x["win_rate"] - x["lose_rate"], reverse=True
    )
    for i, entry in enumerate(ranking, start=1):
        print(
            f"{i}. {entry['agent_name']}({entry['agent_icon']}): "
            f"win={entry['win_rate']:.1f}%, "
            f"lose={entry['lose_rate']:.1f}%, "
            f"draw={entry['draw_rate']:.1f}%"
        )


if __name__ == "__main__":
    main()

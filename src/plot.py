import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot(kwargs):
    filename = os.path.join(kwargs["dir_name"], kwargs["alg_name"], kwargs["map_name"], kwargs["t"], kwargs["filename"])
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(kwargs["map_name"])
    if kwargs["map_name"] in ["lbf", "simple_spread", "PP"]:
        pass
    else:
        ax[0, 0].plot(data["test_battle_won_mean"])
        ax[0, 0].set_title("test_battle_won_mean")
    ax[0, 1].plot([item["value"] for item in data["test_return_mean"]])
    ax[0, 1].set_title("test_return_mean")
    ax[0, 2].plot(data["loss"])
    ax[0, 2].set_title("loss")
    ax[1, 0].plot(data["td_loss"])
    ax[1, 0].set_title("td_loss")
    ax[1, 1].plot(data["consensus_loss"])
    ax[1, 1].set_title("consensus_loss")
    ax[1, 2].plot(data["hidden_state_loss"])
    ax[1, 2].set_title("hidden_state_loss")

    # ax[1, 0].plot(data["td_error_abs"])
    # ax[1, 0].plot("td_error_abs")
    plt.show()


def main(map_name="lbf"):
    kwargs = {}
    kwargs["dir_name"] = os.path.join(os.getcwd(), "results/sacred")
    kwargs["alg_name"] = "macl"
    kwargs["map_name"] = map_name
    kwargs["t"] = "1"
    kwargs["filename"] = "info.json"
    plot(kwargs)


if __name__ == "__main__":
    map_names = ["lbf", "PP", "3s5z", "1c3s5z", "2s_vs_1sc", "10m_vs_11m", "2s3z", "2c_vs_64zg", "MMM2", "5m_vs_6m", "3s_vs_5z", "corridor", "3s5z_vs_3s6z"]
    for map_name in map_names:
        main(map_name)
    # main()
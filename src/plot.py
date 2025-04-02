import os
import json
import numpy as np
import matplotlib.pyplot as plt


def resolve(data):
    confidence = 0.95
    assert type(data) == np.ndarray
    _mean = np.mean(data, axis=0)
    _std = np.std(data, axis=0)
    _max = _mean + _std * confidence
    _min = _mean - _std * confidence
    return _mean, _max, _min, _std


def get_test_result(kwargs):
    test_result = {}
    test_battle_won_means = []
    test_return_means = []
    # for time in kwargs["t"]:
    for time in os.listdir(os.path.join(kwargs["dir_name"], kwargs["alg_name"], kwargs["map_name"])):
        filename = os.path.join(kwargs["dir_name"], kwargs["alg_name"], kwargs["map_name"], time, kwargs["filename"])
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if kwargs["map_name"] in ["lbf", "simple_spread", "PP"]:
                pass
            else:
                test_battle_won_mean = data["test_battle_won_mean"][:kwargs["max_len"]]
                test_battle_won_means.append(test_battle_won_mean)
            test_return_mean = [item["value"] for item in data["test_return_mean"]][:kwargs["max_len"]]
            test_return_means.append(test_return_mean)
    test_result["test_battle_won_mean"] = np.array(test_battle_won_means)
    test_result["test_return_mean"] = np.array(test_return_means)
    return test_result


def plot(kwargs):
    result = get_test_result(kwargs)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(kwargs["map_name"])
    if kwargs["map_name"] in ["lbf", "simple_spread", "PP"]:
        pass
    else:
        _mean, _max, _min, _std = resolve(result["test_battle_won_mean"])
        ax[0].plot(_mean, color="#d62728")
        ax[0].fill_between(_max, _min, facecolor="#d62728", alpha=0.1)
        ax[0].set_title("test_battle_won_mean")
    _mean, _max, _min, _std = resolve(result["test_return_mean"])
    x = np.linspace(0, kwargs["max_len"]//100, kwargs["max_len"])
    ax[1].plot(x, _mean, color="#d62728")
    ax[1].fill_between(x, _max, _min, facecolor="#d62728", alpha=0.1)
    ax[1].set_title("test_return_mean")
    plt.show()


def main():
    kwargs = {}
    kwargs["dir_name"] = os.path.join(os.getcwd(), "results/sacred")
    # kwargs["dir_name"] = os.path.join("/home/oseasy/桌面", "results/sacred")
    kwargs["alg_name"] = "full"
    env_info = {
        "lbf": 400,
        "simple_spread": 200,
        "3s5z": 200,
        "1c3s5z": 200,
        "2s_vs_1sc": 200,
        "10m_vs_11m": 200,
        "2s3z": 200,
        "2c_vs_64zg": 200,
        "MMM2": 200,
        "5m_vs_6m": 200,
        "3s_vs_5z": 200,
        "corridor": 500,
        "3s5z_vs_3s6z": 500
    }
    for map_name, max_len in env_info.items():
        kwargs["map_name"] = map_name
        kwargs["max_len"] = max_len
        kwargs["filename"] = "info.json"
        plot(kwargs)


if __name__ == "__main__":
    main()
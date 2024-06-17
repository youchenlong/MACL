import os
import json
import numpy as np
import matplotlib.pyplot as plt



def plot(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data =json.load(f)

    if "simple_spread" in filename or "lbf" in filename:
        metric = [item["value"] for item in data["test_return_mean"]]
    else:
        metric = data["test_battle_won_mean"]

    plt.plot(metric)
    plt.show()

def main():
    dir_name = os.path.join(os.getcwd(), "results/sacred")
    alg_name = "macl"
    map_name = "lbf"
    t = "1"
    filename = os.path.join(dir_name, alg_name, map_name, t, "info.json")

    plot(filename)

if __name__ == "__main__":
    main()
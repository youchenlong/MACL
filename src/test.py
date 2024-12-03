from envs import REGISTRY
import time
import numpy as np

seed = 3
env = REGISTRY["foraging"](field_size=10, players=4, max_food=4, force_coop=True, partially_observe=True, sight=2, is_print=False, seed=seed, need_render=False)
env.reset()
env.render()
for player in env.env.players:
    print(player.position)
print(env.get_visibility_matrix())

time.sleep(100)
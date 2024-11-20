from functools import partial
from .multiagentenv import MultiAgentEnv
from .stag_hunt import StagHunt
# from .matrix_game.nstep_matrix_game import NStepMatrixGame
# from .rware import RWAREEnv
from .lbforaging import ForagingEnv
# from .sisl import SislEnv
from .mpe import MPEEnv
import sys
import os

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
# REGISTRY["nstep_matrix"] = partial(env_fn, env=NStepMatrixGame)
# REGISTRY["rware"] = partial(env_fn, env=RWAREEnv)
REGISTRY["foraging"] = partial(env_fn, env=ForagingEnv)
# REGISTRY["sisl"] = partial(env_fn, env=SislEnv)
REGISTRY["mpe"] = partial(env_fn, env=MPEEnv)

def register_smac():
    from .smac_wrapper import SMACWrapper
    REGISTRY["sc2"] = partial(env_fn, env=SMACWrapper)

def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper
    REGISTRY["sc2v2"] = partial(env_fn, env=SMACv2Wrapper)
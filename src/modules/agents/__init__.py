REGISTRY = {}

from .rnn_agent import RNNAgent
from .dsr_agent import DSRAgent
from .macl_agent import MACLAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["dsr"] = DSRAgent
REGISTRY["macl"] = MACLAgent
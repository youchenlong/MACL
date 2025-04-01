REGISTRY = {}

from .rnn_agent import RNNAgent
from .dsr_agent import DSRAgent
from .macl_agent import MACLAgent
from .full_agent import FullAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["dsr"] = DSRAgent
REGISTRY["macl"] = MACLAgent
REGISTRY["full"] = FullAgent
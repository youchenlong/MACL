REGISTRY = {}

from .rnn_agent import RNNAgent
from .dsr_agent import DSRAgent
from .tom_agent import ToMAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["dsr"] = DSRAgent
REGISTRY["tom"] = ToMAgent
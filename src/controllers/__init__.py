REGISTRY = {}

from .basic_controller import BasicMAC
from .dsr_controller import DSRMAC
from .macl_controller import MACLMAC
from .maddpg_controller import MADDPGMAC
from .full_controller import FullMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["dsr_mac"] = DSRMAC
REGISTRY["macl_mac"] = MACLMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["full_mac"] = FullMAC
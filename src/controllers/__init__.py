REGISTRY = {}

from .basic_controller import BasicMAC
from .dsr_controller import DSRMAC
from .macl_controller import MACLMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["dsr_mac"] = DSRMAC
REGISTRY["macl_mac"] = MACLMAC
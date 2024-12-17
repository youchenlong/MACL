REGISTRY = {}

from .basic_controller import BasicMAC
from .dsr_controller import DSRMAC
from .tom_controller import ToMMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["dsr_mac"] = DSRMAC
REGISTRY["tom_mac"] = ToMMAC
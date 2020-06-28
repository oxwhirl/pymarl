REGISTRY = {}

from .basic_controller import BasicMAC
from .simple_controller import SimPLeMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["simple_mac"] = SimPLeMAC
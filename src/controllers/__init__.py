REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_central_controller import CentralBasicMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
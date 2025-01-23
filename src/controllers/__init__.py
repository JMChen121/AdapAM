REGISTRY = {}
MASKER_REGISTRY = {}
AIA_REGISTRY = {}
SAIA_REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .n_controller_aia import NMACAIA
from .n_controller_saia import GaussianPolicy
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .basic_central_controller import CentralBasicMAC
from .lica_controller import LICAMAC
from .dop_controller import DOPMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["dop_mac"] = DOPMAC

MASKER_REGISTRY["n_mac"] = NMAC

AIA_REGISTRY["n_mac"] = NMACAIA

SAIA_REGISTRY["n_mac"] = GaussianPolicy

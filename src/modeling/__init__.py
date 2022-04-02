from .signatures_generator import SignaturesGenerator
from .model_utils import LoggingCallback
from .hyper_drf import HyperDRF
from .hyper_pada import HyperPada
from .hyper_dn import HyperDomainName

MODEL_CLASSES = {'generator': SignaturesGenerator, 'hyper-drf': HyperDRF, 'hyper-pada': HyperPada,
                 'hyper-dn': HyperDomainName}

MODEL_PREFIXES = {'generator': 'Generator ',
                  'hyper-drf': 'Hyper DRF ', 'hyper-pada': 'Hyper PADA ',
                  'hyper-dn': 'Hyper DN '}

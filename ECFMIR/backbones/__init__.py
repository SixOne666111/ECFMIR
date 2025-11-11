from .FusionNets.TCL_MAP import TCL_MAP
from .FusionNets.ECFMIR import TRUST
from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.MAG_BERT import MAG_BERT
from .FusionNets.MISA import MISA
from .FusionNets.MULT import MULT
from .FusionNets.SDIF import SDIF

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder
                }

methods_map = {
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'mult': MULT,
    'sdif': SDIF,
    'tcl_map': TCL_MAP,
    'ecfmir': ECFMIR,
}
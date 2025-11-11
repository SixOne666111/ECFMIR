from .TCL_MAP.manager import TCL_MAP
from .MISA.manager import MISA
from .MULT.manager import MULT
from .MAG_BERT.manager import MAG_BERT
from .SDIF.manager import SDIF
from .ECFMIR.manager import ECFMIR

method_map = {
    'text': TEXT,
    'misa': MISA,
    'mult': MULT,
    'sdif': SDIF,
    'mag_bert': MAG_BERT,
    'tcl_map': TCL_MAP,
    'ecfmir': ECFMIR,

}

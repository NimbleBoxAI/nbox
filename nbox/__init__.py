import nbox.utils
import nbox.model
import nbox.load
import nbox.parsers
import nbox.framework

from nbox.model import Model
from nbox.load import load, plug, PRETRAINED_MODELS
from nbox.parsers import ImageParser, TextParser

__version__ = "0.2.0"

if not (nbox.utils.is_there_pt or nbox.utils.is_there_skl):
    import warnings

    warnings.warn("Neither PyTorch nor Scikit-Learn are installed, cannot run local inference")

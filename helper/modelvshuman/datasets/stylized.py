from os.path import join as pjoin

from .base import Dataset
from .dataloaders import PytorchLoader
from .imagenet import ImageNetParams
from .registry import register_dataset
import quba_constants as c
from . import info_mappings, decision_mappings

__all__ = ["stylized", "corr_stylized"]

@register_dataset(name='stylized')
def stylized(model, args):
    params = ImageNetParams(path=pjoin(c._DATASET_DIR, "stylized"),
                            decision_mapping=decision_mappings.ImageNetProbabilitiesTo16ClassesMapping(),
                            info_mapping=info_mappings.InfoMappingWithSessions(),
                            contains_sessions=True)
    return Dataset(name="stylized",
                   model=model,
                   params=params,
                   loader=PytorchLoader,
                   args=args)

@register_dataset(name='corr_stylized')
def corr_stylized(model, args):
    params = ImageNetParams(path=pjoin(c._DATASET_DIR, "corr_stylized"),
                            decision_mapping=decision_mappings.ImageNetProbabilitiesTo16ClassesMapping(),
                            info_mapping=info_mappings.InfoMappingWithSessions(),
                            contains_sessions=True)
    return Dataset(name="corr_stylized",
                   model=model,
                   params=params,
                   loader=PytorchLoader,
                   args=args)
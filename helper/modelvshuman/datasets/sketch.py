from os.path import join as pjoin

from .base import Dataset
from .dataloaders import PytorchLoader
from .imagenet import ImageNetParams
from .registry import register_dataset
import quba_constants as c
from . import info_mappings, decision_mappings

__all__ = ["sketch", "corr_sketch"]

@register_dataset(name='sketch')
def sketch(model, args):
    params = ImageNetParams(path=pjoin(c._DATASET_DIR, "sketch"),
                            decision_mapping=decision_mappings.ImageNetProbabilitiesTo16ClassesMapping(),
                            info_mapping=info_mappings.InfoMappingWithSessions(),
                            contains_sessions=True)
    return Dataset(name="sketch",
                   params=params,
                   model=model,
                   loader=PytorchLoader,
                   args=args)

@register_dataset(name='corr_sketch')
def corr_sketch(model, args):
    params = ImageNetParams(path=pjoin(c._DATASET_DIR, "corr_sketch"),
                            decision_mapping=decision_mappings.ImageNetProbabilitiesTo16ClassesMapping(),
                            info_mapping=info_mappings.InfoMappingWithSessions(),
                            contains_sessions=True)
    return Dataset(name="corr_sketch",
                   params=params,
                   model=model,
                   loader=PytorchLoader,
                   args=args)
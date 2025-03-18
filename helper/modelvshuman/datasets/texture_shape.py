from dataclasses import dataclass, field
from os.path import join as pjoin
from typing import List

from . import decision_mappings, info_mappings
from .base import Dataset
from .dataloaders import PytorchLoader
from .registry import register_dataset
import quba_constants as c
from ..evaluation import metrics as m

__all__ = ["original", "greyscale", "texture", "edge", "silhouette", 
           "corr_edge", "corr_silhouette",
           "cue_conflict"]


@dataclass
class TextureShapeParams:
    path: str
    image_size: int = 224
    metrics: list = field(default_factory=lambda: [m.Accuracy(topk=1)])
    decision_mapping: object = decision_mappings.ImageNetProbabilitiesTo16ClassesMapping()
    info_mapping: object = info_mappings.ImageNetInfoMapping()
    experiments: List = field(default_factory=list)
    contains_sessions: bool = False


def _get_dataset(name, model, args):
    params = TextureShapeParams(path=pjoin(c._DATASET_DIR, name))
    return Dataset(name=name,
                   params=params,
                   model=model,
                   loader=PytorchLoader,
                   args=args)


@register_dataset(name="original")
def original(model, args):
    return _get_dataset(name="original", model=model, args=args)


@register_dataset(name="greyscale")
def greyscale(model, args):
    return _get_dataset(name="greyscale", model=model, args=args)


@register_dataset(name="texture")
def texture(model, args):
    return _get_dataset(name="texture", model=model, args=args)


@register_dataset(name="edge")
def edge(model, args):
    return _get_dataset(name="edge", model=model, args=args)


@register_dataset(name="silhouette")
def silhouette(model, args):
    return _get_dataset("silhouette", model=model, args=args)


@register_dataset(name="cue-conflict")
def cue_conflict(model, args):
    return _get_dataset("cue-conflict", model=model, args=args)

@register_dataset(name="corr_edge")
def corr_edge(model, args):
    return _get_dataset(name="corr_edge", model=model, args=args)


@register_dataset(name="corr_silhouette")
def corr_silhouette(model, args):
    return _get_dataset("corr_silhouette", model=model, args=args)

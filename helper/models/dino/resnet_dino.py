import torch
from torchvision import models as torchvision_models
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

@register_model
def resnet50_dino(pretrained=False, image_res=224, **kwargs):
    if pretrained:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        model.fc = torch.nn.Linear(2048, 5, bias=True)
        torch.nn.init.xavier_normal_(model.fc.weight)
        torch.nn.init.zeros_(model.fc.bias)
    else:
        model = torchvision_models.__dict__["resnet50"]()
    
    return model


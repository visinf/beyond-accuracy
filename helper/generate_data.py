import timm.data
import torchvision.datasets as dset
import torch
import torchvision.transforms as tv_transforms
#from torchvision import transforms
import torchvision.models as models
import os
from robustbench import utils as rob_bench_utils
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import requests
from os.path import join
from tqdm import tqdm
import shutil
import bagnets.pytorchnet
import torch.nn as nn
from abc import abstractmethod
import quba_constants as con
import bcos.models.pretrained as bcosmodels
from bcos import transforms as bcos_transforms
import torch.nn.functional as F
from helper.imagenet import imagenet_templates as openai_imagenet_template
import torch
import quba_constants as wc
from helper.models.model_wrappers import StandardModel, DINOModel, ClipModel, MobileCLIPModel, SigLIPModel, SigLIP2Model, OpenCLIPModel

def load_dataset(model, path, num_workers=8, batch_size=32):
    
    """
    Returns loader for the ImageNet Dataset
    
    :model: model of class StandardModel
    :path: path to dataset
    :batch_size: Batch Size of dataloader
    :num_workers: Number of Workers
    :return: ImageNet Dataloader
    """

    transform = None
    
    dataset = dset.ImageFolder(
                root=path, 
                transform=transform if transform is not None else model.transform
                )
    
    #dataset = MyImageFolder(
    #            root=path, 
    #            transform=transform if transform is not None else model.transform
    #            )

    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)

    return dataset_loader

def download_and_create_model(timm_name, url_filename, new_name):
    # Assuming `download_file` is a function that downloads the file
    download_file(url=wc.RSB_LINK + url_filename, new_name=new_name)
    model = timm.create_model(timm_name, wc.RSB_ROOT + new_name)
    return model, timm_name

def load_model(model_arch, args):
    from quba_constants import TORCH_HUB_MODELS, TORCHVISION_MODELS, SIGLIP_MODELS, CUSTOM_MODELS, TIMM_MODELS, ROB_BENCH_MODELS, A123_MODELS
    print(f"Load model {model_arch}...")

    if model_arch in TORCHVISION_MODELS.keys():
        model_fn, weights = TORCHVISION_MODELS[model_arch]
        model = model_fn(weights=weights)
        transform = weights.transforms()
        return StandardModel(model=model, model_name=model_arch, transform=transform)

    elif model_arch in CUSTOM_MODELS.keys():
        model_fn, transform = CUSTOM_MODELS[model_arch]
        model = model_fn()
        return StandardModel(model=model, model_name=model_arch, transform=transform)  # No transform handling needed

    elif model_arch in TIMM_MODELS.keys():
        timm_name = TIMM_MODELS[model_arch]
        if timm.is_model(timm_name):
            model = timm.create_model(timm_name, pretrained=True)
            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)
        return StandardModel(model=model, model_name=model_arch, transform=transform) 

    elif model_arch in ROB_BENCH_MODELS.keys():
        model_name, transform = ROB_BENCH_MODELS[model_arch]
        model = rob_bench_utils.load_model(model_name=model_name, dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")
        return StandardModel(model=model, model_name=model_arch, transform=transform) 

    elif model_arch in TORCH_HUB_MODELS.keys():
        model_fn, transform = TORCH_HUB_MODELS[model_arch]
        model = model_fn()
        return StandardModel(model=model, model_name=model_arch, transform=transform)  # No transform handling needed

    elif model_arch in A123_MODELS.keys():
        model_fn = A123_MODELS[model_arch]
        model, timm_name = model_fn()
        if timm.is_model(timm_name):
            model = timm.create_model(timm_name, pretrained=True)
            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)
        return StandardModel(model=model, model_name=model_arch, transform=transform) 

    elif "mobileclip" in model_arch:
        return MobileCLIPModel(model_arch, args.device)
    elif model_arch in wc.OPEN_CLIP_MODELS.keys():
        return OpenCLIPModel(model_arch, args.device)
    elif model_arch in wc.OPENAI_CLIP.keys():
        return ClipModel(model_arch, args.device)
    elif model_arch in wc.SIGLIP2_MODELS.keys():
        return SigLIP2Model(model_arch, device=args.device)
    elif model_arch in wc.SIGLIP_MODELS.keys():
        from open_clip import create_model_from_pretrained, get_tokenizer
        model, preprocess = create_model_from_pretrained(wc.SIGLIP_MODELS[model_arch])
        tokenizer = get_tokenizer(wc.SIGLIP_MODELS[model_arch])
        return SigLIPModel(model_name=model_arch, model=model, preprocess=preprocess, tokenizer=tokenizer, device=args.device)

    #DINOv1
    elif "ViT-b-16-DINO-LP" == model_arch:
        import helper.models.dino.vision_transformer as dino_vit
        model = dino_vit.__dict__["vit_base"](patch_size=16, num_classes=0)
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth")
        model.load_state_dict(state_dict)
        ckpt = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth")["state_dict"]
        new_ckpt = {}
        for key, value in ckpt.items():
            new_key = key.replace('module.linear.', '')  
            new_ckpt[new_key] = value
                
        head = nn.Linear(in_features=1536, out_features=1000)
        head.load_state_dict(new_ckpt, strict=True)
        transform = wc.HIERA_LP_DINOV1_TRANSFORM
        return DINOModel(model, head, model_arch, transform)
    
    elif "ViT-s-16-DINO-LP" == model_arch:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16') #dino.vision_transformer.__dict__["vit_small"](patch_size=16, num_classes=0)
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth")
        model.load_state_dict(state_dict)
        ckpt = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth")["state_dict"]
        new_ckpt = {}
        for key, value in ckpt.items():
            new_key = key.replace('module.linear.', '')  
            new_ckpt[new_key] = value
                
        head = nn.Linear(in_features=1536, out_features=1000)
        head.load_state_dict(new_ckpt, strict=True)
        transform = wc.HIERA_LP_DINOV1_TRANSFORM
        return DINOModel(model, head, model_arch, transform)

    #Self-Trained
    elif model_arch == "vit-b-16-mae-lp":
        import helper.models.mae.models_vit as models_vit
        model = models_vit.vit_base_patch16()
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        ckpt = torch.load(wc.MAE_LP)
        model.load_state_dict(ckpt["model"])
        transform = wc.HIERA_LP_DINOV1_TRANSFORM
        return StandardModel(model=model, model_name=model_arch, transform=transform)


    elif model_arch == "ResNet50-DINO-FT":
        import helper.models.dino.resnet_dino
        model = helper.models.dino.resnet_dino.resnet50_dino()
        ckpt = torch.load(wc.DINO_RESNET_FT)
        model.load_state_dict(ckpt["state_dict"])
        transform = wc.HIERA_LP_DINOV1_TRANSFORM
        return StandardModel(model=model, model_name=model_arch, transform=transform)

    elif model_arch == "ViTB-DINO-FT":
        model = timm.create_model("timm/vit_base_patch16_224.dino")
        model.head = nn.Linear(in_features=768, out_features=1000, bias=True)
        ckpt = torch.load(wc.DINO_VIT_FT)
        model.load_state_dict(ckpt["state_dict"])
        transform = wc.HIERA_LP_DINOV1_TRANSFORM
        return StandardModel(model=model, model_name=model_arch, transform=transform)

    elif model_arch == "vit-b-16-mae-ft":
        import helper.models.mae.models_vit as models_vit
        model = models_vit.vit_base_patch16(num_classes=1000, drop_path_rate=0.1, global_pool=True)
        download_file(url="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth", new_name="mae_finetuned_vit_base.pth")
        ckpt = torch.load(wc.MAE_FT)["model"]
        model.load_state_dict(ckpt)
        transform = wc.HIERA_LP_DINOV1_TRANSFORM
        return StandardModel(model=model, model_name=model_arch, transform=transform)

    elif "Hiera-B-LP" == model_arch:
        model = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k")
        state_dict = torch.load(wc.HIERA_LP_B)["model"]
        model.load_state_dict(state_dict) 
        transform = wc.HIERA_LP_DINOV1_TRANSFORM
        return StandardModel(model=model, model_name=model_arch, transform=transform)
    elif "Hiera-S-LP" == model_arch:
        model = torch.hub.load("facebookresearch/hiera", model="hiera_small_224", pretrained=True, checkpoint="mae_in1k")
        state_dict = torch.load(wc.HIERA_LP_S)["model"]
        model.load_state_dict(state_dict) 
        transform = wc.HIERA_LP_DINOV1_TRANSFORM
        return StandardModel(model=model, model_name=model_arch, transform=transform)
    elif "Hiera-T-LP" == model_arch:
        model = torch.hub.load("facebookresearch/hiera", model="hiera_tiny_224", pretrained=True, checkpoint="mae_in1k")
        state_dict = torch.load(wc.HIERA_LP_T)["model"]
        model.load_state_dict(state_dict) 
        transform = wc.HIERA_LP_DINOV1_TRANSFORM
        return StandardModel(model=model, model_name=model_arch, transform=transform)

    elif "ViT-s-14-dinoV2-FT" == model_arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
        ckpt = torch.load(wc.DINOV2_S_FT)["model"]
        model.load_state_dict(ckpt)
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
        return StandardModel(model=model, model_name=model_arch, transform=transform)

    elif "ViT-b-14-dinoV2-FT" == model_arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
        ckpt = torch.load(wc.DINOV2_B_FT, map_location='cpu')["model"]
        model.load_state_dict(ckpt)
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
        return StandardModel(model=model, model_name=model_arch, transform=transform)

    elif "ViT-l-14-dinoV2-FT" == model_arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
        ckpt = torch.load(wc.DINOV2_L_FT, map_location='cpu')["model"]
        model.load_state_dict(ckpt)
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
        return StandardModel(model=model, model_name=model_arch, transform=transform)


    elif "ViT-l-14-dinoV2-FT-Reg" == model_arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc')
        ckpt = torch.load(wc.DINOV2_L_REG_FT, map_location='cpu')["model"]
        model.load_state_dict(ckpt)
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
        return StandardModel(model=model, model_name=model_arch, transform=transform)

    elif "ViT-b-14-dinoV2-FT-Reg" == model_arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
        ckpt = torch.load(wc.DINOV2_B_REG_FT, map_location='cpu')["model"]
        model.load_state_dict(ckpt)
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
        return StandardModel(model=model, model_name=model_arch, transform=transform)

    elif "ViT-s-14-dinoV2-FT-Reg" == model_arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
        ckpt = torch.load(wc.DINOV2_S_REG_FT, map_location='cpu')["model"]
        model.load_state_dict(ckpt)
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
        return StandardModel(model=model, model_name=model_arch, transform=transform)

    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

def download_dataset(dataset_name):
    """
    Downloads the dataset specified by dataset_name

    :dataset_name: Dataset to download. Currently supported: Background Challenge Data, ImageNet-9l, ImageNet-R, Modelvshuman
    """

    target_dir = con._DATA_DIR
    if dataset_name == "bg_challenge":
        download_url = "https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz"
    elif dataset_name == "imagenet-r":
        download_url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    elif dataset_name == "blur":
        download_url = "https://zenodo.org/records/3565846/files/blur.tar?download=1"
    elif dataset_name == "digital":
        download_url = "https://zenodo.org/records/3565846/files/digital.tar?download=1"
    elif dataset_name == "noise":
        download_url = "https://zenodo.org/records/3565846/files/noise.tar?download=1"
    elif dataset_name == "weather":
        download_url = "https://zenodo.org/records/3565846/files/weather.tar?download=1"
    else:
        target_dir = con._MODEL_VS_HUMAN_DIR
        download_url = "https://github.com/bethgelab/model-vs-human/releases/download/v0.1/{NAME}.tar.gz"
        download_url = download_url.format(NAME=dataset_name)
    
    if os.path.exists(target_dir + "/" + dataset_name):
        return
    
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        total_length = response.headers.get('content-length')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        dataset_file = join(target_dir, f'{dataset_name}.tar.gz')
        print(f"Downloading dataset {dataset_name} to {dataset_file}")
        with open(dataset_file, 'wb') as fd:
            if total_length is None:  
                fd.write(response.content)
            else:
                for chunk in tqdm(response.iter_content(chunk_size=4096)):
                    fd.write(chunk)
        shutil.unpack_archive(dataset_file, target_dir)
        os.remove(dataset_file)
        return True
    else:
        return False

def list_models():
    """
    Lists every implemented model and subgroups of the QUBA model Zoo
    """
    model_list = ['AlexNet', 'GoogLeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'VGG11-bn', 'VGG13-bn', 'VGG16-bn', 'VGG19-bn', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'WRN-50-2', 'WRN-101-2', 'SqueezeNet', 'InceptionV3', 'InceptionV4', 'Inception-ResNetv2', 'ResNeXt50-32x4d', 'ResNeXt101-32x8d', 'ResNeXt101-64x4d', 'DenseNet121', 
        'DenseNet161', 'DenseNet169', 'DenseNet201', 'Xception', 'MobileNetV2', 'ShuffleNet-v2-05', 'ShuffleNet-v2-1', 'ShuffleNet-v2-15', 'ShuffleNet-v2-2', 'NasNet-l', 'MobileNetV3-s', 'MobileNetV3-l', 'MobileNetV3-l-21k', 'BagNet9', 'BagNet17', 'BagNet33', 'MnasNet-05', 'MnasNet-075', 'MnasNet-1', 'MnasNet-13', 'EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2', 'EfficientNet-B3', 
        'EfficientNet-B4', 'EfficientNet-B5', 'EfficientNet-B6', 'EfficientNet-B7', 'NS-EfficientNet-B0', 'NS-EfficientNet-B1', 'NS-EfficientNet-B2', 'NS-EfficientNet-B3', 'NS-EfficientNet-B4', 'NS-EfficientNet-B5', 'NS-EfficientNet-B6', 'NS-EfficientNet-B7', 
        'Salman2020Do-RN50-2', 'Salman2020Do-RN50', 'BiTM-resnetv2-50x1', 'BiTM-resnetv2-50x3', 'BiTM-resnetv2-101x1', 'BiTM-resnetv2-152x2', 'RegNet-y-400mf', 'RegNet-y-800mf', 'RegNet-y-1-6gf', 'RegNet-y-3-2gf', 'RegNet-y-8gf', 'RegNet-y-16gf', 'RegNet-y-32gf', 'ViT-b-16', 'ViT-l-16', 'ViT-b-32', 'ViT-l-32', 'Swin-T', 'Swin-S', 'Swin-B', 'EfficientNet-v2-S', 'EfficientNet-v2-S-21k', 'EfficientNet-v2-M', 'EfficientNet-v2-M-21k', 'EfficientNet-v2-L', 'EfficientNet-v2-L-21k', 'DeiT-t', 'DeiT-s', 'DeiT-b', 'ConViT-t', 'ConViT-s', 'ConViT-b', 'CaiT-xxs24', 'CaiT-xs24', 'CaiT-s24', 'CrossViT-9dagger', 'CrossViT-15dagger', 'CrossViT-18dagger', 'XCiT-s24-16', 'XCiT-m24-16', 'XCiT-l24-16', 'LeViT-128', 'LeViT-256', 'LeViT-384', 'PiT-t', 'PiT-xs', 'PiT-s', 'PiT-b', 'CoaT-t-lite', 'CoaT-mi-lite', 'CoaT-s-lite', 'CoaT-me-lite', 'MaxViT-t', 'MaxViT-b', 'MaxViT-l', 'DeiT3-s', 'DeiT3-s-21k', 'DeiT3-m', 'DeiT3-m-21k', 'DeiT3-b', 'DeiT3-b-21k', 'DeiT3-l', 'DeiT3-l-21k', 'MViTv2-t', 'MViTv2-s', 'MViTv2-b', 'MViTv2-l', 'SwinV2-T-W8', 'SwinV2-S-W8', 'SwinV2-B-W8', 'SwinV2-t-W16', 'SwinV2-s-Win16', 'SwinV2-b-Win16', 'SwinV2-b-Win12to16-21k', 'SwinV2-l-Win12to16-21k', 'ViT-t5-16', 'ViT-t5-16-21k', 'ViT-t11-16', 'ViT-t11-16-21k', 'ViT-t21-16', 'ViT-t21-16-21k', 'ViT-s-16', 'ViT-s-16-21k', 'ViT-b-16-21k', 'ViT-b-32-21k', 'ViT-l-16-21k', 'ViT-l-32-21k', 'ConvNext-T', 'ConvNext-T-21k', 'ConvNext-S', 'ConvNext-S-21k', 'ConvNext-B', 'ConvNext-B-21k', 'ConvNext-L', 'ConvNext-L-21k', 'BeiT-b', 'EfficientFormer-l1', 'EfficientFormer-l3', 'EfficientFormer-l7', 'DaViT-t', 'DaViT-s', 'DaViT-b', 'Liu2023Comprehensive-Swin-B', 'Liu2023Comprehensive-Swin-L', 'Liu2023Comprehensive-ConvNeXt-B', 'Liu2023Comprehensive-ConvNeXt-L', 'Singh2023Revisiting-ConvNeXt-T-ConvStem', 'Singh2023Revisiting-ConvNeXt-S-ConvStem', 'Singh2023Revisiting-ConvNeXt-B-ConvStem', 'Singh2023Revisiting-ConvNeXt-L-ConvStem', 'Singh2023Revisiting-ViT-B-ConvStem', 
        'ConvNextV2-N', 'ConvNextV2-N-21k', 'ConvNextV2-T', 'ConvNextV2-T-21k', 'ConvNextV2-B', 'ConvNextV2-B-21k', 'ConvNextV2-L', 'ConvNextV2-L-21k', 'Hiera-T', 'Hiera-S', 'Hiera-B', 'Hiera-B-Plus', 'Hiera-L', 'EVA02-t-21k', 'EVA02-s-21k', 'EVA02-b-21k', 'InceptionNext-t', 'InceptionNext-s', 'InceptionNext-b', 'FastViT-sa12', 'FastViT-sa24', 'FastViT-sa36', 
        'BeiTV2-b', 'SeNet154', 'ResNet50d', 'ResNeXt50-32x4d-YFCCM100', 'ResNet50-yfcc100m', 'ResNet50-ig1B', 'ResNeXt101-32x8d-IG1B', 'ResNeXt50-32x4d-IG1B', 'ResNet18-IG1B', 
        'vit-b-16-mae-ft', 'vit-b-16-mae-lp', 'Hiera-B-LP', 'ViT-b-16-DINO-LP', 'ViTB-DINO-FT', 'ResNet50-DINO-LP', 'ResNet50-DINO-FT', 'ViT-l-14-dinoV2-LP', 
        'ViT-b-14-dinoV2-LP', 'ViT-s-14-dinoV2-LP', 'vit-t-16-21k', 'siglip-b-16', 
        'clip-resnet50', 'clip-vit-b-16', 'clip-resnet101', 'clip-vit-b-32', 
        'mobileclip-s0', 'mobileclip-s1', 'mobileclip-s2', 'mobileclip-b', 
        'EfficientNet-b0-A1', 'EfficientNet-b1-A1', 'EfficientNet-b2-A1', 'EfficientNet-b3-A1', 'EfficientNet-b4-A1', 'EfficientNetv2-M-A1', 'EfficientNetv2-S-A1', 'RegNety-040-A1', 'RegNety-080-A1', 'RegNety-160-A1', 'RegNety-320-A1', 'ResNet101-A1', 'ResNet152-A1', 'ResNet18-A1', 'ResNet34-A1', 'ResNet50-A1', 'ResNet50d-A1', 'ResNext50-32x4d-A1', 'SeNet154-A1', 'EfficientNet-b0-A2', 'EfficientNet-b1-A2', 'EfficientNet-b2-A2', 'EfficientNet-b3-A2', 'EfficientNet-b4-A2', 'EfficientNetv2-M-A2', 'EfficientNetv2-S-A2', 'RegNety-040-A2', 'RegNety-080-A2', 'RegNety-160-A2', 'RegNety-320-A2', 'ResNet101-A2', 'ResNet152-A2', 'ResNet18-A2', 'ResNet34-A2', 'ResNet50-A2', 'ResNet50d-A2', 'ResNext50-32x4d-A2', 'SeNet154-A2', 'EfficientNet-b0-A3', 'EfficientNet-b1-A3', 'EfficientNet-b2-A3', 'EfficientNet-b3-A3', 'EfficientNet-b4-A3', 'EfficientNetv2-M-A3', 'EfficientNetv2-S-A3', 'RegNety-040-A3', 'RegNety-080-A3', 'RegNety-160-A3', 'RegNety-320-A3', 'ResNet101-A3', 'ResNet152-A3', 'ResNet18-A3', 'ResNet34-A3', 'ResNet50-A3', 'ResNet50d-A3', 'ResNext50-32x4d-A3', 'SeNet154-A3', 
        'bcos-convnext-base', 'bcos-convnext-tiny', 'bcos-DenseNet121', 'bcos-DenseNet161', 'bcos-DenseNet169', 'bcos-DenseNet201', 'bcos-ResNet152', 'bcos-ResNet18', 'bcos-ResNet34', 'bcos-ResNet50', 'bcos-simple-vit-b-patch16-224', 'RegNet-y-4gf', 'mobileclip-blt', 'ViT-s-16-DINO-LP', 'siglip-l-16', 'bcos-ResNet101', 'metaclip-b16', 'convnext-large-d-clip', 'metaclip-l14', 'Singh2023Revisiting-ViT-S-ConvStem', 'convnext-base-w-320-clip', 'convnext-large-d-320-clip', 'Hiera-S-LP', 'Hiera-T-LP',
        "ViT-l-14-dinoV2-FT", "ViT-b-14-dinoV2-FT", "ViT-s-14-dinoV2-FT", "ViT-l-14-dinoV2-FT-Reg", "ViT-b-14-dinoV2-FT-Reg", "ViT-s-14-dinoV2-FT-Reg", "CLIP-B16-V-OpenAI", "CLIP-B16-V-Laion2B", "CLIP-B32-V-OpenAI", "CLIP-B32-V-Laion2B", "ViT-l-14-dinov2-reg-LP", "ViT-b-14-dinov2-reg-LP", "ViT-s-14-dinov2-reg-LP",
        "CLIP-B16-DataCompXL", "CLIP-B16-Laion2B", "CLIP-B16-CommonPool-XL-DFN2B", 
        "CLIP-L14-OpenAI", "CLIP-L14-DataCompXL", "CLIP-L14-Laion2B", "CLIP-L14-CommonPool-XL-DFN2B",
        "ViT-B-16-SigLIP2", "ViT-L-16-SigLIP2-256"
    ]
    group_list = [key for key in con.MODEL_MAP]
    table = {"Models": model_list, "Groups": group_list}
    return table

def download_file(url, base_dir=".", sub_dir="checkpoint", new_name=None):
    import wget
    """
    Downloads a file from a URL, ensures the directory structure exists,
    and handles renaming if the file already exists.
    
    Args:
        url (str): The URL of the file to download.
        base_dir (str): The base directory for saving files.
        sub_dir (str): Sub-directory where files should be saved.
        new_name (str): New name for the downloaded file (optional).
    """
    save_dir = os.path.join(base_dir, sub_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    temp_path = wget.download(url, out=save_dir)
    print(f"\nDownloaded to {temp_path}")

    if not new_name:
        new_name = os.path.basename(temp_path) 
    
    final_path = os.path.join(save_dir, new_name)

    if os.path.exists(final_path):
        return
    
    os.rename(temp_path, final_path)
    print(f"File saved as {final_path}")

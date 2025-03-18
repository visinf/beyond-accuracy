#Paths
_DATA_DIR = "DUMMY"
_IMAGENET1K_DIR = _DATA_DIR + "imagenet/val/" 
_IMAGENETC_DIR = _DATA_DIR + "imagenet-c/"
_MODEL_VS_HUMAN_DIR = _DATA_DIR + "modelvshuman"
_IMAGENETR_DIR = _DATA_DIR + "imagenet-r"
_MIXED_RAND_DIR = _DATA_DIR + "bg_challenge/mixed_rand/val" 
_MIXED_SAME_DIR = _DATA_DIR + "bg_challenge/mixed_same/val"

#Checkpoints
MAE_LP = "DUMMY"
DINO_RESNET_FT = "DUMMY"
DINO_VIT_FT = "DUMMY"
HIERA_LP_T = "DUMMY"
HIERA_LP_S = "DUMMY"
HIERA_LP_B = "DUMMY"
DINOV2_S_FT = "DUMMY"
DINOV2_B_FT = "DUMMY"
DINOV2_L_FT = "DUMMY"
DINOV2_S_REG_FT = "DUMMY"
DINOV2_B_REG_FT = "DUMMY"
DINOV2_L_REG_FT = "DUMMY"

RSB_ROOT = "./checkpoint/"
RSB_LINK = "https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-rsb-weights/"
MOBILECLIP_ROOT = "./checkpoint/"
MAE_FT = "./checkpoint/mae_finetuned_vit_base.pth"

from os.path import join as pjoin

_DATASET_DIR = _MODEL_VS_HUMAN_DIR
_PROJ_DIR = "/visinf/home/vimb06/code/beyond-accuracy/helper" #Path to helper directory
_CODE_DIR = pjoin(_PROJ_DIR, "modelvshuman")
_RAW_DATA_DIR = pjoin(_PROJ_DIR, "raw-data")

_SELF_SL_ALL = ['BeiT-b', 'ConvNextV2-N', 'ConvNextV2-N-21k', 'ConvNextV2-T', 'ConvNextV2-T-21k', 'ConvNextV2-B', 'ConvNextV2-B-21k', 'ConvNextV2-L', 'ConvNextV2-L-21k', 'Hiera-T', 'Hiera-S', 'Hiera-B', 'Hiera-B-Plus', 'Hiera-L', 'EVA02-t-21k', 'EVA02-s-21k', 'EVA02-b-21k', 'BeiTV2-b', 'vit-b-16-mae-ft', 'vit-b-16-mae-lp', 'Hiera-B-LP', 'ViT-b-16-DINO-LP', 'ViTB-DINO-FT', 'ResNet50-DINO-LP', 'ResNet50-DINO-FT', 'ViT-l-14-dinoV2-LP', 'ViT-b-14-dinoV2-LP', 'ViT-s-14-dinoV2-LP', 'siglip-b-16', 'clip-resnet50', 'clip-vit-b-16', 'clip-resnet101', 'clip-vit-b-32', 'mobileclip-s0', 'mobileclip-s1', 'mobileclip-s2', 'mobileclip-b', 'mobileclip-blt', 'ViT-s-16-DINO-LP', 'siglip-l-16', 'metaclip-b16', 'convnext-large-d-clip', 'metaclip-l14', 'convnext-base-w-320-clip', 'convnext-large-d-320-clip', 'Hiera-S-LP', 'Hiera-T-LP', "ViT-l-14-dinoV2-FT", "ViT-b-14-dinoV2-FT", "ViT-s-14-dinoV2-FT", "ViT-l-14-dinoV2-FT-Reg", "ViT-b-14-dinoV2-FT-Reg", "ViT-s-14-dinoV2-FT-Reg", "CLIP-B16-V-OpenAI", "CLIP-B16-V-Laion2B", "CLIP-B32-V-OpenAI", "CLIP-B32-V-Laion2B", "ViT-l-14-dinov2-reg-LP", "ViT-b-14-dinov2-reg-LP", "ViT-s-14-dinov2-reg-LP",]
_SELF_SL_LP = ['vit-b-16-mae-lp', 'Hiera-B-LP', 'ViT-b-16-DINO-LP', 'ResNet50-DINO-LP', 'ViT-l-14-dinoV2-LP', 'ViT-b-14-dinoV2-LP', 'ViT-s-14-dinoV2-LP', 'ViT-s-16-DINO-LP', 'Hiera-S-LP', 'Hiera-T-LP', "ViT-l-14-dinov2-reg-LP", "ViT-b-14-dinov2-reg-LP", "ViT-s-14-dinov2-reg-LP"]
_SELF_SL_FT = ['BeiT-b', 'ConvNextV2-N', 'ConvNextV2-N-21k', 'ConvNextV2-T', 'ConvNextV2-T-21k', 'ConvNextV2-B', 'ConvNextV2-B-21k', 'ConvNextV2-L', 'ConvNextV2-L-21k', 'Hiera-T', 'Hiera-S', 'Hiera-B', 'Hiera-B-Plus', 'Hiera-L', 'EVA02-t-21k', 'EVA02-s-21k', 'EVA02-b-21k', 'BeiTV2-b', 'vit-b-16-mae-ft', 'ViTB-DINO-FT', 'ResNet50-DINO-FT', "ViT-l-14-dinoV2-FT", "ViT-b-14-dinoV2-FT", "ViT-s-14-dinoV2-FT", "ViT-l-14-dinoV2-FT-Reg", "ViT-b-14-dinoV2-FT-Reg", "ViT-s-14-dinoV2-FT-Reg", "CLIP-B16-V-OpenAI", "CLIP-B16-V-Laion2B", "CLIP-B32-V-OpenAI", "CLIP-B32-V-Laion2B",]
_AT = ['Salman2020Do-RN50-2', 'Salman2020Do-RN50', 'Liu2023Comprehensive-Swin-B', 'Liu2023Comprehensive-Swin-L', 'Liu2023Comprehensive-ConvNeXt-B', 'Liu2023Comprehensive-ConvNeXt-L', 'Singh2023Revisiting-ConvNeXt-T-ConvStem', 'Singh2023Revisiting-ConvNeXt-S-ConvStem', 'Singh2023Revisiting-ConvNeXt-B-ConvStem', 'Singh2023Revisiting-ConvNeXt-L-ConvStem', 'Singh2023Revisiting-ViT-B-ConvStem', 'Singh2023Revisiting-ViT-S-ConvStem']
_SL = ['AlexNet', 'GoogLeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'VGG11-bn', 'VGG13-bn', 'VGG16-bn', 'VGG19-bn', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'WRN-50-2', 'WRN-101-2', 'SqueezeNet', 'InceptionV3', 'InceptionV4', 'Inception-ResNetv2', 'ResNeXt50-32x4d', 'ResNeXt101-32x8d', 'ResNeXt101-64x4d', 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201', 'Xception', 'MobileNetV2', 'ShuffleNet-v2-05', 'ShuffleNet-v2-1', 'ShuffleNet-v2-15', 'ShuffleNet-v2-2', 'NasNet-l', 'MobileNetV3-s', 'MobileNetV3-l', 'MobileNetV3-l-21k', 'BagNet9', 'BagNet17', 'BagNet33', 'MnasNet-05', 'MnasNet-075', 'MnasNet-1', 'MnasNet-13', 'EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2', 'EfficientNet-B3', 'EfficientNet-B4', 'EfficientNet-B5', 'EfficientNet-B6', 'EfficientNet-B7', 'BiTM-resnetv2-50x1', 'BiTM-resnetv2-50x3', 'BiTM-resnetv2-101x1', 'BiTM-resnetv2-152x2', 'RegNet-y-400mf', 'RegNet-y-800mf', 'RegNet-y-1-6gf', 'RegNet-y-3-2gf', 'RegNet-y-8gf', 'RegNet-y-16gf', 'RegNet-y-32gf', 'ViT-b-16', 'ViT-l-16', 'ViT-b-32', 'ViT-l-32', 'Swin-T', 'Swin-S', 'Swin-B', 'EfficientNet-v2-S', 'EfficientNet-v2-S-21k', 'EfficientNet-v2-M', 'EfficientNet-v2-M-21k', 'EfficientNet-v2-L', 'EfficientNet-v2-L-21k', 'DeiT-t', 'DeiT-s', 'DeiT-b', 'ConViT-t', 'ConViT-s', 'ConViT-b', 'CaiT-xxs24', 'CaiT-xs24', 'CaiT-s24', 'CrossViT-9dagger', 'CrossViT-15dagger', 'CrossViT-18dagger', 'XCiT-s24-16', 'XCiT-m24-16', 'XCiT-l24-16', 'LeViT-128', 'LeViT-256', 'LeViT-384', 'PiT-t', 'PiT-xs', 'PiT-s', 'PiT-b', 'CoaT-t-lite', 'CoaT-mi-lite', 'CoaT-s-lite', 'CoaT-me-lite', 'MaxViT-t', 'MaxViT-b', 'MaxViT-l', 'DeiT3-s', 'DeiT3-s-21k', 'DeiT3-m', 'DeiT3-m-21k', 'DeiT3-b', 'DeiT3-b-21k', 'DeiT3-l', 'DeiT3-l-21k', 'MViTv2-t', 'MViTv2-s', 'MViTv2-b', 'MViTv2-l', 'SwinV2-T-W8', 'SwinV2-S-W8', 'SwinV2-B-W8', 'SwinV2-t-W16', 'SwinV2-s-Win16', 'SwinV2-b-Win16', 'SwinV2-b-Win12to16-21k', 'SwinV2-l-Win12to16-21k', 'ViT-t5-16', 'ViT-t5-16-21k', 'ViT-t11-16', 'ViT-t11-16-21k', 'ViT-t21-16', 'ViT-t21-16-21k', 'ViT-s-16', 'ViT-s-16-21k', 'ViT-b-16-21k', 'ViT-b-32-21k', 'ViT-l-16-21k', 'ViT-l-32-21k', 'ConvNext-T', 'ConvNext-T-21k', 'ConvNext-S', 'ConvNext-S-21k', 'ConvNext-B', 'ConvNext-B-21k', 'ConvNext-L', 'ConvNext-L-21k', 'EfficientFormer-l1', 'EfficientFormer-l3', 'EfficientFormer-l7', 'DaViT-t', 'DaViT-s', 'DaViT-b', 'InceptionNext-t', 'InceptionNext-s', 'InceptionNext-b', 'FastViT-sa12', 'FastViT-sa24', 'FastViT-sa36', 'SeNet154', 'ResNet50d', 'vit-t-16-21k', 'bcos-convnext-base', 'bcos-convnext-tiny', 'bcos-DenseNet121', 'bcos-DenseNet161', 'bcos-DenseNet169', 'bcos-DenseNet201', 'bcos-ResNet152', 'bcos-ResNet18', 'bcos-ResNet34', 'bcos-ResNet50', 'bcos-simple-vit-b-patch16-224', 'RegNet-y-4gf', 'bcos-ResNet101']
_SEMISL = ['NS-EfficientNet-B0', 'NS-EfficientNet-B1', 'NS-EfficientNet-B2', 'NS-EfficientNet-B3', 'NS-EfficientNet-B4', 'NS-EfficientNet-B5', 'NS-EfficientNet-B6', 'NS-EfficientNet-B7', 'ResNeXt50-32x4d-YFCCM100', 'ResNet50-yfcc100m', 'ResNet50-ig1B', 'ResNeXt101-32x8d-IG1B', 'ResNeXt50-32x4d-IG1B', 'ResNet18-IG1B']
_A1 = ['EfficientNet-b0-A1', 'EfficientNet-b1-A1', 'EfficientNet-b2-A1', 'EfficientNet-b3-A1', 'EfficientNet-b4-A1', 'EfficientNetv2-M-A1', 'EfficientNetv2-S-A1', 'RegNety-040-A1', 'RegNety-080-A1', 'RegNety-160-A1', 'RegNety-320-A1', 'ResNet101-A1', 'ResNet152-A1', 'ResNet18-A1', 'ResNet34-A1', 'ResNet50-A1', 'ResNet50d-A1', 'ResNext50-32x4d-A1', 'SeNet154-A1']
_A2 = ['EfficientNet-b0-A2', 'EfficientNet-b1-A2', 'EfficientNet-b2-A2', 'EfficientNet-b3-A2', 'EfficientNet-b4-A2', 'EfficientNetv2-M-A2', 'EfficientNetv2-S-A2', 'RegNety-040-A2', 'RegNety-080-A2', 'RegNety-160-A2', 'RegNety-320-A2', 'ResNet101-A2', 'ResNet152-A2', 'ResNet18-A2', 'ResNet34-A2', 'ResNet50-A2', 'ResNet50d-A2', 'ResNext50-32x4d-A2', 'SeNet154-A2']
_A3 = ['EfficientNet-b0-A3', 'EfficientNet-b1-A3', 'EfficientNet-b2-A3', 'EfficientNet-b3-A3', 'EfficientNet-b4-A3', 'EfficientNetv2-M-A3', 'EfficientNetv2-S-A3', 'RegNety-040-A3', 'RegNety-080-A3', 'RegNety-160-A3', 'RegNety-320-A3', 'ResNet101-A3', 'ResNet152-A3', 'ResNet18-A3', 'ResNet34-A3', 'ResNet50-A3', 'ResNet50d-A3', 'ResNext50-32x4d-A3', 'SeNet154-A3']

_IN1K = ['AlexNet', 'GoogLeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'VGG11-bn', 'VGG13-bn', 'VGG16-bn', 'VGG19-bn', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'WRN-50-2', 'WRN-101-2', 'SqueezeNet', 'InceptionV3', 'InceptionV4', 'Inception-ResNetv2', 'ResNeXt50-32x4d', 'ResNeXt101-32x8d', 'ResNeXt101-64x4d', 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201', 'Xception', 'MobileNetV2', 'ShuffleNet-v2-05', 'ShuffleNet-v2-1', 'ShuffleNet-v2-15', 'ShuffleNet-v2-2', 'NasNet-l', 'MobileNetV3-s', 'MobileNetV3-l', 'BagNet9', 'BagNet17', 'BagNet33', 'MnasNet-05', 'MnasNet-075', 'MnasNet-1', 'MnasNet-13', 'EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2', 'EfficientNet-B3', 'EfficientNet-B4', 'EfficientNet-B5', 'EfficientNet-B6', 'EfficientNet-B7', 'Salman2020Do-RN50-2', 'Salman2020Do-RN50', 'RegNet-y-400mf', 'RegNet-y-800mf', 'RegNet-y-1-6gf', 'RegNet-y-3-2gf', 'RegNet-y-8gf', 'RegNet-y-16gf', 'RegNet-y-32gf', 'ViT-b-16', 'ViT-l-16', 'ViT-b-32', 'ViT-l-32', 'Swin-T', 'Swin-S', 'Swin-B', 'EfficientNet-v2-S', 'EfficientNet-v2-M', 'EfficientNet-v2-L', 'DeiT-t', 'DeiT-s', 'DeiT-b', 'ConViT-t', 'ConViT-s', 'ConViT-b', 'CaiT-xxs24', 'CaiT-xs24', 'CaiT-s24', 'CrossViT-9dagger', 'CrossViT-15dagger', 'CrossViT-18dagger', 'XCiT-s24-16', 'XCiT-m24-16', 'XCiT-l24-16', 'LeViT-128', 'LeViT-256', 'LeViT-384', 'PiT-t', 'PiT-xs', 'PiT-s', 'PiT-b', 'CoaT-t-lite', 'CoaT-mi-lite', 'CoaT-s-lite', 'CoaT-me-lite', 'MaxViT-t', 'MaxViT-b', 'MaxViT-l', 'DeiT3-s', 'DeiT3-m', 'DeiT3-b', 'DeiT3-l', 'MViTv2-t', 'MViTv2-s', 'MViTv2-b', 'MViTv2-l', 'SwinV2-T-W8', 'SwinV2-S-W8', 'SwinV2-B-W8', 'SwinV2-t-W16', 'SwinV2-s-Win16', 'SwinV2-b-Win16', 'ViT-t5-16', 'ViT-t11-16', 'ViT-t21-16', 'ViT-s-16', 'ConvNext-T', 'ConvNext-S', 'ConvNext-B', 'ConvNext-L', 'EfficientFormer-l1', 'EfficientFormer-l3', 'EfficientFormer-l7', 'DaViT-t', 'DaViT-s', 'DaViT-b', 'Liu2023Comprehensive-Swin-B', 'Liu2023Comprehensive-Swin-L', 'Liu2023Comprehensive-ConvNeXt-B', 'Liu2023Comprehensive-ConvNeXt-L', 'Singh2023Revisiting-ConvNeXt-T-ConvStem', 'Singh2023Revisiting-ConvNeXt-S-ConvStem', 'Singh2023Revisiting-ConvNeXt-B-ConvStem', 'Singh2023Revisiting-ConvNeXt-L-ConvStem', 'Singh2023Revisiting-ViT-B-ConvStem', 'ConvNextV2-N', 'ConvNextV2-T', 'ConvNextV2-B', 'ConvNextV2-L', 'Hiera-T', 'Hiera-S', 'Hiera-B', 'Hiera-B-Plus', 'Hiera-L', 'InceptionNext-t', 'InceptionNext-s', 'InceptionNext-b', 'FastViT-sa12', 'FastViT-sa24', 'FastViT-sa36', 'BeiTV2-b', 'SeNet154', 'ResNet50d', 'vit-b-16-mae-ft', 'vit-b-16-mae-lp', 'Hiera-B-LP', 'ViT-b-16-DINO-LP', 'ViTB-DINO-FT', 'ResNet50-DINO-LP', 'ResNet50-DINO-FT', 'EfficientNet-b0-A1', 'EfficientNet-b1-A1', 'EfficientNet-b2-A1', 'EfficientNet-b3-A1', 'EfficientNet-b4-A1', 'EfficientNetv2-M-A1', 'EfficientNetv2-S-A1', 'RegNety-040-A1', 'RegNety-080-A1', 'RegNety-160-A1', 'RegNety-320-A1', 'ResNet101-A1', 'ResNet152-A1', 'ResNet18-A1', 'ResNet34-A1', 'ResNet50-A1', 'ResNet50d-A1', 'ResNext50-32x4d-A1', 'SeNet154-A1', 'EfficientNet-b0-A2', 'EfficientNet-b1-A2', 'EfficientNet-b2-A2', 'EfficientNet-b3-A2', 'EfficientNet-b4-A2', 'EfficientNetv2-M-A2', 'EfficientNetv2-S-A2', 'RegNety-040-A2', 'RegNety-080-A2', 'RegNety-160-A2', 'RegNety-320-A2', 'ResNet101-A2', 'ResNet152-A2', 'ResNet18-A2', 'ResNet34-A2', 'ResNet50-A2', 'ResNet50d-A2', 'ResNext50-32x4d-A2', 'SeNet154-A2', 'EfficientNet-b0-A3', 'EfficientNet-b1-A3', 'EfficientNet-b2-A3', 'EfficientNet-b3-A3', 'EfficientNet-b4-A3', 'EfficientNetv2-M-A3', 'EfficientNetv2-S-A3', 'RegNety-040-A3', 'RegNety-080-A3', 'RegNety-160-A3', 'RegNety-320-A3', 'ResNet101-A3', 'ResNet152-A3', 'ResNet18-A3', 'ResNet34-A3', 'ResNet50-A3', 'ResNet50d-A3', 'ResNext50-32x4d-A3', 'SeNet154-A3', 'bcos-convnext-base', 'bcos-convnext-tiny', 'bcos-DenseNet121', 'bcos-DenseNet161', 'bcos-DenseNet169', 'bcos-DenseNet201', 'bcos-ResNet152', 'bcos-ResNet18', 'bcos-ResNet34', 'bcos-ResNet50', 'bcos-simple-vit-b-patch16-224', 'RegNet-y-4gf', 'ViT-s-16-DINO-LP', 'bcos-ResNet101', 'Singh2023Revisiting-ViT-S-ConvStem', 'Hiera-S-LP', 'Hiera-T-LP']
_IN21K = ['MobileNetV3-l-21k', 'BiTM-resnetv2-50x1', 'BiTM-resnetv2-50x3', 'BiTM-resnetv2-101x1', 'BiTM-resnetv2-152x2', 'EfficientNet-v2-S-21k', 'EfficientNet-v2-M-21k', 'EfficientNet-v2-L-21k', 'DeiT3-s-21k', 'DeiT3-m-21k', 'DeiT3-b-21k', 'DeiT3-l-21k', 'SwinV2-b-Win12to16-21k', 'SwinV2-l-Win12to16-21k', 'ViT-t5-16-21k', 'ViT-t11-16-21k', 'ViT-t21-16-21k', 'ViT-s-16-21k', 'ViT-b-16-21k', 'ViT-b-32-21k', 'ViT-l-16-21k', 'ViT-l-32-21k', 'ConvNext-T-21k', 'ConvNext-S-21k', 'ConvNext-B-21k', 'ConvNext-L-21k', 'BeiT-b', 'ConvNextV2-N-21k', 'ConvNextV2-T-21k', 'ConvNextV2-B-21k', 'ConvNextV2-L-21k', 'EVA02-t-21k', 'EVA02-s-21k', 'EVA02-b-21k', 'vit-t-16-21k']
_BD = ['NS-EfficientNet-B0', 'NS-EfficientNet-B1', 'NS-EfficientNet-B2', 'NS-EfficientNet-B3', 'NS-EfficientNet-B4', 'NS-EfficientNet-B5', 'NS-EfficientNet-B6', 'NS-EfficientNet-B7', 'ResNeXt50-32x4d-YFCCM100', 'ResNet50-yfcc100m', 'ResNet50-ig1B', 'ResNeXt101-32x8d-IG1B', 'ResNeXt50-32x4d-IG1B', 'ResNet18-IG1B', 'ViT-l-14-dinoV2-LP', 'ViT-b-14-dinoV2-LP', 'ViT-s-14-dinoV2-LP', 'siglip-b-16', 'clip-resnet50', 'clip-vit-b-16', 'clip-resnet101', 'clip-vit-b-32', 'mobileclip-s0', 'mobileclip-s1', 'mobileclip-s2', 'mobileclip-b', 'mobileclip-blt', 'siglip-l-16', 'metaclip-b16', 'convnext-large-d-clip', 'metaclip-l14', 'convnext-base-w-320-clip', 'convnext-large-d-320-clip']

_VIL = ['siglip-b-16', 'clip-resnet50', 'clip-vit-b-16', 'clip-resnet101', 'clip-vit-b-32', 'mobileclip-s0', 'mobileclip-s1', 'mobileclip-s2', 'mobileclip-b', 'mobileclip-blt', 'siglip-l-16', 'metaclip-b16', 'convnext-large-d-clip', 'metaclip-l14', 'convnext-base-w-320-clip', 'convnext-large-d-320-clip', "CLIP-B16-DataCompXL", "CLIP-B16-Laion2B", "CLIP-B16-CommonPool-XL-DFN2B", 
        "CLIP-L14-OpenAI", "CLIP-L14-DataCompXL", "CLIP-L14-Laion2B", "CLIP-L14-CommonPool-XL-DFN2B",
        "ViT-B-16-SigLIP2", "ViT-L-16-SigLIP2-256"]
_CNN = ['AlexNet', 'GoogLeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'VGG11-bn', 'VGG13-bn', 'VGG16-bn', 'VGG19-bn', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'WRN-50-2', 'WRN-101-2', 'SqueezeNet', 'InceptionV3', 'InceptionV4', 'Inception-ResNetv2', 'ResNeXt50-32x4d', 'ResNeXt101-32x8d', 'ResNeXt101-64x4d', 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201', 'Xception', 'MobileNetV2', 'ShuffleNet-v2-05', 'ShuffleNet-v2-1', 'ShuffleNet-v2-15', 'ShuffleNet-v2-2', 'NasNet-l', 'MobileNetV3-s', 'MobileNetV3-l', 'MobileNetV3-l-21k', 'BagNet9', 'BagNet17', 'BagNet33', 'MnasNet-05', 'MnasNet-075', 'MnasNet-1', 'MnasNet-13', 'EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2', 'EfficientNet-B3', 'EfficientNet-B4', 'EfficientNet-B5', 'EfficientNet-B6', 'EfficientNet-B7', 'NS-EfficientNet-B0', 'NS-EfficientNet-B1', 'NS-EfficientNet-B2', 'NS-EfficientNet-B3', 'NS-EfficientNet-B4', 'NS-EfficientNet-B5', 'NS-EfficientNet-B6', 'NS-EfficientNet-B7', 'Salman2020Do-RN50-2', 'Salman2020Do-RN50', 'BiTM-resnetv2-50x1', 'BiTM-resnetv2-50x3', 'BiTM-resnetv2-101x1', 'BiTM-resnetv2-152x2', 'RegNet-y-400mf', 'RegNet-y-800mf', 'RegNet-y-1-6gf', 'RegNet-y-3-2gf', 'RegNet-y-8gf', 'RegNet-y-16gf', 'RegNet-y-32gf', 'EfficientNet-v2-S', 'EfficientNet-v2-S-21k', 'EfficientNet-v2-M', 'EfficientNet-v2-M-21k', 'EfficientNet-v2-L', 'EfficientNet-v2-L-21k', 'ConvNext-T', 'ConvNext-T-21k', 'ConvNext-S', 'ConvNext-S-21k', 'ConvNext-B', 'ConvNext-B-21k', 'ConvNext-L', 'ConvNext-L-21k', 'Liu2023Comprehensive-ConvNeXt-B', 'Liu2023Comprehensive-ConvNeXt-L', 'Singh2023Revisiting-ConvNeXt-T-ConvStem', 'Singh2023Revisiting-ConvNeXt-S-ConvStem', 'Singh2023Revisiting-ConvNeXt-B-ConvStem', 'Singh2023Revisiting-ConvNeXt-L-ConvStem', 'ConvNextV2-N', 'ConvNextV2-N-21k', 'ConvNextV2-T', 'ConvNextV2-T-21k', 'ConvNextV2-B', 'ConvNextV2-B-21k', 'ConvNextV2-L', 'ConvNextV2-L-21k', 'SeNet154', 'ResNet50d', 'ResNeXt50-32x4d-YFCCM100', 'ResNet50-yfcc100m', 'ResNet50-ig1B', 'ResNeXt101-32x8d-IG1B', 'ResNeXt50-32x4d-IG1B', 'ResNet18-IG1B', 'ResNet50-DINO-LP', 'ResNet50-DINO-FT', 'EfficientNet-b0-A1', 'EfficientNet-b1-A1', 'EfficientNet-b2-A1', 'EfficientNet-b3-A1', 'EfficientNet-b4-A1', 'EfficientNetv2-M-A1', 'EfficientNetv2-S-A1', 'RegNety-040-A1', 'RegNety-080-A1', 'RegNety-160-A1', 'RegNety-320-A1', 'ResNet101-A1', 'ResNet152-A1', 'ResNet18-A1', 'ResNet34-A1', 'ResNet50-A1', 'ResNet50d-A1', 'ResNext50-32x4d-A1', 'SeNet154-A1', 'EfficientNet-b0-A2', 'EfficientNet-b1-A2', 'EfficientNet-b2-A2', 'EfficientNet-b3-A2', 'EfficientNet-b4-A2', 'EfficientNetv2-M-A2', 'EfficientNetv2-S-A2', 'RegNety-040-A2', 'RegNety-080-A2', 'RegNety-160-A2', 'RegNety-320-A2', 'ResNet101-A2', 'ResNet152-A2', 'ResNet18-A2', 'ResNet34-A2', 'ResNet50-A2', 'ResNet50d-A2', 'ResNext50-32x4d-A2', 'SeNet154-A2', 'EfficientNet-b0-A3', 'EfficientNet-b1-A3', 'EfficientNet-b2-A3', 'EfficientNet-b3-A3', 'EfficientNet-b4-A3', 'EfficientNetv2-M-A3', 'EfficientNetv2-S-A3', 'RegNety-040-A3', 'RegNety-080-A3', 'RegNety-160-A3', 'RegNety-320-A3', 'ResNet101-A3', 'ResNet152-A3', 'ResNet18-A3', 'ResNet34-A3', 'ResNet50-A3', 'ResNet50d-A3', 'ResNext50-32x4d-A3', 'SeNet154-A3', 'RegNet-y-4gf']
_TRA = ['ViT-b-16', 'ViT-l-16', 'ViT-b-32', 'ViT-l-32', 'Swin-T', 'Swin-S', 'Swin-B', 'DeiT-t', 'DeiT-s', 'DeiT-b', 'ConViT-t', 'ConViT-s', 'ConViT-b', 'CaiT-xxs24', 'CaiT-xs24', 'CaiT-s24', 'CrossViT-9dagger', 'CrossViT-15dagger', 'CrossViT-18dagger', 'XCiT-s24-16', 'XCiT-m24-16', 'XCiT-l24-16', 'LeViT-128', 'LeViT-256', 'LeViT-384', 'PiT-t', 'PiT-xs', 'PiT-s', 'PiT-b', 'CoaT-t-lite', 'CoaT-mi-lite', 'CoaT-s-lite', 'CoaT-me-lite', 'MaxViT-t', 'MaxViT-b', 'MaxViT-l', 'DeiT3-s', 'DeiT3-s-21k', 'DeiT3-m', 'DeiT3-m-21k', 'DeiT3-b', 'DeiT3-b-21k', 'DeiT3-l', 'DeiT3-l-21k', 'MViTv2-t', 'MViTv2-s', 'MViTv2-b', 'MViTv2-l', 'SwinV2-T-W8', 'SwinV2-S-W8', 'SwinV2-B-W8', 'SwinV2-t-W16', 'SwinV2-s-Win16', 'SwinV2-b-Win16', 'SwinV2-b-Win12to16-21k', 'SwinV2-l-Win12to16-21k', 'ViT-t5-16', 'ViT-t5-16-21k', 'ViT-t11-16', 'ViT-t11-16-21k', 'ViT-t21-16', 'ViT-t21-16-21k', 'ViT-s-16', 'ViT-s-16-21k', 'ViT-b-16-21k', 'ViT-b-32-21k', 'ViT-l-16-21k', 'ViT-l-32-21k', 'BeiT-b', 'EfficientFormer-l1', 'EfficientFormer-l3', 'EfficientFormer-l7', 'DaViT-t', 'DaViT-s', 'DaViT-b', 'Liu2023Comprehensive-Swin-B', 'Liu2023Comprehensive-Swin-L', 'Singh2023Revisiting-ViT-B-ConvStem', 'Hiera-T', 'Hiera-S', 'Hiera-B', 'Hiera-B-Plus', 'Hiera-L', 'EVA02-t-21k', 'EVA02-s-21k', 'EVA02-b-21k', 'InceptionNext-t', 'InceptionNext-s', 'InceptionNext-b', 'FastViT-sa12', 'FastViT-sa24', 'FastViT-sa36', 'BeiTV2-b', 'vit-b-16-mae-ft', 'vit-b-16-mae-lp', 'Hiera-B-LP', 'ViT-b-16-DINO-LP', 'ViTB-DINO-FT', 'ViT-l-14-dinoV2-LP', 'ViT-b-14-dinoV2-LP', 'ViT-s-14-dinoV2-LP', 'vit-t-16-21k', 'ViT-s-16-DINO-LP', 'Singh2023Revisiting-ViT-S-ConvStem', 'Hiera-S-LP', 'Hiera-T-LP']
_BCOS = ['bcos-convnext-base', 'bcos-convnext-tiny', 'bcos-DenseNet121', 'bcos-DenseNet161', 'bcos-DenseNet169', 'bcos-DenseNet201', 'bcos-ResNet152', 'bcos-ResNet18', 'bcos-ResNet34', 'bcos-ResNet50', 'bcos-simple-vit-b-patch16-224', 'bcos-ResNet101']

_ALL = ['AlexNet', 'GoogLeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'VGG11-bn', 'VGG13-bn', 'VGG16-bn', 'VGG19-bn', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'WRN-50-2', 'WRN-101-2', 'SqueezeNet', 'InceptionV3', 'InceptionV4', 'Inception-ResNetv2', 'ResNeXt50-32x4d', 'ResNeXt101-32x8d', 'ResNeXt101-64x4d', 'DenseNet121', 
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

OPEN_CLIP_MODELS = {
    # MetaCLIP
    "metaclip-b16": "hf-hub:timm/vit_base_patch16_clip_224.metaclip_2pt5b",
    "metaclip-l14": "hf-hub:timm/vit_large_patch14_clip_224.metaclip_2pt5b",

    # CLIP with ConvNext Backbone
    "convnext-large-d-320-clip": "hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
    "convnext-large-d-clip": "hf-hub:laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg",
    "convnext-base-w-320-clip": "hf-hub:laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg",

    # CLIP trained on Laion2B
    "clip-laion2b-b16": "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "clip-laion2b-b32": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "clip-laion2b-l14": "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",

    # Commonpool 12.8B Variants
    "clip-dfn2b-l14": "hf-hub:apple/DFN2B-CLIP-ViT-L-14",
    "clip-dfn2b-b16": "hf-hub:apple/DFN2B-CLIP-ViT-B-16",

    # CommonPool L Models
    "clip-commonpool-l-b16": "hf-hub:laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K",
    "clip-commonpool-l-basic-b16": "hf-hub:laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K",
    "clip-commonpool-l-text-b16": "hf-hub:laion/CLIP-ViT-B-16-CommonPool.L.text-s1B-b8K",
    "clip-commonpool-l-image-b16": "hf-hub:laion/CLIP-ViT-B-16-CommonPool.L.image-s1B-b8K",
    "clip-commonpool-l-laion-b16": "hf-hub:laion/CLIP-ViT-B-16-CommonPool.L.laion-s1B-b8K",
    "clip-commonpool-l-clip-b16": "hf-hub:laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K",
    "clip-datacomp-l-b16": "hf-hub:laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K",

    # Commonpool XL
    "clip-datacomp-xl-l14": "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", 
    "clip-commonpool-xl-clip-l14": "hf-hub:laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K",
    "clip-commonpool-xl-laion-l14": "hf-hub:laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K",
    "clip-commonpool-xl-l14": "hf-hub:laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K",
    "clip-datacomp-xl-b16": "hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    "clip-datacomp-xl-b32": "hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",

    # "clip-datacomp-m-b32", "clip-commonpool-m-clip-b32", "clip-commonpool-m-laion-b32", "clip-commonpool-m-image-b32", "clip-commonpool-m-text-b32", "clip-commonpool-m-basic-b32", "clip-commonpool-m-nofilter-b32", "clip-datacomp-s-b32", "clip-commonpool-s-clip-b32", "clip-commonpool-s-laion-b32", "clip-commonpool-s-image-b32", "clip-commonpool-s-text-b32", "clip-commonpool-s-basic-b32", "clip-commonpool-s-nofilter-b32",
    
    # CommonPool M
    "clip-datacomp-m-b32": "hf-hub:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K", 
    "clip-commonpool-m-clip-b32": "hf-hub:laion/CLIP-ViT-B-32-CommonPool.M.clip-s128M-b4K",
    "clip-commonpool-m-laion-b32": "hf-hub:laion/CLIP-ViT-B-32-CommonPool.M.laion-s128M-b4K", 
    "clip-commonpool-m-image-b32": "hf-hub:laion/CLIP-ViT-B-32-CommonPool.M.image-s128M-b4K",
    "clip-commonpool-m-text-b32": "hf-hub:laion/CLIP-ViT-B-32-CommonPool.M.text-s128M-b4K", 
    "clip-commonpool-m-basic-b32": "hf-hub:laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K", 
    "clip-commonpool-m-nofilter-b32": "hf-hub:laion/CLIP-ViT-B-32-CommonPool.M-s128M-b4K", 

    # CommonPool S
    "clip-datacomp-s-b32": "hf-hub:laion/CLIP-ViT-B-32-DataComp.S-s13M-b4K", 
    "clip-commonpool-s-clip-b32": "hf-hub:laion/CLIP-ViT-B-32-CommonPool.S.clip-s13M-b4K", 
    "clip-commonpool-s-laion-b32": "hf-hub:laion/CLIP-ViT-B-32-CommonPool.S.laion-s13M-b4K", 
    "clip-commonpool-s-image-b32": "hf-hub:laion/CLIP-ViT-B-32-CommonPool.S.image-s13M-b4K", 
    "clip-commonpool-s-text-b32": "hf-hub:laion/CLIP-ViT-B-32-CommonPool.S.text-s13M-b4K", 
    "clip-commonpool-s-basic-b32": "hf-hub:laion/CLIP-ViT-B-32-CommonPool.S.basic-s13M-b4K", 
    "clip-commonpool-s-nofilter-b32": "hf-hub:laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K", 
}

SIGLIP_MODELS = {
    "siglip-b-16": 'hf-hub:timm/ViT-B-16-SigLIP',
    "siglip-l-16": 'hf-hub:timm/ViT-L-16-SigLIP',
}

SIGLIP2_MODELS = {
    "ViT-B-16-SigLIP2": "ViT-B-16-SigLIP2", 
    "ViT-L-16-SigLIP2-256": "ViT-L-16-SigLIP2-256"
}

import bagnets.pytorchnet
from robustbench import utils as rob_bench_utils
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import timm
import torchvision.models as models
import bcos.models.pretrained as bcosmodels
import torch
import torchvision.transforms as tv_transforms
import os
from bcos import transforms as bcos_transforms
import clip

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


def download_and_create_model(timm_name, url_filename, new_name):
    # Assuming `download_file` is a function that downloads the file
    download_file(url=RSB_LINK + url_filename, new_name=new_name)
    model = timm.create_model(timm_name, RSB_ROOT + new_name)
    return model, timm_name


HIERA_LP_DINOV1_TRANSFORM = tv_transforms.Compose([
                tv_transforms.Resize(256, interpolation=3),
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

TORCHVISION_MODELS = {
    "AlexNet": (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1),
    
    "GoogLeNet": (models.googlenet, models.GoogLeNet_Weights.IMAGENET1K_V1),
    
    "ResNet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "ResNet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "ResNet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "ResNet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
    "ResNet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),

    "SqueezeNet": (models.squeezenet1_1, models.SqueezeNet1_1_Weights.IMAGENET1K_V1),
    
    "ResNeXt50-32x4d": (models.resnext50_32x4d, models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
    "ResNeXt101-32x8d": (models.resnext101_32x8d, models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1),
    "ResNeXt101-64x4d": (models.resnext101_64x4d, models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1),
    
    "DenseNet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    "DenseNet161": (models.densenet161, models.DenseNet161_Weights.IMAGENET1K_V1),
    "DenseNet169": (models.densenet169, models.DenseNet169_Weights.IMAGENET1K_V1),
    "DenseNet201": (models.densenet201, models.DenseNet201_Weights.IMAGENET1K_V1),

    "VGG11": (models.vgg11, models.VGG11_Weights.IMAGENET1K_V1),
    "VGG13": (models.vgg13, models.VGG13_Weights.IMAGENET1K_V1),
    "VGG16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    "VGG19": (models.vgg19, models.VGG19_Weights.IMAGENET1K_V1),
    
    "VGG11-bn": (models.vgg11_bn, models.VGG11_BN_Weights.IMAGENET1K_V1),
    "VGG13-bn": (models.vgg13_bn, models.VGG13_BN_Weights.IMAGENET1K_V1),
    "VGG16-bn": (models.vgg16_bn, models.VGG16_BN_Weights.IMAGENET1K_V1),
    "VGG19-bn": (models.vgg19_bn, models.VGG19_BN_Weights.IMAGENET1K_V1),
    
    "ViT-b-16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
    "ViT-l-16": (models.vit_l_16, models.ViT_L_16_Weights.IMAGENET1K_V1),
    "ViT-b-32": (models.vit_b_32, models.ViT_B_32_Weights.IMAGENET1K_V1),
    "ViT-l-32": (models.vit_l_32, models.ViT_L_32_Weights.IMAGENET1K_V1),

    "WRN-50-2": (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.IMAGENET1K_V1),
    "WRN-101-2": (models.wide_resnet101_2, models.Wide_ResNet101_2_Weights.IMAGENET1K_V1),
    
    "Swin-T": (models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1),
    "Swin-S": (models.swin_s, models.Swin_S_Weights.IMAGENET1K_V1),
    "Swin-B": (models.swin_b, models.Swin_B_Weights.IMAGENET1K_V1),

    "SwinV2-T-W8": (models.swin_v2_t, models.Swin_V2_T_Weights.IMAGENET1K_V1),
    "SwinV2-S-W8": (models.swin_v2_s, models.Swin_V2_S_Weights.IMAGENET1K_V1),
    "SwinV2-B-W8": (models.swin_v2_b, models.Swin_V2_B_Weights.IMAGENET1K_V1),

    "ConvNext-T": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "ConvNext-S": (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1),
    "ConvNext-B": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
    "ConvNext-L": (models.convnext_large, models.ConvNeXt_Large_Weights.IMAGENET1K_V1),

    "InceptionV3": (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1),

    "MobileNetV2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
    "MobileNetV3-s": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "MobileNetV3-l": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),

    "MaxViT-t": (models.maxvit_t, models.MaxVit_T_Weights.IMAGENET1K_V1),

    "MnasNet-05": (models.mnasnet0_5, models.MNASNet0_5_Weights.IMAGENET1K_V1),
    "MnasNet-075": (models.mnasnet0_75, models.MNASNet0_75_Weights.IMAGENET1K_V1),
    "MnasNet-1": (models.mnasnet1_0, models.MNASNet1_0_Weights.IMAGENET1K_V1),
    "MnasNet-13": (models.mnasnet1_3, models.MNASNet1_3_Weights.IMAGENET1K_V1),
    "ShuffleNet-v2-05": (models.shufflenet_v2_x0_5, models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1),
    "ShuffleNet-v2-1": (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
    "ShuffleNet-v2-15": (models.shufflenet_v2_x1_5, models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1),
    "ShuffleNet-v2-2": (models.shufflenet_v2_x2_0, models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1),

    "RegNet-y-400mf": (models.regnet_y_400mf, models.RegNet_Y_400MF_Weights.IMAGENET1K_V1),
    "RegNet-y-800mf": (models.regnet_y_800mf, models.RegNet_Y_800MF_Weights.IMAGENET1K_V1),
    "RegNet-y-1-6gf": (models.regnet_y_1_6gf, models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1),
    "RegNet-y-3-2gf": (models.regnet_y_3_2gf, models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1),
    "RegNet-y-8gf": (models.regnet_y_8gf, models.RegNet_Y_8GF_Weights.IMAGENET1K_V1),
    "RegNet-y-16gf": (models.regnet_y_16gf, models.RegNet_Y_16GF_Weights.IMAGENET1K_V1),
    "RegNet-y-32gf": (models.regnet_y_32gf, models.RegNet_Y_32GF_Weights.IMAGENET1K_V1),

    "EfficientNet-B0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "EfficientNet-B1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V1),
    "EfficientNet-B2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1),
    "EfficientNet-B3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1),
    "EfficientNet-B4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1),
    "EfficientNet-B5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.IMAGENET1K_V1),
    "EfficientNet-B6": (models.efficientnet_b6, models.EfficientNet_B6_Weights.IMAGENET1K_V1),
    "EfficientNet-B7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.IMAGENET1K_V1)

}

OPENAI_CLIP = {
    "clip-resnet50": lambda: clip.load("RN50"),
    "clip-vit-b-16": lambda: clip.load("ViT-B/16"),
    "clip-resnet101": lambda: clip.load("RN101"),
    "clip-vit-b-32": lambda: clip.load("ViT-B/32"),
    "clip-vit-l-14": lambda: clip.load("ViT-L/14")
}

BAGNET_TRANSFORM = tv_transforms.Compose([
            tv_transforms.Resize(224),                  
            tv_transforms.CenterCrop(224),                
            tv_transforms.ToTensor(),                      
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
HIERA_TRANSFORM = tv_transforms.Compose([
            tv_transforms.Resize(224),                  
            tv_transforms.CenterCrop(224),                
            tv_transforms.ToTensor(),                      
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
BCOS_TRANSFORM = tv_transforms.Compose([tv_transforms.Resize(256), tv_transforms.CenterCrop(224), tv_transforms.ToTensor(), bcos_transforms.AddInverse()])
CUSTOM_MODELS = {
    "bcos-ResNet18": (lambda: bcosmodels.resnet18(pretrained=True), BCOS_TRANSFORM),
    "bcos-ResNet34": (lambda: bcosmodels.resnet34(pretrained=True), BCOS_TRANSFORM),
    "bcos-ResNet50": (lambda: bcosmodels.resnet50(pretrained=True), BCOS_TRANSFORM),
    "bcos-ResNet101": (lambda: bcosmodels.resnet101(pretrained=True), BCOS_TRANSFORM),
    "bcos-ResNet152": (lambda: bcosmodels.resnet152(pretrained=True), BCOS_TRANSFORM),
    "bcos-DenseNet121": (lambda: bcosmodels.densenet121(pretrained=True), BCOS_TRANSFORM),
    "bcos-DenseNet161": (lambda: bcosmodels.densenet161(pretrained=True), BCOS_TRANSFORM),
    "bcos-DenseNet169": (lambda: bcosmodels.densenet169(pretrained=True), BCOS_TRANSFORM),
    "bcos-DenseNet201": (lambda: bcosmodels.densenet201(pretrained=True), BCOS_TRANSFORM),
    "bcos-simple-vit-b-patch16-224": (lambda: bcosmodels.simple_vit_b_patch16_224(pretrained=True), BCOS_TRANSFORM),
    "bcos-convnext-tiny": (lambda: bcosmodels.convnext_tiny(pretrained=True), BCOS_TRANSFORM),
    "bcos-convnext-base": (lambda: bcosmodels.convnext_base(pretrained=True), BCOS_TRANSFORM),

    "BagNet9": (lambda: bagnets.pytorchnet.bagnet9(pretrained=True), BAGNET_TRANSFORM),
    "BagNet17": (lambda: bagnets.pytorchnet.bagnet17(pretrained=True), BAGNET_TRANSFORM),
    "BagNet33": (lambda: bagnets.pytorchnet.bagnet33(pretrained=True), BAGNET_TRANSFORM),

    "Hiera-T": (lambda: torch.hub.load("facebookresearch/hiera", model="hiera_tiny_224", pretrained=True, checkpoint="mae_in1k_ft_in1k"), HIERA_TRANSFORM),
    "Hiera-S": (lambda: torch.hub.load("facebookresearch/hiera", model="hiera_small_224", pretrained=True, checkpoint="mae_in1k_ft_in1k"), HIERA_TRANSFORM),
    "Hiera-B": (lambda: torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k_ft_in1k"), HIERA_TRANSFORM),
    "Hiera-B-Plus": (lambda: torch.hub.load("facebookresearch/hiera", model="hiera_base_plus_224", pretrained=True, checkpoint="mae_in1k_ft_in1k"), HIERA_TRANSFORM),    
    "Hiera-L": (lambda: torch.hub.load("facebookresearch/hiera", model="hiera_large_224", pretrained=True, checkpoint="mae_in1k_ft_in1k"), HIERA_TRANSFORM),
}
TIMM_MODELS = {

    "ViT-t5-16": "tiny_vit_5m_224.in1k",
    "ViT-t11-16": "tiny_vit_11m_224.in1k",
    "ViT-t21-16": "tiny_vit_21m_224.in1k",
    "ViT-t5-16-21k": "tiny_vit_5m_224.dist_in22k_ft_in1k",
    "ViT-t11-16-21k": "tiny_vit_11m_224.dist_in22k_ft_in1k",
    "ViT-t21-16-21k": "tiny_vit_21m_224.dist_in22k_ft_in1k",

    "ViT-s-16": "vit_small_patch16_224.augreg_in1k",
    "ViT-s-16-21k": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "ViT-b-16-21k": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "ViT-l-16-21k": "vit_large_patch16_224.augreg_in21k_ft_in1k",
    "ViT-b-32-21k": "vit_base_patch32_224.augreg_in21k_ft_in1k",
    "ViT-l-32-21k": "vit_large_patch32_384.orig_in21k_ft_in1k",

    "BiTM-resnetv2-50x1": "resnetv2_50x1_bit.goog_in21k_ft_in1k",
    "BiTM-resnetv2-50x3": "resnetv2_50x3_bit.goog_in21k_ft_in1k",
    "BiTM-resnetv2-101x1": "resnetv2_101x1_bit.goog_in21k_ft_in1k",
    "BiTM-resnetv2-152x2": "resnetv2_152x2_bit.goog_in21k_ft_in1k",

    "ConvNext-T-21k": "convnext_tiny.fb_in22k_ft_in1k",
    "ConvNext-S-21k": "convnext_small.fb_in22k_ft_in1k",
    "ConvNext-B-21k": "convnext_base.fb_in22k_ft_in1k",
    "ConvNext-L-21k": "convnext_large.fb_in22k_ft_in1k",
    
    "ConvNextV2-N": "convnextv2_nano.fcmae_ft_in1k",
    "ConvNextV2-T": "convnextv2_tiny.fcmae_ft_in1k",
    "ConvNextV2-B": "convnextv2_base.fcmae_ft_in1k",
    "ConvNextV2-L": "convnextv2_large.fcmae_ft_in1k",
    
    "ConvNextV2-N-21k": "convnextv2_nano.fcmae_ft_in22k_in1k",
    "ConvNextV2-T-21k": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "ConvNextV2-B-21k": "convnextv2_base.fcmae_ft_in22k_in1k",
    "ConvNextV2-L-21k": "convnextv2_large.fcmae_ft_in22k_in1k",

    "EfficientNet-v2-S": "tf_efficientnetv2_s.in1k",
    "EfficientNet-v2-M": "tf_efficientnetv2_m.in1k",
    "EfficientNet-v2-L": "tf_efficientnetv2_l.in1k",
    
    "EfficientNet-v2-S-21k": "tf_efficientnetv2_s.in21k_ft_in1k",
    "EfficientNet-v2-M-21k": "tf_efficientnetv2_m.in21k_ft_in1k",
    "EfficientNet-v2-L-21k": "tf_efficientnetv2_l.in21k_ft_in1k",

    "InceptionV4": "inception_v4.tf_in1k",

    "NS-EfficientNet-B0": "tf_efficientnet_b0.ns_jft_in1k",
    "NS-EfficientNet-B1": "tf_efficientnet_b1.ns_jft_in1k",
    "NS-EfficientNet-B2": "tf_efficientnet_b2.ns_jft_in1k",
    "NS-EfficientNet-B3": "tf_efficientnet_b3.ns_jft_in1k",
    "NS-EfficientNet-B4": "tf_efficientnet_b4.ns_jft_in1k",
    "NS-EfficientNet-B5": "tf_efficientnet_b5.ns_jft_in1k",
    "NS-EfficientNet-B6": "tf_efficientnet_b6.ns_jft_in1k",
    "NS-EfficientNet-B7": "tf_efficientnet_b7.ns_jft_in1k",

    "DeiT-t": "deit_tiny_patch16_224.fb_in1k",
    "DeiT-s": "deit_small_patch16_224.fb_in1k",
    "DeiT-b": "deit_base_patch16_224.fb_in1k",

    "DeiT3-s": "deit3_small_patch16_224.fb_in1k",
    "DeiT3-m": "deit3_medium_patch16_224.fb_in1k",
    "DeiT3-b": "deit3_base_patch16_224.fb_in1k",
    "DeiT3-l": "deit3_large_patch16_224.fb_in1k",

    "DeiT3-s-21k": "deit3_small_patch16_224.fb_in22k_ft_in1k",
    "DeiT3-m-21k": "deit3_medium_patch16_224.fb_in22k_ft_in1k",
    "DeiT3-b-21k": "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "DeiT3-l-21k": "deit3_large_patch16_224.fb_in22k_ft_in1k",

    "MaxViT-b": "maxvit_base_tf_224.in1k",
    "MaxViT-l": "maxvit_large_tf_224.in1k",

    "CrossViT-9dagger": "crossvit_9_dagger_240.in1k",
    "CrossViT-15dagger": "crossvit_15_dagger_240.in1k",
    "CrossViT-18dagger": "crossvit_18_dagger_240.in1k",

    "FastViT-sa12": "fastvit_sa12.apple_in1k",
    "FastViT-sa24": "fastvit_sa24.apple_in1k",
    "FastViT-sa36": "fastvit_sa36.apple_in1k",

    "XCiT-s24-16": "xcit_small_24_p16_224.fb_in1k",
    "XCiT-m24-16": "xcit_medium_24_p16_224.fb_in1k",
    "XCiT-l24-16": "xcit_large_24_p16_224.fb_in1k",

    "LeViT-128": "levit_128.fb_dist_in1k",
    "LeViT-256": "levit_256.fb_dist_in1k",
    "LeViT-384": "levit_384.fb_dist_in1k",

    "MViTv2-t": "mvitv2_tiny.fb_in1k",
    "MViTv2-s": "mvitv2_small.fb_in1k",
    "MViTv2-b": "mvitv2_base.fb_in1k",
    "MViTv2-l": "mvitv2_large.fb_in1k",

    "BeiT-b": "beit_base_patch16_224.in22k_ft_in22k_in1k",
    
    "ConViT-t": "convit_tiny.fb_in1k",
    "ConViT-s": "convit_small.fb_in1k",
    "ConViT-b": "convit_base.fb_in1k",

    "CaiT-xxs24": "cait_xxs24_384.fb_dist_in1k",
    "CaiT-xs24": "cait_xs24_384.fb_dist_in1k",
    "CaiT-s24": "cait_s24_384.fb_dist_in1k",

    "EVA02-t-21k": "eva02_tiny_patch14_336.mim_in22k_ft_in1k",
    "EVA02-s-21k": "eva02_small_patch14_336.mim_in22k_ft_in1k",
    "EVA02-b-21k": "eva02_base_patch14_448.mim_in22k_ft_in1k",
    
    "Inception-ResNetv2": "inception_resnet_v2.tf_in1k",
    
    "SwinV2-t-W16": "swinv2_tiny_window16_256.ms_in1k",
    "SwinV2-s-Win16": "swinv2_small_window16_256.ms_in1k",
    "SwinV2-b-Win16": "swinv2_base_window16_256.ms_in1k",
    "SwinV2-b-Win12to16-21k": "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    "SwinV2-l-Win12to16-21k": "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",

    "InceptionNext-t": "inception_next_tiny.sail_in1k",
    "InceptionNext-s": "inception_next_small.sail_in1k",
    "InceptionNext-b": "inception_next_base.sail_in1k",
    
    "Xception": "legacy_xception.tf_in1k",
    
    "NasNet-l": "nasnetalarge.tf_in1k",
    
    "PiT-t": "pit_ti_224.in1k",
    "PiT-xs": "pit_xs_224.in1k",
    "PiT-s": "pit_s_224.in1k",
    "PiT-b": "pit_b_224.in1k",
    
    "EfficientFormer-l1": "efficientformer_l1.snap_dist_in1k",
    "EfficientFormer-l3": "efficientformer_l3.snap_dist_in1k",
    "EfficientFormer-l7": "efficientformer_l7.snap_dist_in1k",
    
    "MobileNetV3-l-21k": "mobilenetv3_large_100.miil_in21k_ft_in1k",
    
    "DaViT-t": "davit_tiny.msft_in1k",
    "DaViT-s": "davit_small.msft_in1k",
    "DaViT-b": "davit_base.msft_in1k",
    
    "CoaT-t-lite": "coat_lite_tiny.in1k",
    "CoaT-mi-lite": "coat_lite_mini.in1k",
    "CoaT-s-lite": "coat_lite_small.in1k",
    "CoaT-me-lite": "coat_lite_medium.in1k",

    "ResNet50-a1": "resnet50.a1_in1k",
    "ResNet50-ig1B": "resnet50.fb_swsl_ig1b_ft_in1k",
    "ResNet50-yfcc100m": "resnet50.fb_ssl_yfcc100m_ft_in1k",
    "ResNet18-A1": "resnet18.a1_in1k",
    "ResNext50_32x4d_A1": "resnext50_32x4d.a1_in1k",
    "ResNet34-A1": "resnet34.a1_in1k",
    "ResNet101-A1": "resnet101.a1_in1k",
    "ResNet152-A1": "resnet152.a1_in1k",
    "BeiTV2-b": "beitv2_base_patch16_224.in1k_ft_in1k",

    "vit-t-16-21k": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",

    "ResNeXt101-32x8d-IG1B": "resnext101_32x8d.fb_swsl_ig1b_ft_in1k",
    "ResNeXt50-32x4d-YFCCM100": "resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k",
    "ResNeXt50-32x4d-IG1B": "resnext50_32x4d.fb_swsl_ig1b_ft_in1k",
    "ResNet18-IG1B": "resnet18.fb_swsl_ig1b_ft_in1k",

    'SeNet154': "senet154.gluon_in1k",
    'ResNet50d': "resnet50d.gluon_in1k",
    'RegNet-y-4gf': "regnety_040.pycls_in1k",

    "CLIP-B32-V-OpenAI": "vit_base_patch32_clip_224.openai_ft_in1k",
    "CLIP-B32-V-Laion2B": "vit_base_patch32_clip_224.laion2b_ft_in1k",
    "CLIP-B16-V-OpenAI": "vit_base_patch16_clip_224.openai_ft_in1k",
    "CLIP-B16-V-Laion2B": "vit_base_patch16_clip_224.laion2b_ft_in1k",

}
SALMAN_TRANSFORM = tv_transforms.Compose([
            tv_transforms.Resize(256),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
        ])
LIU_SINGH_TRANSFORM = tv_transforms.Compose([tv_transforms.Resize(256, tv_transforms.InterpolationMode.BICUBIC), tv_transforms.CenterCrop(224), tv_transforms.ToTensor()])
ROB_BENCH_MODELS = {
    "Salman2020Do-RN50": ("Salman2020Do_R50", SALMAN_TRANSFORM),
    "Salman2020Do-RN50-2": ("Salman2020Do_50_2", SALMAN_TRANSFORM),
    
    "Liu2023Comprehensive-Swin-L": ("Liu2023Comprehensive_Swin-L", LIU_SINGH_TRANSFORM),
    "Liu2023Comprehensive-ConvNeXt-L": ("Liu2023Comprehensive_ConvNeXt-L", LIU_SINGH_TRANSFORM),
    "Liu2023Comprehensive-Swin-B": ("Liu2023Comprehensive_Swin-B", LIU_SINGH_TRANSFORM),
    "Liu2023Comprehensive-ConvNeXt-B": ("Liu2023Comprehensive_ConvNeXt-B", LIU_SINGH_TRANSFORM),
    
    "Singh2023Revisiting-ConvNeXt-L-ConvStem": ("Singh2023Revisiting_ConvNeXt-L-ConvStem", LIU_SINGH_TRANSFORM),
    "Singh2023Revisiting-ConvNeXt-B-ConvStem": ("Singh2023Revisiting_ConvNeXt-B-ConvStem", LIU_SINGH_TRANSFORM),
    "Singh2023Revisiting-ViT-B-ConvStem": ("Singh2023Revisiting_ViT-B-ConvStem", LIU_SINGH_TRANSFORM),
    "Singh2023Revisiting-ViT-S-ConvStem": ("Singh2023Revisiting_ViT-S-ConvStem", LIU_SINGH_TRANSFORM),
    "Singh2023Revisiting-ConvNeXt-S-ConvStem": ("Singh2023Revisiting_ConvNeXt-S-ConvStem", LIU_SINGH_TRANSFORM),
    "Singh2023Revisiting-ConvNeXt-T-ConvStem": ("Singh2023Revisiting_ConvNeXt-T-ConvStem", LIU_SINGH_TRANSFORM),
}

DINOV2_TRANSFORM = tv_transforms.Compose([
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

TORCH_HUB_MODELS = {
    "ViT-b-14-dinoV2-LP": (lambda: torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc'), DINOV2_TRANSFORM),
    "ViT-s-14-dinoV2-LP": (lambda: torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc'), DINOV2_TRANSFORM),
    "ViT-l-14-dinoV2-LP": (lambda: torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc'), DINOV2_TRANSFORM),
    "ViT-b-14-dinov2-reg-LP": (lambda: torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc'), DINOV2_TRANSFORM),
    "ViT-s-14-dinov2-reg-LP": (lambda: torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc'), DINOV2_TRANSFORM),
    "ViT-l-14-dinov2-reg-LP": (lambda: torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc'), DINOV2_TRANSFORM)
}

A123_MODELS = {
    'EfficientNet-b0-A1': lambda: download_and_create_model("tf_efficientnet_b0.in1k", "tf_efficientnet_b0_a1_0-9188dd46.pth", "tf_efficientnet_b0_a1.pth"),
    'EfficientNet-b1-A1': lambda: download_and_create_model("tf_efficientnet_b1.in1k", "tf_efficientnet_b1_a1_0-b55e845c.pth", "tf_efficientnet_b1_a1.pth"),
    'EfficientNet-b2-A1': lambda: download_and_create_model("tf_efficientnet_b2.in1k", "tf_efficientnet_b2_a1_0-f1382665.pth", "tf_efficientnet_b2_a1.pth"),
    'EfficientNet-b3-A1': lambda: download_and_create_model("tf_efficientnet_b3.in1k", "tf_efficientnet_b3_a1_0-efc81b92.pth", "tf_efficientnet_b3_a1.pth"),
    'EfficientNet-b4-A1': lambda: download_and_create_model("tf_efficientnet_b4.in1k", "f_efficientnet_b4_a1_0-182bef54.pth", "tf_efficientnet_b4_a1.pth"),
    'EfficientNetv2-M-A1': lambda: download_and_create_model("efficientnetv2_rw_m.agc_in1k", "efficientnetv2_rw_m_a1_0-b788290c.pth", "efficientnetv2_rw_m_a1.pth"),
    'EfficientNetv2-S-A1': lambda: download_and_create_model("efficientnetv2_rw_s.ra2_in1k", "efficientnetv2_rw_s_a1_0-59d76611.pth", "efficientnetv2_rw_s_a1.pth"),
    'RegNety-040-A1': lambda: download_and_create_model("regnety_040.pycls_in1k", "regnety_040_a1_0-453380cb.pth", "regnety_040_a1.pth"),
    'RegNety-080-A1': lambda: download_and_create_model("regnety_080.pycls_in1k", "regnety_080_a1_0-7d647454.pth", "regnety_080_a1.pth"),
    'RegNety-160-A1': lambda: download_and_create_model("regnety_160.pycls_in1k", "regnety_160_a1_0-ed74711e.pth", "regnety_160_a1.pth"),
    'RegNety-320-A1': lambda: download_and_create_model("regnety_320.pycls_in1k", "regnety_320_a1_0-6c920aed.pth", "regnety_320_a1.pth"),
    'ResNet18-A1': lambda: download_and_create_model("resnet18", "resnet18_a1_0-d63eafa0.pth", "resnet18_a1.pth"),
    'ResNet34-A1': lambda: download_and_create_model("resnet34", "resnet34_a1_0-46f8f793.pth", "resnet34_a1.pth"),
    'ResNet50-A1': lambda: download_and_create_model("resnet50", "resnet50_a1_0-14fe96d1.pth", "resnet50_a1.pth"),
    'ResNet50d-A1': lambda: download_and_create_model("resnet50d.gluon_in1k", "resnet50d_a1_0-e20cff14.pth", "resnet50d_a1.pth"),
    'ResNet101-A1': lambda: download_and_create_model("resnet101", "resnet101_a1_0-cdcb52a9.pth", "resnet101_a1.pth"),
    'ResNet152-A1': lambda: download_and_create_model("resnet152", "resnet152_a1_0-2eee8a7a.pth", "resnet152_a1.pth"),
    'ResNext50-32x4d-A1': lambda: download_and_create_model("resnext50_32x4d.gluon_in1k", "resnext50_32x4d_a1_0-b5a91a1d.pth", "resnext50_32x4d_a1.pth"),
    'SeNet154-A1': lambda: download_and_create_model("senet154.gluon_in1k", "gluon_senet154_a1_0-ef9d383e.pth", "gluon_senet154_a1.pth"),
    
    'EfficientNet-b0-A3': lambda: download_and_create_model("tf_efficientnet_b0.in1k", "tf_efficientnet_b0_a3_0-94e799dc.pth", "tf_efficientnet_b0_a3.pth"),
    'EfficientNet-b1-A3': lambda: download_and_create_model("tf_efficientnet_b1.in1k", "tf_efficientnet_b1_a3_0-ee9f9669.pth", "tf_efficientnet_b1_a3.pth"),
    'EfficientNet-b2-A3': lambda: download_and_create_model("tf_efficientnet_b2.in1k", "tf_efficientnet_b2_a3_0-61f0f688.pth", "tf_efficientnet_b2_a3.pth"),
    'EfficientNet-b3-A3': lambda: download_and_create_model("tf_efficientnet_b3.in1k", "tf_efficientnet_b3_a3_0-0a50fa9a.pth", "tf_efficientnet_b3_a3.pth"),
    'EfficientNet-b4-A3': lambda: download_and_create_model("tf_efficientnet_b4.in1k", "tf_efficientnet_b4_a3_0-a6a8179a.pth", "tf_efficientnet_b4_a3.pth"),
    'EfficientNetv2-M-A3': lambda: download_and_create_model("efficientnetv2_rw_m.agc_in1k", "efficientnetv2_rw_m_a3_0-68b15d26.pth", "efficientnetv2_rw_m_a3.pth"),
    'EfficientNetv2-S-A3': lambda: download_and_create_model("efficientnetv2_rw_s.ra2_in1k", "efficientnetv2_rw_s_a3_0-11105c48.pth", "efficientnetv2_rw_s_a3.pth"),
    'RegNety-040-A3': lambda: download_and_create_model("regnety_040.pycls_in1k", "regnety_040_a3_0-9705a0d6.pth", "regnety_040_a3.pth"),
    'RegNety-080-A3': lambda: download_and_create_model("regnety_080.pycls_in1k", "regnety_080_a3_0-2fb073a0.pth", "regnety_080_a3.pth"),
    'RegNety-160-A3': lambda: download_and_create_model("regnety_160.pycls_in1k", "regnety_160_a3_0-9ee45d21.pth", "regnety_160_a3.pth"),
    'RegNety-320-A3': lambda: download_and_create_model("regnety_320.pycls_in1k", "regnety_320_a3_0-242d2987.pth", "regnety_320_a3.pth"),
    'ResNet18-A3': lambda: download_and_create_model("resnet18", "resnet18_a3_0-40c531c8.pth", "resnet18_a3.pth"),
    'ResNet34-A3': lambda: download_and_create_model("resnet34", "resnet34_a3_0-a20cabb6.pth", "resnet34_a3.pth"),
    'ResNet50-A3': lambda: download_and_create_model("resnet50", "resnet50_a3_0-59cae1ef.pth", "resnet50_a3.pth"),
    'ResNet50d-A3': lambda: download_and_create_model("resnet50d.gluon_in1k", "resnet50d_a3_0-403fdfad.pth", "resnet50d_a3.pth"),
    'ResNet101-A3': lambda: download_and_create_model("resnet101", "resnet101_a3_0-1db14157.pth", "resnet101_a3.pth"),
    'ResNet152-A3': lambda: download_and_create_model("resnet152", "resnet152_a3_0-134d4688.pth", "resnet152_a3.pth"),
    'ResNext50-32x4d-A3': lambda: download_and_create_model("resnext50_32x4d.gluon_in1k", "resnext50_32x4d_a3_0-3e450271.pth", "resnext50_32x4d_a3.pth"),
    'SeNet154-A3': lambda: download_and_create_model("senet154.gluon_in1k", "gluon_senet154_a3_0-d8df0fde.pth", "gluon_senet154_a3.pth"),
    
    'EfficientNet-b0-A2': lambda: download_and_create_model("tf_efficientnet_b0.in1k", "tf_efficientnet_b0_a2_0-48bede62.pth", "tf_efficientnet_b0_a2.pth"),
    'EfficientNet-b1-A2': lambda: download_and_create_model("tf_efficientnet_b1.in1k", "tf_efficientnet_b1_a2_0-d342a7bf.pth", "tf_efficientnet_b1_a2.pth"),
    'EfficientNet-b2-A2': lambda: download_and_create_model("tf_efficientnet_b2.in1k", "tf_efficientnet_b2_a2_0-ae4f4996.pth", "tf_efficientnet_b2_a2.pth"),
    'EfficientNet-b3-A2': lambda: download_and_create_model("tf_efficientnet_b3.in1k", "tf_efficientnet_b3_a2_0-e183dbec.pth", "tf_efficientnet_b3_a2.pth"),
    'EfficientNet-b4-A2': lambda: download_and_create_model("tf_efficientnet_b4.in1k", "tf_efficientnet_b4_a2_0-bc5f172e.pth", "tf_efficientnet_b4_a2.pth"),
    'EfficientNetv2-M-A2': lambda: download_and_create_model("efficientnetv2_rw_m.agc_in1k", "efficientnetv2_rw_m_a2_0-12297cd3.pth", "efficientnetv2_rw_m_a2.pth"),
    'EfficientNetv2-S-A2': lambda: download_and_create_model("efficientnetv2_rw_s.ra2_in1k", "efficientnetv2_rw_s_a2_0-cafb8f99.pth", "efficientnetv2_rw_s_a2.pth"),
    'RegNety-040-A2': lambda: download_and_create_model("regnety_040.pycls_in1k", "regnety_040_a2_0-acda0189.pth", "regnety_040_a2.pth"),
    'RegNety-080-A2': lambda: download_and_create_model("regnety_080.pycls_in1k", "regnety_080_a2_0-2298ae4e.pth", "regnety_080_a2.pth"),
    'RegNety-160-A2': lambda: download_and_create_model("regnety_160.pycls_in1k", "regnety_160_a2_0-6631355e.pth", "regnety_160_a2.pth"),
    'RegNety-320-A2': lambda: download_and_create_model("regnety_320.pycls_in1k", "regnety_320_a2_0-a9fedcbf.pth", "regnety_320_a2.pth"),
    'ResNet18-A2': lambda: download_and_create_model("resnet18", "resnet18_a2_0-b61bd467.pth", "resnet18_a2.pth"),
    'ResNet34-A2': lambda: download_and_create_model("resnet34", "resnet34_a2_0-82d47d71.pth", "resnet34_a2.pth"),
    'ResNet50-A2': lambda: download_and_create_model("resnet50", "resnet50_a2_0-a2746f79.pth", "resnet50_a2.pth"),
    'ResNet50d-A2': lambda: download_and_create_model("resnet50d.gluon_in1k", "resnet50d_a2_0-a3adc64d.pth", "resnet50d_a2.pth"),
    'ResNet101-A2': lambda: download_and_create_model("resnet101", "resnet101_a2_0-6edb36c7.pth", "resnet101_a2.pth"),
    'ResNet152-A2': lambda: download_and_create_model("resnet152", "resnet152_a2_0-b4c6978f.pth", "resnet152_a2.pth"),
    'ResNext50-32x4d-A2': lambda: download_and_create_model("resnext50_32x4d.gluon_in1k", "resnext50_32x4d_a2_0-efc76add.pth", "resnext50_32x4d_a2.pth"),
    'SeNet154-A2': lambda: download_and_create_model("senet154.gluon_in1k", "gluon_senet154_a2_0-63cb3b08.pth", "gluon_senet154_a2.pth"),

}


MODEL_MAP = {
    #SETUPS
    "SELF_SL_ALL": _SELF_SL_ALL,
    "SELF_SL_LP": _SELF_SL_LP,
    "SELF_SL_FT": _SELF_SL_FT,
    "AT": _AT,
    "SL": _SL,
    "SEMISL": _SEMISL,
    "A1": _A1,
    "A2": _A2,
    "A3": _A3,

    #Dataset
    "IN1k": _IN1K,
    "IN21k": _IN21K,
    "BD": _BD,

    #Architecture
    "ViL": _VIL,
    "Bcos": _BCOS,
    "CNN": _CNN,
    "TRA": _TRA,

    #ALL
    "ALL": _ALL

}

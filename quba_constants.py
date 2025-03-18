#Paths
_DATA_DIR = "DUMMY"
_IMAGENET1K_DIR = _DATA_DIR + "imagenet/val/" 
_IMAGENETC_DIR = _DATA_DIR + "imagenet-c/"
_MODEL_VS_HUMAN_DIR = _DATA_DIR + "modelvshuman"
_IMAGENETR_DIR = _DATA_DIR + "imagenet-r"
_MIXED_RAND_DIR = _DATA_DIR + "bg_challenge/mixed_rand/val" 
_MIXED_SAME_DIR = _DATA_DIR + "bg_challenge/mixed_same/val"
# CUDA_VISIBLE_DEVICES=3 python eval.py --model CNN --accuracy --adv_rob --c_rob --ood_rob --object_focus --calibration_error --fairness --shape_bias --size --batch_size 16 --device cuda:0
# CUDA_VISIBLE_DEVICES=1 python eval.py --model CNN --accuracy --adv_rob --c_rob --ood_rob --object_focus --calibration_error --fairness --shape_bias --size --batch_size 32 --device cuda:0 
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
_PROJ_DIR = "DUMMY" #Path to helper directory
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
        'mobileclip-s0', 'mobileclip-s1', 'mobileclip-s2', 'mobileclip-b', 'EfficientNet-b0-A1', 'EfficientNet-b1-A1', 'EfficientNet-b2-A1', 'EfficientNet-b3-A1', 'EfficientNet-b4-A1', 'EfficientNetv2-M-A1', 'EfficientNetv2-S-A1', 'RegNety-040-A1', 'RegNety-080-A1', 'RegNety-160-A1', 'RegNety-320-A1', 'ResNet101-A1', 'ResNet152-A1', 'ResNet18-A1', 'ResNet34-A1', 'ResNet50-A1', 'ResNet50d-A1', 'ResNext50-32x4d-A1', 'SeNet154-A1', 'EfficientNet-b0-A2', 'EfficientNet-b1-A2', 'EfficientNet-b2-A2', 'EfficientNet-b3-A2', 'EfficientNet-b4-A2', 'EfficientNetv2-M-A2', 'EfficientNetv2-S-A2', 'RegNety-040-A2', 'RegNety-080-A2', 'RegNety-160-A2', 'RegNety-320-A2', 'ResNet101-A2', 'ResNet152-A2', 'ResNet18-A2', 'ResNet34-A2', 'ResNet50-A2', 'ResNet50d-A2', 'ResNext50-32x4d-A2', 'SeNet154-A2', 'EfficientNet-b0-A3', 'EfficientNet-b1-A3', 'EfficientNet-b2-A3', 'EfficientNet-b3-A3', 'EfficientNet-b4-A3', 'EfficientNetv2-M-A3', 'EfficientNetv2-S-A3', 'RegNety-040-A3', 'RegNety-080-A3', 'RegNety-160-A3', 'RegNety-320-A3', 'ResNet101-A3', 'ResNet152-A3', 'ResNet18-A3', 'ResNet34-A3', 'ResNet50-A3', 'ResNet50d-A3', 'ResNext50-32x4d-A3', 'SeNet154-A3', 
        'bcos-convnext-base', 'bcos-convnext-tiny', 'bcos-DenseNet121', 'bcos-DenseNet161', 'bcos-DenseNet169', 'bcos-DenseNet201', 'bcos-ResNet152', 'bcos-ResNet18', 'bcos-ResNet34', 'bcos-ResNet50', 'bcos-simple-vit-b-patch16-224', 'RegNet-y-4gf', 'mobileclip-blt', 'ViT-s-16-DINO-LP', 'siglip-l-16', 'bcos-ResNet101', 'metaclip-b16', 'convnext-large-d-clip', 'metaclip-l14', 'Singh2023Revisiting-ViT-S-ConvStem', 'convnext-base-w-320-clip', 'convnext-large-d-320-clip', 'Hiera-S-LP', 'Hiera-T-LP',
        "ViT-l-14-dinoV2-FT", "ViT-b-14-dinoV2-FT", "ViT-s-14-dinoV2-FT", "ViT-l-14-dinoV2-FT-Reg", "ViT-b-14-dinoV2-FT-Reg", "ViT-s-14-dinoV2-FT-Reg", "CLIP-B16-V-OpenAI", "CLIP-B16-V-Laion2B", "CLIP-B32-V-OpenAI", "CLIP-B32-V-Laion2B", "ViT-l-14-dinov2-reg-LP", "ViT-b-14-dinov2-reg-LP", "ViT-s-14-dinov2-reg-LP",
        "CLIP-B16-DataCompXL", "CLIP-B16-Laion2B", "CLIP-B16-CommonPool-XL-DFN2B", 
        "CLIP-L14-OpenAI", "CLIP-L14-DataCompXL", "CLIP-L14-Laion2B", "CLIP-L14-CommonPool-XL-DFN2B",
        "ViT-B-16-SigLIP2", "ViT-L-16-SigLIP2-256"
    ]

MODEL_CONFIGS = {
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

import torch
from helper.generate_data import load_dataset, load_model
from quba_constants import MODEL_MAP, _IMAGENET1K_DIR
from helper import benchmarks as bench
import argparse
from scipy.stats.mstats import gmean
from data_utils import print_result_to_excel, calculate_rank_correlation_with_p, compute_quba_score, compute_normalized_values, combine_dict
import random
import numpy as np
import data_utils

NOT_AVAILABLE_MODELS = ['ViTB-DINO-FT', 'ResNet50-DINO-FT', 'Hiera-B-LP', 'Hiera-S-LP', 'Hiera-T-LP', 'vit-b-16-mae-lp', 
                        "ViT-s-14-dinoV2-FT", "ViT-b-14-dinoV2-FT", "ViT-l-14-dinoV2-FT", "ViT-s-14-dinoV2-FT-Reg", "ViT-b-14-dinoV2-FT-Reg", "ViT-l-14-dinoV2-FT-Reg"]

parser = argparse.ArgumentParser(description="Evaluation script for: Beyond Accuracy: What Matters in Designing Well-Behaved Models?")

parser.add_argument("--model",
                    choices=[
                        'AlexNet', 'GoogLeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'VGG11-bn', 
                        'VGG13-bn', 'VGG16-bn', 'VGG19-bn', 'ResNet18', 'ResNet34', 'ResNet50', 
                        'ResNet101', 'ResNet152', 'WRN-50-2', 'WRN-101-2', 'SqueezeNet', 'InceptionV3', 
                        'InceptionV4', 'Inception-ResNetv2', 'ResNeXt50-32x4d', 'ResNeXt101-32x8d', 
                        'ResNeXt101-64x4d', 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201', 
                        'Xception', 'MobileNetV2', 'ShuffleNet-v2-05', 'ShuffleNet-v2-1', 'ShuffleNet-v2-15', 
                        'ShuffleNet-v2-2', 'NasNet-l', 'MobileNetV3-s', 'MobileNetV3-l', 'MobileNetV3-l-21k', 
                        'BagNet9', 'BagNet17', 'BagNet33', 'MnasNet-05', 'MnasNet-075', 'MnasNet-1', 'MnasNet-13', 
                        'EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2', 'EfficientNet-B3', 'EfficientNet-B4', 
                        'EfficientNet-B5', 'EfficientNet-B6', 'EfficientNet-B7', 'NS-EfficientNet-B0', 
                        'NS-EfficientNet-B1', 'NS-EfficientNet-B2', 'NS-EfficientNet-B3', 'NS-EfficientNet-B4', 
                        'NS-EfficientNet-B5', 'NS-EfficientNet-B6', 'NS-EfficientNet-B7', 'Salman2020Do-RN50-2', 
                        'Salman2020Do-RN50', 'BiTM-resnetv2-50x1', 'BiTM-resnetv2-50x3', 'BiTM-resnetv2-101x1', 
                        'BiTM-resnetv2-152x2', 'RegNet-y-400mf', 'RegNet-y-800mf', 'RegNet-y-1-6gf', 'RegNet-y-3-2gf', 
                        'RegNet-y-8gf', 'RegNet-y-16gf', 'RegNet-y-32gf', 'VIT-b-16', 'VIT-l-16', 'VIT-b-32', 'VIT-l-32', 
                        'Swin-T', 'Swin-S', 'Swin-B', 'EfficientNet-v2-S', 'EfficientNet-v2-S-21k', 'EfficientNet-v2-M', 
                        'EfficientNet-v2-M-21k', 'EfficientNet-v2-L', 'EfficientNet-v2-L-21k', 'DeiT-t', 'DeiT-s', 'DeiT-b', 
                        'ConViT-t', 'ConViT-s', 'ConViT-b', 'CaiT-xxs24', 'CaiT-xs24', 'CaiT-s24', 'CrossViT-9dagger', 
                        'CrossViT-15dagger', 'CrossViT-18dagger', 'XCiT-s24-16', 'XCiT-m24-16', 'XCiT-l24-16', 'LeViT-128', 
                        'LeViT-256', 'LeViT-384', 'PiT-t', 'PiT-xs', 'PiT-s', 'PiT-b', 'CoaT-t-lite', 'CoaT-mi-lite', 
                        'CoaT-s-lite', 'CoaT-me-lite', 'MaxViT-t', 'MaxViT-b', 'MaxViT-l', 'DeiT3-s', 'DeiT3-s-21k', 
                        'DeiT3-m', 'DeiT3-m-21k', 'DeiT3-b', 'DeiT3-b-21k', 'DeiT3-l', 'DeiT3-l-21k', 'MViTv2-t', 
                        'MViTv2-s', 'MViTv2-b', 'MViTv2-l', 'SwinV2-T-Win8', 'SwinV2-S-WIn8', 'SwinV2-B-Win8', 
                        'SwinV2-t-W16', 'SwinV2-s-Win16', 'SwinV2-b-Win16', 'SwinV2-b-Win12to16-21k', 
                        'SwinV2-l-Win12to16-21k', 'ViT-t5-16', 'ViT-t5-16-21k', 'ViT-t11-16', 'ViT-t11-16-21k', 
                        'ViT-t21-16', 'ViT-t21-16-21k', 'ViT-s-16', 'ViT-s-16-21k', 'ViT-b-16-21k', 'ViT-b-32-21k', 
                        'ViT-l-16-21k', 'ViT-l-32-21k', 'ConvNext-T', 'ConvNext-T-21k', 'ConvNext-S', 'ConvNext-S-21k', 
                        'ConvNext-B', 'ConvNext-B-21k', 'ConvNext-L', 'ConvNext-L-21k', 'BeiT-b', 'EfficientFormer-l1', 
                        'EfficientFormer-l3', 'EfficientFormer-l7', 'DaViT-t', 'DaViT-s', 'DaViT-b', 
                        'Liu2023Comprehensive-Swin-B', 'Liu2023Comprehensive-Swin-L', 'Liu2023Comprehensive-ConvNeXt-B', 
                        'Liu2023Comprehensive-ConvNeXt-L', 'Singh2023Revisiting-ConvNeXt-T-ConvStem', 
                        'Singh2023Revisiting-ConvNeXt-S-ConvStem', 'Singh2023Revisiting-ConvNeXt-B-ConvStem', 
                        'Singh2023Revisiting-ConvNeXt-L-ConvStem', 'Singh2023Revisiting-ViT-B-ConvStem', 'ConvNextV2-N', 
                        'ConvNextV2-N-21k', 'ConvNextV2-T', 'ConvNextV2-T-21k', 'ConvNextV2-B', 'ConvNextV2-B-21k', 
                        'ConvNextV2-L', 'ConvNextV2-L-21k', 'Hiera-T', 'Hiera-S', 'Hiera-B', 'Hiera-B-Plus', 'Hiera-L', 
                        'EVA02-t-21k', 'EVA02-s-21k', 'EVA02-b-21k', 'InceptionNext-t', 'InceptionNext-s', 
                        'InceptionNext-b', 'FastViT-sa12', 'FastViT-sa24', 'FastViT-sa36', 'BeiTV2-b', 'SeNet154', 
                        'ResNet50d', 'ResNeXt50-32x4d-YFCCM100', 'ResNet50-yfcc100m', 'ResNet50-ig1B', 
                        'ResNeXt101-32x8d-IG1B', 'ResNeXt50-32x4d-IG1B', 'ResNet18-IG1B', 'vit-b-16-mae-ft', 
                        'vit-b-16-mae-lp', 'Hiera-B-LP', 'ViT-b-16-DINO-LP', 'ViTB-DINO-FT', 'ResNet50-DINO-LP', 
                        'ResNet50-DINO-FT', 'ViT-l-14-dinoV2-LP', 'ViT-b-14-dinoV2', 'ViT-s-14-dinoV2-LP', 
                        'vit-t-16-21k', 'siglip-b-16', 'clip-resnet50', 'clip-vit-b-16', 'clip-resnet101', 
                        'clip-vit-b-32', 'mobileclip-s0', 'mobileclip-s1', 'mobileclip-s2', 'mobileclip-b', 
                        'EfficientNet-b0-A1', 'EfficientNet-b1-A1', 'EfficientNet-b2-A1', 'EfficientNet-b3-A1', 
                        'EfficientNet-b4-A1', 'EfficientNetv2-M-A1', 'EfficientNetv2-S-A1', 'RegNety-040-A1', 
                        'RegNety-080-A1', 'RegNety-160-A1', 'RegNety-320-A1', 'ResNet101-A1', 'ResNet152-A1', 
                        'ResNet18-A1', 'ResNet34-A1', 'ResNet50-A1', 'ResNet50d-A1', 'ResNext50-32x4d-A1', 'SeNet154-A1',
                        'EfficientNet-b0-A2', 'EfficientNet-b1-A2', 'EfficientNet-b2-A2', 'EfficientNet-b3-A2', 
                        'EfficientNet-b4-A2', 'EfficientNetv2-M-A2', 'EfficientNetv2-S-A2', 'RegNety-040-A2', 
                        'RegNety-080-A2', 'RegNety-160-A2', 'RegNety-320-A2', 'ResNet101-A2', 'ResNet152-A2', 
                        'ResNet18-A2', 'ResNet34-A2', 'ResNet50-A2', 'ResNet50d-A2', 'ResNext50-32x4d-A2', 'SeNet154-A2', 
                        'EfficientNet-b0-A3', 'EfficientNet-b1-A3', 'EfficientNet-b2-A3', 'EfficientNet-b3-A3', 
                        'EfficientNet-b4-A3', 'EfficientNetv2-M-A3', 'EfficientNetv2-S-A3', 'RegNety-040-A3', 
                        'RegNety-080-A3', 'RegNety-160-A3', 'RegNety-320-A3', 'ResNet101-A3', 'ResNet152-A3', 
                        'ResNet18-A3', 'ResNet34-A3', 'ResNet50-A3', 'ResNet50d-A3', 'ResNext50-32x4d-A3', 'SeNet154-A3', 
                        'bcos-convnext-base', 'bcos-convnext-tiny', 'bcos-DenseNet121', 'bcos-DenseNet161', 
                        'bcos-DenseNet169', 'bcos-DenseNet201', 'bcos-ResNet152', 'bcos-ResNet18', 'bcos-ResNet34', 
                        'bcos-ResNet50', 'bcos-simple-vit-b-patch16-224', 'RegNet-y-4gf', 'mobileclip-blt', 
                        'ViT-s-16-DINO-LP', 'siglip-l-16', 'bcos-ResNet101', 'metaclip-b16', 'convnext-large-d-clip', 
                        'metaclip-l14', 'Singh2023Revisiting-ViT-S-ConvStem', 'convnext-base-w-320-clip', 
                        'convnext-large-d-320-clip', 'Hiera-S-LP', 'Hiera-T-LP',
                        "ViT-l-14-dinoV2-FT", "ViT-b-14-dinoV2-FT", "ViT-s-14-dinoV2-FT", "ViT-l-14-dinoV2-FT-Reg", "ViT-b-14-dinoV2-FT-Reg", "ViT-s-14-dinoV2-FT-Reg", "CLIP-B16-V-OpenAI", "CLIP-B16-V-Laion2B", "CLIP-B32-V-OpenAI", "CLIP-B32-V-Laion2B", "ViT-l-14-dinov2-reg-LP", "ViT-b-14-dinov2-reg-LP", "ViT-s-14-dinov2-reg-LP",
                        "CLIP-B16-DataCompXL", "CLIP-B16-Laion2B", "CLIP-B16-CommonPool-XL-DFN2B", 
                        "CLIP-L14-OpenAI", "CLIP-L14-DataCompXL", "CLIP-L14-Laion2B", "CLIP-L14-CommonPool-XL-DFN2B",
                        "ViT-B-16-SigLIP2", "ViT-L-16-SigLIP2-256"

                        "SELF_SL_ALL", "SELF_SL_LP", "SELF_SL_FT", "AT", "SL", "SEMISL",
                        "A1", "A2", "A3", "IN1k", "IN21k", "BD", "ViL",
                        "Bcos", "CNN", "TRA", "ALL"

                    ],
                    default="ALL",
                    help='Specify a model architecture or a group you want to test, all models will be loaded if you dont.')

#General
parser.add_argument("--device", default="cuda:0",
                    help="Choose your device.")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch Size for Dataset")
parser.add_argument("--file", default="results.xlsx", 
                    help="file for printing results")
parser.add_argument("--num_workers", default=10)
parser.add_argument("--shuffle", default=True)
parser.add_argument('--num_classes', default=1000)
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--compute_corr', default=False, action="store_true", help='computes the rank correlation matrix if true')

#Quality Dimensions
parser.add_argument("--accuracy", default=False, action="store_true",
                    help="compute accuracy")
parser.add_argument("--adv_rob", default=False, action="store_true",
                    help="compute adversarial robustness by geometric mean of pgd and fgsm")
parser.add_argument("--c_rob", default=False, action="store_true",
                    help="compute relative Top-1 Accuracy on imagenet-c dataset")
parser.add_argument("--ood_rob", default=False, action="store_true",
                    help="compute out-of-distribution robustness by gmean of relative top-1 accuracies on various domain shift datasets")
parser.add_argument("--object_focus", default=False, action="store_true",
                    help="compute the object focus by measuring 1-BG Gap")
parser.add_argument("--calibration_error", default=False, action="store_true",
                    help="compute calibration error by gmean of ace and ece")
parser.add_argument("--fairness", default=False, action="store_true",
                    help="compute fairness measured by gmean of std of mean class accuracies and mean class confidences")
parser.add_argument("--ood_detection", default=False, action="store_true",
                    help="compute ood detection")
parser.add_argument("--shape_bias", default=False, action="store_true",
                    help="compute shape bias on modelvshuman dataset")
parser.add_argument("--params", default=False, action="store_true",
                    help="compute number of parameters")

def benchmark(model, dataloader, args):
    """
    Tests different properties of a given model
    :param model: The Model you want to test
    :param dataloader: The Dataset you want to test the model on
    :return: A dictionary of results in the form (Property, Result)
    """
    results = {"Model": [model.model_name]}
    usecols = []
    model.eval() 

    if args.accuracy:
        with torch.no_grad():
            results["Acc"] = [bench.test_accuracy(model, dataloader, args)]
            usecols.append("Acc")
            print("Accuracy Done", results["Acc"])

    if args.adv_rob:
        pgd = bench.test_adversarial_robustness(model=model, dataloader=dataloader, device=args.device, attack="PGD")
        fgsm = bench.test_adversarial_robustness(model=model, dataloader=dataloader, device=args.device, attack="FGSM")
        results["Adv. Rob."] = [gmean([pgd, fgsm])]
        usecols.append("Adv. Rob.")
        print("Adversarial Robustness Done", results["Adv. Rob."])

    if args.c_rob:
        with torch.no_grad():
            results["C-Rob."] = [bench.test_c_robustness(model, dataloader, args)]
            usecols.append("C-Rob.")
            print("C-Robustness Done", results["C-Rob."])

    if args.ood_rob:
        with torch.no_grad():
            results["OOD Rob."] = [bench.test_ood_robustness(model, dataloader, args)]
            usecols.append("OOD Rob.")
            print("OOD Robustness Done", results["OOD Rob."])

    if args.calibration_error:
        with torch.no_grad():
            results["Cal. Err."] = [bench.test_calibration_error(model, dataloader, args)]
            usecols.append("Cal. Err.")
            print("Calibration Error Done", results["Cal. Err."])
    
    if args.fairness:
        with torch.no_grad():
            fair_pred = bench.test_fairness(model, dataloader, args.device, args.num_classes)
            fair_conf = bench.test_fair_confidence(model, dataloader, args.device)
            results["Fairness"] = [gmean([fair_conf, fair_pred])]
            usecols.append("Fairness")
            print("Fairness done", results["Fairness"])
    
    if args.object_focus:
        with torch.no_grad():
            results["Obj. Foc."] = [bench.test_object_focus(model, dataloader,args.batch_size, args)]
            usecols.append("Obj. Foc.")
            print("Object Focus Done", results["Obj. Foc."])

    if args.shape_bias:
        with torch.no_grad():
            results["Shape Bias"] = [bench.test_shape_bias(model=model, args=args)]
            usecols.append("Shape Bias")
            print("Shape Bias Done", results["Shape Bias"])

    if args.params:
        results["Params"] = [bench.test_size(model)]
        usecols.append("Params")
        print("Params Done", results["Params"])
            
    return results, usecols

def main():

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
 
    models = None
    if args.model:
        if args.model in MODEL_MAP:
            models = MODEL_MAP[args.model]
        else:
            models = [args.model]
    else:
        raise ValueError("Please choose a valid model")

    num_workers=int(args.num_workers)
    empty_sheet = args.model 
    print("Sheet Name:", empty_sheet)
    results = {}
    total_models = len(models)
    models_evaluated = 0
    usecols = []
    print(total_models, "model to evaluate")
    
    # CUDA_VISIBLE_DEVICES=6 python eval.py --model CNN --accuracy --adv_rob --c_rob --ood_rob --object_focus --calibration_error --fairness --shape_bias --size --batch_size 4 --device cuda:0 
    # CUDA_VISIBLE_DEVICES=7 python eval.py --model CNN --size --batch_size 4 --device cuda:0 
    
    ### REMOVE WHEN CODE GOES PUBLIC #########
    models = [model for model in models if model not in NOT_AVAILABLE_MODELS]
    ##########################################

    for model_arch in models: 
    
        #Load Model
        model = load_model(model_arch, args) 
        model.to(args.device)
        
        #Load dataset
        loader = load_dataset(model = model, path=_IMAGENET1K_DIR, batch_size=args.batch_size, num_workers=num_workers) 
        
        #Start benchmark
        print("Start Benchmarks for Model:", model.model_name)
        result, usecols = benchmark(model=model, dataloader=loader, args=args)
        results = combine_dict(results, result)
        print(results)
        print_result_to_excel(args.file, empty_sheet, results)
        
        models_evaluated += 1
        print(f"{models_evaluated}/{total_models} Models evaluated")

    print_result_to_excel(args.file, empty_sheet, results)
    if args.compute_corr:
        if models_evaluated > 1 and len(usecols) > 1:
            data_utils.calculate_rank_correlation_with_p(args.file, empty_sheet, usecols)
        else:
            print("Not enough evaluated models or quality dimensions for computing correlation. You need at least two of both.")
    if usecols == ["Acc", "Adv. Rob.", "C-Rob.", "OOD Rob.", "Cal. Err.", "Fairness", "Params"]:
        data_utils.compute_normalized_values(args.file, empty_sheet)
        data_utils.compute_quba_score(args.file, "NORM")

    print("Finished")

if __name__ == "__main__":
    main()

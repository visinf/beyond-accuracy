import torch
import numpy as np
import torchvision.datasets as dset
from helper import generate_data as gd
import torchattacks
from torchmetrics.classification import MulticlassCalibrationError, MulticlassConfusionMatrix
from helper.modelvshuman.helper.plotting_helper import read_data
import os
import pandas as pd
import json
from helper.modelvshuman.datasets import sketch as load_sketch
from helper.modelvshuman.datasets import stylized as load_stylized
from helper.modelvshuman.datasets import corr_sketch as load_corr_sketch
from helper.modelvshuman.datasets import corr_stylized as load_corr_stylized
from helper.modelvshuman.datasets import texture_shape
import helper.uncertainty as uncertainty
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats.mstats import gmean
import quba_constants as con


def test_accuracy(model, dataloader, args):
    """
    Computes the top-1 accuracy on a dataset specified by the dataloader

    :model: model to test 
    :dataloader: dataset you want to use
    :return: Top-1 Accuracy
    """

    correct = 0
    total = 0
    print("Computing Top-1 Accuracy...")
    for inputs, targets in tqdm(dataloader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        outputs = model(inputs) 
        outputs = outputs.to(args.device)

        correct += torch.eq(outputs.argmax(1), targets).sum().item()
        total += targets.size(0)

    acc = correct / total
    return acc

def test_adversarial_robustness(model, dataloader, attack, device):
    """
    Test the adverserial robustness of a given model on a given dataset by using a {attack} attack
    :param model: the model you want to test
    :param dataloader: the dataset you want to test the model on
    :param: attack: the type of attack of attack you want to use. The options are PGD and FGSM
    :device: device for computations
    :return: the accuracy on the dataset corrupted by a chosen attack
    """
    chosen_attack=None
    if attack == "PGD":
        chosen_attack = torchattacks.PGD(model)
    elif attack == "FGSM":
        chosen_attack = torchattacks.FGSM(model, eps=8/255)
    else:
        raise ValueError("Please choose a valid attack")

    correct = 0
    correct_corr = 0
    total = 0
    adv_inputs = None
    print("Computing Top-1 Accuracy after", attack, "attack...")
    for input, target in tqdm(dataloader):

        input = input.to(device)
        target = target.to(device)

        input.requires_grad = True

        adv_inputs = chosen_attack(input, target)
        outputs = model(input)
        correct += torch.eq(outputs.argmax(1), target).sum().item()
        
        adv_inputs = adv_inputs.to(device)
        outputs_corr = model(adv_inputs)
        correct_corr += torch.eq(outputs_corr.argmax(1), target).sum().item()
        
        total += target.size(0)

    acc_corr = correct_corr / total
    acc_normal = correct / total
    return acc_corr/acc_normal

def test_c_robustness(model, imagenet_dataloader, args):
    """
    Tests the Corruption Robustness of the Model on the ImageNet-C Dataset

    :param model: The Model you want to test
    :imagenet_dataloader: iterable ImageNet-1k dataloader
    :return: Mean Corruption Accuracy
    """

    imagenet_acc = test_accuracy(model, imagenet_dataloader, args)

    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    ]
    
    all_accs = []
    results = []
    for distortion in distortions:

        distortion_acc_list = []

        for severity in range(1, 6):
            distorted_dataset = dset.ImageFolder(
                root= con._IMAGENETC_DIR + distortion + "/" + str(severity),
                transform=model.transform)

            distorted_dataset_loader = torch.utils.data.DataLoader(
                distorted_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=int(args.num_workers))
            correct = 0
            total = 0
            print("Computing relative Top-1 Accuracy on Distortion", distortion, "with Severity", severity, "...")
            for (data, target) in tqdm(distorted_dataset_loader):
                data = data.to(args.device)
                target = target.to(args.device)
                
                output = model(data)

                pred = output.data.max(1)[1]

                correct += pred.eq(target).sum()
                total += target.size(0)
            
            acc = (correct / total) / imagenet_acc
            acc = acc.cpu().numpy()
            distortion_acc_list.append(acc)

        distortion_err = np.mean(distortion_acc_list)
        results.append(((distortion, "mCE", distortion_err)))
        all_accs.append(distortion_err)
    
    mCA = np.mean(all_accs)
    results.append((("Total", "mCE", mCA)))

    return mCA 

def test_ood_robustness(model, imagenet_loader, args):
    """
    Tests the ability of model to generalize by calculating the relative top-1 accuracy on ImageNet-R, Stylied-ImageNet,
    ImageNet-Sketch, Edge and Silhouette
    :param model: The model you want to test
    :param model_name: The name of the model
    :return: A list in which the elements are structured like the following (dataset, metric name, metric value)
    """

    gd.download_dataset("edge")
    gd.download_dataset("stylized")
    gd.download_dataset("silhouette")
    gd.download_dataset("sketch")
    gd.download_dataset("imagenet-r")

    imagenet_acc = test_accuracy(model, imagenet_loader, args)

    edge = texture_shape.edge(model=model, args=args)
    silhouette = texture_shape.silhouette(model=model, args=args)
    sketch = load_sketch(model=model, args=args)
    stylized = load_stylized(model=model, args=args)

    dataloader_list = [edge, silhouette, sketch, stylized]
    results = []
    for dataset in dataloader_list:
        print("Computing relative Top-1 Accuracy on", dataset.name, "...")
        categories = dataset.decision_mapping.categories
        correct = 0
        total = 0
        for images, targets, paths in tqdm(dataset.loader):
            images = images.to(args.device)
            targets = torch.tensor([categories.index(targ) for targ in targets])
            targets = targets.to(args.device)

            logits = model(images) 
            softmax_output = F.softmax(logits, dim=1)
            predictions, probs = dataset.decision_mapping(softmax_output.cpu().numpy())

            probs = torch.from_numpy(probs)
            probs = probs.to(args.device)

            correct += torch.eq(probs.argmax(1), targets).sum().item()
            total += targets.size(0)

            """
            if isinstance(target, torch.Tensor):
                batch_targets = model.to_numpy(target)
            else:
                batch_targets = target
            predictions, _ = dataset.decision_mapping(softmax_output.cpu().numpy())

            for metric in dataset.metrics:
                metric.update(predictions, batch_targets)
            """
        
        #for metric in dataset.metrics:
        #    results.append((metric.value/imagenet_acc))
        results.append((correct/total)/imagenet_acc)

    r_acc = _test_r_robustness(model, args)
    r_acc = [x / imagenet_acc for x in r_acc] #r_acc/imagenet_acc

    results = [*results, *r_acc]
    geo_mean = gmean(results)
    return geo_mean

def test_expected_calibration_error(model, dataloader, device, num_classes=1000, norm="l1"):
    """
    Tests the Calibration of a model on a given Dataset by calculating and returning the Expected Calibration Error (ECE)
    specified in "Obtaining Well Calibrated Probabilities Using Bayesian Binning" Naeini et al. 2015

    :param model: The model you want to test
    :param dataloader: The dataset you want to use
    :return: the Expected Calibration Error
    """
    metric = MulticlassCalibrationError(num_classes=num_classes, norm=norm)
    metric = metric.to(device)

    print("Computing Expected Calibration Error...")
    for inputs, targets in tqdm(dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        metric.update(outputs, targets.to(device))
    return metric.compute().cpu().item()

def test_adapative_calibration_error(model, dataloader, device):
    """
    Tests the Calibration of a model on a given Dataset by calculating and returning the Adaptive Calibration Error (ACE)
    specified in "Measuring Calibration in Deep Learning" Nixon et al. 2019

    :param model: The model you want to test
    :param dataloader: The dataset you want to use
    :return: the Adaptive Calibration Error
    """

    ACE = uncertainty._GeneralCalibrationErrorMetric(binning_scheme='adaptive', threshold=0, class_conditional=True, max_prob=False, norm='l1', num_bins=15)

    labels = torch.tensor([], dtype=torch.long, device=device)
    predictions = torch.tensor([], device=device)
    print("Computing Adaptive Calibration Error...")
    for inputs, targets in tqdm(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs) 
        outputs = outputs.softmax(dim=1)
        outputs = outputs.detach()
        targets = targets.detach()

        predictions = torch.cat((predictions, outputs), dim=0)
        labels = torch.cat((labels, targets), dim=0)
    
    ACE.update_state(labels.cpu(), predictions.cpu())

    return ACE.result().item()

def test_calibration_error(model, dataloader, args):
    ace = test_adapative_calibration_error(model, dataloader, args.device)
    ece = test_expected_calibration_error(model, dataloader, args.device, num_classes=args.num_classes)
    geo_mean = gmean([ace, ece])
    return geo_mean

def _test_r_robustness(model, args):
    """
    Tests the Rendition Robustness of the Model with the ImageNet-R Dataset (https://arxiv.org/abs/2006.16241)

    :param model: The Model you want to test
    :param device: The device on which the calculations are performed
    :return: Accuracy of the model on the ImageNet-R Dataset

    """

    all_wnids = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457', 'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244', 'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675', 'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143', 'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672', 'n01860187', 'n01871265', 'n01872401', 'n01873310', 'n01877812', 'n01882714', 'n01883070', 'n01910747', 'n01914609', 'n01917289', 'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01945685', 'n01950731', 'n01955084', 'n01968897', 'n01978287', 'n01978455', 'n01980166', 'n01981276', 'n01983481', 'n01984695', 'n01985128', 'n01986214', 'n01990800', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849', 'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 'n02056570', 'n02058221', 'n02066245', 'n02071294', 'n02074367', 'n02077923', 'n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110627', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02114367', 'n02114548', 'n02114712', 'n02114855', 'n02115641', 'n02115913', 'n02116738', 'n02117135', 'n02119022', 'n02119789', 'n02120079', 'n02120505', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604', 'n02130308', 'n02132136', 'n02133161', 'n02134084', 'n02134418', 'n02137549', 'n02138441', 'n02165105', 'n02165456', 'n02167151', 'n02168699', 'n02169497', 'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338', 'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853', 'n02276258', 'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n02317335', 'n02319095', 'n02321529', 'n02325366', 'n02326432', 'n02328150', 'n02342885', 'n02346627', 'n02356798', 'n02361337', 'n02363005', 'n02364673', 'n02389026', 'n02391049', 'n02395406', 'n02396427', 'n02397096', 'n02398521', 'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 'n02437616', 'n02441942', 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 'n02454379', 'n02457408', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 'n02484975', 'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166', 'n02490219', 'n02492035', 'n02492660', 'n02493509', 'n02493793', 'n02494079', 'n02497673', 'n02500267', 'n02504013', 'n02504458', 'n02509815', 'n02510455', 'n02514041', 'n02526121', 'n02536864', 'n02606052', 'n02607072', 'n02640242', 'n02641379', 'n02643566', 'n02655020', 'n02666196', 'n02667093', 'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373', 'n02692877', 'n02699494', 'n02701002', 'n02704792', 'n02708093', 'n02727426', 'n02730930', 'n02747177', 'n02749479', 'n02769748', 'n02776631', 'n02777292', 'n02782093', 'n02783161', 'n02786058', 'n02787622', 'n02788148', 'n02790996', 'n02791124', 'n02791270', 'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071', 'n02802426', 'n02804414', 'n02804610', 'n02807133', 'n02808304', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02817516', 'n02823428', 'n02823750', 'n02825657', 'n02834397', 'n02835271', 'n02837789', 'n02840245', 'n02841315', 'n02843684', 'n02859443', 'n02860847', 'n02865351', 'n02869837', 'n02870880', 'n02871525', 'n02877765', 'n02879718', 'n02883205', 'n02892201', 'n02892767', 'n02894605', 'n02895154', 'n02906734', 'n02909870', 'n02910353', 'n02916936', 'n02917067', 'n02927161', 'n02930766', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02951585', 'n02963159', 'n02965783', 'n02966193', 'n02966687', 'n02971356', 'n02974003', 'n02977058', 'n02978881', 'n02979186', 'n02980441', 'n02981792', 'n02988304', 'n02992211', 'n02992529', 'n02999410', 'n03000134', 'n03000247', 'n03000684', 'n03014705', 'n03016953', 'n03017168', 'n03018349', 'n03026506', 'n03028079', 'n03032252', 'n03041632', 'n03042490', 'n03045698', 'n03047690', 'n03062245', 'n03063599', 'n03063689', 'n03065424', 'n03075370', 'n03085013', 'n03089624', 'n03095699', 'n03100240', 'n03109150', 'n03110669', 'n03124043', 'n03124170', 'n03125729', 'n03126707', 'n03127747', 'n03127925', 'n03131574', 'n03133878', 'n03134739', 'n03141823', 'n03146219', 'n03160309', 'n03179701', 'n03180011', 'n03187595', 'n03188531', 'n03196217', 'n03197337', 'n03201208', 'n03207743', 'n03207941', 'n03208938', 'n03216828', 'n03218198', 'n03220513', 'n03223299', 'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280', 'n03271574', 'n03272010', 'n03272562', 'n03290653', 'n03291819', 'n03297495', 'n03314780', 'n03325584', 'n03337140', 'n03344393', 'n03345487', 'n03347037', 'n03355925', 'n03372029', 'n03376595', 'n03379051', 'n03384352', 'n03388043', 'n03388183', 'n03388549', 'n03393912', 'n03394916', 'n03400231', 'n03404251', 'n03417042', 'n03424325', 'n03425413', 'n03443371', 'n03444034', 'n03445777', 'n03445924', 'n03447447', 'n03447721', 'n03450230', 'n03452741', 'n03457902', 'n03459775', 'n03461385', 'n03467068', 'n03476684', 'n03476991', 'n03478589', 'n03481172', 'n03482405', 'n03483316', 'n03485407', 'n03485794', 'n03492542', 'n03494278', 'n03495258', 'n03496892', 'n03498962', 'n03527444', 'n03529860', 'n03530642', 'n03532672', 'n03534580', 'n03535780', 'n03538406', 'n03544143', 'n03584254', 'n03584829', 'n03590841', 'n03594734', 'n03594945', 'n03595614', 'n03598930', 'n03599486', 'n03602883', 'n03617480', 'n03623198', 'n03627232', 'n03630383', 'n03633091', 'n03637318', 'n03642806', 'n03649909', 'n03657121', 'n03658185', 'n03661043', 'n03662601', 'n03666591', 'n03670208', 'n03673027', 'n03676483', 'n03680355', 'n03690938', 'n03691459', 'n03692522', 'n03697007', 'n03706229', 'n03709823', 'n03710193', 'n03710637', 'n03710721', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03729826', 'n03733131', 'n03733281', 'n03733805', 'n03742115', 'n03743016', 'n03759954', 'n03761084', 'n03763968', 'n03764736', 'n03769881', 'n03770439', 'n03770679', 'n03773504', 'n03775071', 'n03775546', 'n03776460', 'n03777568', 'n03777754', 'n03781244', 'n03782006', 'n03785016', 'n03786901', 'n03787032', 'n03788195', 'n03788365', 'n03791053', 'n03792782', 'n03792972', 'n03793489', 'n03794056', 'n03796401', 'n03803284', 'n03804744', 'n03814639', 'n03814906', 'n03825788', 'n03832673', 'n03837869', 'n03838899', 'n03840681', 'n03841143', 'n03843555', 'n03854065', 'n03857828', 'n03866082', 'n03868242', 'n03868863', 'n03871628', 'n03873416', 'n03874293', 'n03874599', 'n03876231', 'n03877472', 'n03877845', 'n03884397', 'n03887697', 'n03888257', 'n03888605', 'n03891251', 'n03891332', 'n03895866', 'n03899768', 'n03902125', 'n03903868', 'n03908618', 'n03908714', 'n03916031', 'n03920288', 'n03924679', 'n03929660', 'n03929855', 'n03930313', 'n03930630', 'n03933933', 'n03935335', 'n03937543', 'n03938244', 'n03942813', 'n03944341', 'n03947888', 'n03950228', 'n03954731', 'n03956157', 'n03958227', 'n03961711', 'n03967562', 'n03970156', 'n03976467', 'n03976657', 'n03977966', 'n03980874', 'n03982430', 'n03983396', 'n03991062', 'n03992509', 'n03995372', 'n03998194', 'n04004767', 'n04005630', 'n04008634', 'n04009552', 'n04019541', 'n04023962', 'n04026417', 'n04033901', 'n04033995', 'n04037443', 'n04039381', 'n04040759', 'n04041544', 'n04044716', 'n04049303', 'n04065272', 'n04067472', 'n04069434', 'n04070727', 'n04074963', 'n04081281', 'n04086273', 'n04090263', 'n04099969', 'n04111531', 'n04116512', 'n04118538', 'n04118776', 'n04120489', 'n04125021', 'n04127249', 'n04131690', 'n04133789', 'n04136333', 'n04141076', 'n04141327', 'n04141975', 'n04146614', 'n04147183', 'n04149813', 'n04152593', 'n04153751', 'n04154565', 'n04162706', 'n04179913', 'n04192698', 'n04200800', 'n04201297', 'n04204238', 'n04204347', 'n04208210', 'n04209133', 'n04209239', 'n04228054', 'n04229816', 'n04235860', 'n04238763', 'n04239074', 'n04243546', 'n04251144', 'n04252077', 'n04252225', 'n04254120', 'n04254680', 'n04254777', 'n04258138', 'n04259630', 'n04263257', 'n04264628', 'n04265275', 'n04266014', 'n04270147', 'n04273569', 'n04275548', 'n04277352', 'n04285008', 'n04286575', 'n04296562', 'n04310018', 'n04311004', 'n04311174', 'n04317175', 'n04325704', 'n04326547', 'n04328186', 'n04330267', 'n04332243', 'n04335435', 'n04336792', 'n04344873', 'n04346328', 'n04347754', 'n04350905', 'n04355338', 'n04355933', 'n04356056', 'n04357314', 'n04366367', 'n04367480', 'n04370456', 'n04371430', 'n04371774', 'n04372370', 'n04376876', 'n04380533', 'n04389033', 'n04392985', 'n04398044', 'n04399382', 'n04404412', 'n04409515', 'n04417672', 'n04418357', 'n04423845', 'n04428191', 'n04429376', 'n04435653', 'n04442312', 'n04443257', 'n04447861', 'n04456115', 'n04458633', 'n04461696', 'n04462240', 'n04465501', 'n04467665', 'n04476259', 'n04479046', 'n04482393', 'n04483307', 'n04485082', 'n04486054', 'n04487081', 'n04487394', 'n04493381', 'n04501370', 'n04505470', 'n04507155', 'n04509417', 'n04515003', 'n04517823', 'n04522168', 'n04523525', 'n04525038', 'n04525305', 'n04532106', 'n04532670', 'n04536866', 'n04540053', 'n04542943', 'n04548280', 'n04548362', 'n04550184', 'n04552348', 'n04553703', 'n04554684', 'n04557648', 'n04560804', 'n04562935', 'n04579145', 'n04579432', 'n04584207', 'n04589890', 'n04590129', 'n04591157', 'n04591713', 'n04592741', 'n04596742', 'n04597913', 'n04599235', 'n04604644', 'n04606251', 'n04612504', 'n04613696', 'n06359193', 'n06596364', 'n06785654', 'n06794110', 'n06874185', 'n07248320', 'n07565083', 'n07579787', 'n07583066', 'n07584110', 'n07590611', 'n07613480', 'n07614500', 'n07615774', 'n07684084', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07711569', 'n07714571', 'n07714990', 'n07715103', 'n07716358', 'n07716906', 'n07717410', 'n07717556', 'n07718472', 'n07718747', 'n07720875', 'n07730033', 'n07734744', 'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592', 'n07754684', 'n07760859', 'n07768694', 'n07802026', 'n07831146', 'n07836838', 'n07860988', 'n07871810', 'n07873807', 'n07875152', 'n07880968', 'n07892512', 'n07920052', 'n07930864', 'n07932039', 'n09193705', 'n09229709', 'n09246464', 'n09256479', 'n09288635', 'n09332890', 'n09399592', 'n09421951', 'n09428293', 'n09468604', 'n09472597', 'n09835506', 'n10148035', 'n10565667', 'n11879895', 'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546', 'n12768682', 'n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560', 'n13133613', 'n15075141']
    imagenet_r_wnids = {'n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677'}
    imagenet_r_mask = [wnid in imagenet_r_wnids for wnid in all_wnids]
    
    imagenet_r = dset.ImageFolder(root=con._IMAGENETR_DIR, transform=model.transform)
    imagenet_r_loader = torch.utils.data.DataLoader(imagenet_r, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=True, num_workers=int(args.num_workers))

    correct = 0
    total = 0
    print("Computing Top-1 Accuracy on ImageNet-R...")
    for inputs, targets in tqdm(imagenet_r_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        outputs = model(inputs)[:,imagenet_r_mask] 
        outputs = outputs.to(outputs) 

        pred = outputs.argmax(1)
        correct += pred.eq(targets.data).sum().item()

        total += targets.size(0)

    acc = correct /total
    return [acc]

def test_fair_accuracies(model, loader, device, num_classes=1000):
    """
    Tests the class accuracy balance of the specified model on a given dataset

    :param model: the model to test
    :param device: the device to compute it on
    :return: The standard deviation of the class accuracies
    """

    fairness_eval = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    total = 0
    print("Computing Prediction Balance...")
    for input, target in tqdm(loader):
        input = input.to(device)
        target = target.to(device)

        preds = model(input)
        preds = preds.to(device)
        preds = (torch.argmax(preds, dim=1)).to(device)
        
        fairness_eval.update(preds, target)

        total += target.size(0)

    val = fairness_eval.compute()

    class_acc = torch.diagonal(val) / val.sum(1)

    std = torch.std(class_acc) 
    return 1 - std.cpu().item()

def test_fair_confidence(model, loader, device):
    """
    Tests the class confidence balance of the specified model on a given dataset

    :param model: the model to test
    :param loader: the dataloader
    :param device: the device
    :return: The standard deviation of the class accuracies
    """

    results = {}
    print("Computing Confidence Balance...")
    for input, target in tqdm(loader):
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        output = F.softmax(output, dim=1)

        target_np = target.cpu().numpy()
        output_np = output.cpu().numpy()
        
        for t, o in zip(target_np, output_np):
            results[str(t)] = results.get(str(t), []) + [o[t]]

    for key in results.keys():
        results[key] = np.std(results[key])
    
    return 1 - sum(results.values()) / len(results)  

def _object_focus(loader, model, args, map_to_in9, map_in_to_in9=True):
    """
    :loader: dataloader for iterating the dataset
    :model: model to evaluate
    :map_in_to_in9: whether or not to map model outputs from ImageNet class labels to ImageNet9 class labels
    :returns: The average top1 accuracy across the epoch.
    """
    correct = 0
    for inp, target in tqdm(loader):

        inp = inp.to(args.device)
        output = model(inp)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.cpu().detach()[:, 0]
        if map_in_to_in9:
            if map_to_in9 is None:
                raise ValueError('Need to pass in mapping from IN to IN9')
            pred_list = list(pred.numpy())
            pred = torch.LongTensor([map_to_in9[str(x)] for x in pred_list])
        correct += (pred==target).sum().item()
    
    return correct/len(loader.dataset)

def test_object_focus(model, dataloader, batch_size, args):
    """
    Tests the object focus of a given model 

    :param model: the model you want to test
    :dataloader: dataloader of the imagenet dataset
    :return: the object focus metric
    """

    map_to_in9 = {}
    PATH_TO_MAP = "./helper/backgrounds_challenge/in_to_in9.json"  
    with open(PATH_TO_MAP, 'r') as f:
        map_to_in9.update(json.load(f))

    gd.download_dataset("bg_challenge")
    path_rand = con._MIXED_RAND_DIR 
    path_same = con._MIXED_SAME_DIR 

    val_loader_rand = gd.load_dataset(model=model, path=path_rand, batch_size=batch_size)
    val_loader_same = gd.load_dataset(model=model, path=path_same, batch_size=batch_size)

    model.eval()

    imagenet_acc = test_accuracy(model, dataloader, args)
    print(model.model_name)
    acc_ms = _object_focus(val_loader_same, model, args=args, map_to_in9=map_to_in9, map_in_to_in9=True)
    acc_mr = _object_focus(val_loader_rand, model, args=args, map_to_in9=map_to_in9, map_in_to_in9=True)

    bg_gap = (acc_ms - acc_mr) /imagenet_acc

    return 1 - bg_gap

def test_shape_bias(model, args,):
    """
    Tests the Shape Bias of a given Model on the cue-conflict Dataset
    :param model: the model you want to test
    :param args: arguments including device
    :param dataloader: the dataset loader you want to use
    :return: The computed shape bias value
    """
    results = []
    gd.download_dataset("cue-conflict")
    cue_conflict = texture_shape.cue_conflict(model=model, args=args)

    print("Computing Shape Bias...")
    for images, target, paths in tqdm(cue_conflict.loader):
        images = images.to(args.device)
        logits = model(images)
        softmax_output = F.softmax(logits, dim=1)
        
        if isinstance(target, torch.Tensor):
            batch_targets = target.cpu().numpy()
        else:
            batch_targets = target
        
        predictions, _ = cue_conflict.decision_mapping(softmax_output.cpu().numpy())
        
        for pred, targets, path in zip(predictions, batch_targets, paths):
            results.append({
                "object_response": pred[0],
                "category": targets,
                "imagename": path.split('/')[-1]
            })

    return _compute_shape_bias(results)["shape-bias"]

def _get_texture_category(imagename):
    """Return texture category from imagename.

    e.g. 'XXX_dog10-bird2.png' -> 'bird
    '"""
    assert type(imagename) is str

    # remove unneccessary words
    a = imagename.split("_")[-1]
    # remove .png etc.
    b = a.split(".")[0]
    # get texture category (last word)
    c = b.split("-")[-1]
    # remove number, e.g. 'bird2' -> 'bird'
    d = ''.join([i for i in c if not i.isdigit()])
    return d

def _compute_shape_bias(results):
    """Compute shape bias from the results."""
    df = []
    for result in results:
        texture = _get_texture_category(result['imagename'])
        df.append({
            "object_response": result['object_response'],
            "correct_shape": result['category'],
            "correct_texture": texture
        })

    # Filter out rows where shape = texture
    df2 = [row for row in df if row['correct_shape'] != row['correct_texture']]

    correct_shape = sum(1 for row in df2 if row['object_response'] == row['correct_shape'])
    correct_texture = sum(1 for row in df2 if row['object_response'] == row['correct_texture'])

    total = len(df)
    fraction_correct_shape = correct_shape / total
    fraction_correct_texture = correct_texture / total

    shape_bias = fraction_correct_shape / (fraction_correct_shape + fraction_correct_texture)

    return {
        "fraction-correct-shape": fraction_correct_shape,
        "fraction-correct-texture": fraction_correct_texture,
        "shape-bias": shape_bias
    }

def test_size(model):
    """
    Returns the total amount of paramaters in Millions of a given model

    :param model: the model to be examined
    :return: total parameters and depth
    """
    total_params = sum(param.numel() for param in (model.parameters()))
    return round(total_params / 1000000, 1)

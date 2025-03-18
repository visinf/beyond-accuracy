import logging
from os.path import join


logger = logging.getLogger(__name__)
dataset_base_url = "https://github.com/bethgelab/model-vs-human/releases/download/v0.1/{NAME}.tar.gz"

"""
def load_model(model_name, *args):
    if model_name in zoomodels.__dict__:
        model = eval("pytorch_model_zoo.model_pytorch")(model_name, *args)
        framework = 'pytorch'
    elif model_name in list_models("pytorch"):
        model = eval(f"pytorch_model_zoo.{model_name}")(model_name, *args)
        framework = 'pytorch'
    elif model_name in list_models('tensorflow'):
        model = eval(f"tensorflow_model_zoo.{model_name}")(model_name, *args)
        framework = 'tensorflow'
    else:
        raise NameError(f"Model {model_name} is not supported.")
    return model, framework
"""

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

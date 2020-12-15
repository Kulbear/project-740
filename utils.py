import numpy as np
import torch


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('In total: {} trainable parameters.'.format(params))


def set_seed(setting):
    torch.manual_seed(setting['random_seed'])
    np.random.seed(setting['random_seed'])


def set_cudnn(setting):
    torch.backends.cudnn.deterministic = setting['deterministic']
    torch.backends.cudnn.benchmark = setting['benchmark']


# convert a list of multi-label to a ndarray
def multi2array(l, class_num=1103):
    label_list = []
    for lbl in l:
        lbl_array = np.zeros((class_num,), dtype=int)
        lbl_array[np.array(list(lbl))] = 1
        label_list.append(lbl_array)
    return np.array(label_list)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
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
        return str(self.avg)

    def __repr__(self):
        return str(self.avg)

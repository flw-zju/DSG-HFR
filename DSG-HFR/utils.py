import numpy as np
import torch
import torch.nn.functional as F


def contrastive_loss(x, y, l):
    z = abs(x @ y.permute(1, 0))
    tar = torch.from_numpy(np.equal.outer(l, l)).float().to(x.device)
    loss1 = F.mse_loss(z, tar)
    loss2 = (1 - torch.diag(z)).mean()
    loss = loss1 + 2 * loss2
    return loss


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(lr, step, optimizer, epoch):
    scale = 0.457305051927326
    lr = lr * (scale ** (epoch // step))

    if (epoch != 0) & (epoch % step == 0) & (epoch <= 2):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale
        print('lr: {}'.format(lr))


class AverageMeter(object):
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


def load_weights(model, model_path, device):
    weights = torch.load(model_path, map_location=device)
    pretrained_dict = weights["state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
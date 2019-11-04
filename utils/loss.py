import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def one_hot(index, classes):
    # index is not flattened (pypass ignore) ############
    # size = index.size()[:1] + (classes,) + index.size()[1:]
    # view = index.size()[:1] + (1,) + index.size()[1:]
    #####################################################
    # index is flatten (during ignore) ##################
    size = index.size()[:1] + (classes, )
    view = index.size()[:1] + (1, )
    #####################################################

    # mask = torch.Tensor(size).fill_(0).to(device)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):
    def __init__(self,
                 gamma=0,
                 eps=1e-7,
                 size_average=True,
                 one_hot=True,
                 ignore=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore

    def forward(self, input_data, target):
        '''
        only support ignore at 0
        '''
        B, C, H, W = input_data.size()
        input_data = input_data.permute(0, 2, 3, 1).contiguous().view(
            -1, C)  # B * H * W, C = P, C
        target = target.view(-1)
        if self.ignore is not None:
            valid = (target != self.ignore)
            input_data = input_data[valid]
            target = target[valid]

        # print('before one_hot', np.unique(target))
        if self.one_hot: target = one_hot(target, input_data.size(1))
        # print('after one_hot', np.unique(target))
        import pdb
        # pdb.set_trace()
        # print("check probs * target")
        probs = F.softmax(input_data, dim=1)
        probs = (probs * target).sum(1)

        probs = probs.clamp(self.eps, 1. - self.eps)
        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
            # try:
            #     loss = batch_loss.mean()
            # except Exception as e:
            #     print(e)
        else:
            loss = batch_loss.sum()
        return loss


class SoftCrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss2d, self).__init__()

    def forward(self, inputs, targets):
        loss = 0
        inputs = -F.log_softmax(inputs, dim=1)
        for index in range(inputs.size()[0]):
            loss += F.conv2d(inputs[range(index, index + 1)], targets[range(
                index, index + 1)]) / (targets.size()[2] * targets.size()[3])
        return loss

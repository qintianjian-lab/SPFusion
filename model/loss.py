import torch
import torch.nn as nn


class RegressionLoss(nn.Module):
    def __init__(
            self,
            loss_weight: list,
            reduction: str = 'mean',
    ):
        super().__init__()
        self.losses = [
            [nn.MSELoss(reduction=reduction), weight]
            for weight in loss_weight
        ]

    def forward(self, pred, target):
        loss = torch.zeros(1, device=pred.device)
        unweighted_losses = []
        weighted_losses = []
        for i, (loss_fun, weight) in enumerate(self.losses):
            _loss = loss_fun(pred[:, i], target[:, i])
            loss += weight * _loss
            unweighted_losses.append(_loss)
            weighted_losses.append(weight * _loss)
        return loss, unweighted_losses, weighted_losses


class UncertaintyLoss(nn.Module):
    def __init__(self, out_channel: int, reduction: str = 'mean'):
        super().__init__()
        sigma = torch.randn(out_channel)
        self.sigma = nn.Parameter(sigma)
        self.out_channel = out_channel
        self.mse = [nn.MSELoss(reduction=reduction) for _ in range(out_channel)]

    def forward(self, pred, target):
        loss = torch.zeros(1, device=pred.device)
        unweighted_losses = []
        weighted_losses = []
        for i in range(self.out_channel):
            _unweighted_loss = self.mse[i](pred, target)
            _weighted_loss = _unweighted_loss * (1 / (2 * self.sigma[i] ** 2))
            loss += _weighted_loss
            unweighted_losses.append(_unweighted_loss)
            weighted_losses.append(_weighted_loss)
        loss += torch.log(self.sigma.pow(2).prod())
        return loss, unweighted_losses, weighted_losses


class Loss(nn.Module):
    def __init__(
            self,
            loss_fun_name: str,
            out_channel: int,
            loss_weight: list,
            reduction: str = 'mean'
    ):
        super().__init__()
        assert loss_fun_name in ['regression', 'uncertainty'], '[Error] loss_fun_name must be regression or uncertainty'
        if loss_fun_name == 'regression':
            self.loss = RegressionLoss(loss_weight, reduction)
        elif loss_fun_name == 'uncertainty':
            print('[Info] Using uncertainty loss, loss_weight is ignored')
            self.loss = UncertaintyLoss(out_channel)
        else:
            raise NotImplementedError

    def forward(self, pred, target):
        return self.loss(pred, target)

import numpy as np
import torch
import torch.nn as nn


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    if any(map(lambda x: x is None or (isinstance(x, float) and x != x), [mae, mse, rmse, mape, mspe, rse, corr])):
        print("nan here")
    return mae, mse, rmse, mape, mspe, rse, corr

def nonan_drop(pred, true, nonan=None):
    """Dropping the nan targets and their predictions"""
    if nonan is not None:
        pred = pred[nonan].reshape(-1, 1)
        true = true[nonan].reshape(-1, 1)
    return pred, true


def nontail_drop(pred, true):
    """Dropping the non-tail targets and their predictions"""
    pred = pred[np.abs(true) > 0.4].reshape(-1, 1)
    true = true[np.abs(true) > 0.4].reshape(-1, 1)
    return pred, true


def get_decile(pred, true):
    """Getting the decile corresponding values by the pred result"""
    idx = np.argsort(np.abs(pred), axis=0)[-round(len(pred) / 10):]
    pred = np.take(pred, idx)
    true = np.take(true, idx)
    return pred, true


def tail_MSE(pred, true):
    pred, true = nontail_drop(pred, true)
    return MSE(pred, true)


def tail_CORR(pred, true):
    pred, true = nontail_drop(pred, true)
    return CORR(pred, true)


def decile_CORR(pred, true):
    pred, true = get_decile(pred, true)
    return CORR(pred, true)


def SACC(pred, true):
    return (
        ((np.sign(pred) * np.sign(true)) == 1.).sum() /
        ((np.sign(pred) * np.sign(true)) != 0.).sum())


def tail_SACC(pred, true):
    pred, true = nontail_drop(pred, true)
    return SACC(pred, true)


def decile_SACC(pred, true):
    pred, true = get_decile(pred, true)
    return SACC(pred, true)


def binatix_metric(pred, true, nonan=None):
    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    if nonan is not None:
        nonan = nonan.detach().cpu().numpy()
    else:
        nonan = np.ones(pred.shape).astype(bool)
    pred, true = nonan_drop(pred, true, nonan)

    mse = MSE(pred, true).item()
    tail_mse = tail_MSE(pred, true).item()
    corr = CORR(pred, true).item()
    tail_corr = tail_CORR(pred, true).item()
    decile_corr = decile_CORR(pred, true).item()
    sacc = SACC(pred, true).item()
    tail_sacc = tail_SACC(pred, true).item()
    decile_sacc = decile_SACC(pred, true).item()
    if any(map(lambda x: x is None or (isinstance(x, float) and x != x), [mse, tail_mse, corr, tail_corr, sacc, tail_sacc, decile_corr, decile_sacc])):
        print("nan here")
    return mse, tail_mse, corr, tail_corr, sacc, tail_sacc, decile_corr, decile_sacc


def binatix_metric_features(pred, true, nonan=None):
    trues = true.detach().cpu().numpy()
    preds = pred.detach().cpu().numpy()
    l_mse = []
    l_tail_mse = []
    l_corr = []
    l_tail_corr = []
    l_sacc = []
    l_tail_sacc = []
    l_decile_corr = []
    l_decile_sacc = []
    for i in range(pred.shape[-1]):
        true = trues[..., i]
        pred = preds[..., i]
        nonan_i = nonan[..., i]
        if nonan_i is not None:
            nonan_i = nonan_i.detach().cpu().numpy()
        else:
            nonan_i = np.ones(pred.shape).astype(bool)
        pred, true = nonan_drop(pred, true, nonan_i)

        mse = MSE(pred, true).item()
        tail_mse = tail_MSE(pred, true).item()
        corr = CORR(pred, true).item()
        tail_corr = tail_CORR(pred, true).item()
        decile_corr = decile_CORR(pred, true).item()
        sacc = SACC(pred, true).item()
        tail_sacc = tail_SACC(pred, true).item()
        decile_sacc = decile_SACC(pred, true).item()

        l_mse.append(mse)
        l_tail_mse.append(tail_mse)
        l_corr.append(corr)
        l_tail_corr.append(tail_corr)
        l_sacc.append(sacc)
        l_tail_sacc.append(tail_sacc)
        l_decile_corr.append(decile_corr)
        l_decile_sacc.append(decile_sacc)

    mse = np.array(l_mse)
    tail_mse = np.array(l_tail_mse)
    corr = np.array(l_corr)
    tail_corr = np.array(l_tail_corr)
    sacc = np.array(l_sacc)
    tail_sacc = np.array(l_tail_sacc)
    decile_corr = np.array(l_decile_corr)
    decile_sacc = np.array(l_decile_sacc)

    return mse, tail_mse, corr, tail_corr, sacc, tail_sacc, decile_corr, decile_sacc


class MSELoss_Masked(nn.Module):
    def __init__(self, drop_low=False):
        super(MSELoss_Masked, self).__init__()
        self.drop_low = drop_low
        self.drop_val = 0.3

    def forward(self, input: torch.Tensor, target: torch.Tensor, nonan=None) -> torch.Tensor:
        input, target = nonan_drop(input, target, nonan)
        if self.drop_low:
            mask = torch.zeros_like(target)
            lower = torch.abs(target) < self.drop_val
            mask[lower] = mask[lower].bernoulli_(0.1)
            mask[~lower] = 1
            input = input * mask
            target = target * mask

        return nn.functional.mse_loss(input, target)


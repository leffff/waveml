import torch


def RMSE(y_true, y_pred):
    error = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return error


def MSLE(y_true, y_pred):
    error = torch.mean((torch.log(y_true + 1) - torch.log(y_pred + 1)) ** 2)
    return error


def MSE(y_true, y_pred):
    error = torch.mean((y_true - y_pred) ** 2)
    return error


def SSE(y_true, y_pred):
    error = torch.sum((y_true - y_pred) ** 2)
    return error


def MAE(y_true, y_pred):
    error = torch.mean(torch.abs(y_true - y_pred))
    return error


def SAE(y_true, y_pred):
    error = torch.sum(torch.abs(y_true - y_pred))
    return error


def MAPE(y_true, y_pred):
    error = torch.mean(torch.abs((y_true - y_pred) / y_true))
    return error


def MBE(y_true, y_pred):
    error = torch.sum((y_true - y_pred) / y_true)
    return error

def Accuracy(y_true, y_pred):
    total = y_true.__len__()
    guessed = y_pred[y_pred == y_true].__len__()
    res = torch.tensor([guessed / total], requires_grad=True)
    res.retain_grad()
    return res


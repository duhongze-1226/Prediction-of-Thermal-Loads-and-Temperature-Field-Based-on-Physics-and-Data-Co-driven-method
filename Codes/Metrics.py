import torch
import numpy as np

def ERMS(T_pred,T_true):
    return torch.mean(torch.abs(T_true.cpu()-T_pred.cpu())/torch.max(torch.abs(T_true.cpu())))

def MAE(T_pred,T_true):
    return torch.mean(torch.abs(T_true.cpu()-T_pred.cpu()))

def MSE(T_pred,T_true):
    return torch.mean((T_true.cpu()-T_pred.cpu())**2)

def NMAE(T_pred,T_true):
    maxs,indice=torch.max(torch.abs(T_true.squeeze(1).cpu()).flatten(1,-1),1)
    #print (maxs.unsqueeze(-1).unsqueeze(-1).size())
    return torch.mean(torch.abs(T_true.cpu()-T_pred.cpu())/maxs.unsqueeze(-1).unsqueeze(-1))

def RMSE(T_pred,T_true):
    return torch.sqrt(torch.mean((T_true.cpu()-T_pred.cpu())**2))
import torch
import torch.nn as nn 
from torch import Tensor

# this quantile loss is for Neural network with output size of len(quantile_list)

class QuantileLoss(nn.Module):
    """
    Parameters
    ----------
    1) target : 1d Tensor
    2) input : 1d Tensor, Predicted value.
    3) quantile : float, 0. ~ 1.
        Quantile to be evaluated, e.g., 0.5 for median.
        Quantileloss with quenatile of 0.5 is equal to mean absolute error
    """
    def __init__(self, quantile_list: list) -> None:
        super(QuantileLoss, self).__init__()
        self.quantile_lst = quantile_list

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        
        if input[:, 0].unsqueeze(-1).shape != target.shape:
            print('Check input shape!')
            print(f'target shape: {target.shape}, but input shape: {input[:, 0].shape}')

        # in this case, we have n quantiles and n outputs from neurons.
        loss = []
        for index, quantile in enumerate(self.quantile_lst):
            residual = target - input[:, index].unsqueeze(-1) # target - ith neurons output
            loss.append(torch.maximum(quantile * residual, (quantile - 1) * residual).unsqueeze(-1))
    
        loss = torch.mean(torch.sum(torch.cat(loss, dim=1), dim=1))
        return loss
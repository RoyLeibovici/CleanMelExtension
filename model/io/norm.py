from torch import nn
from torch import Tensor
from typing import *

import torch

# The function is identical to `forgetting_normalization` in
# https://github.com/Audio-WestlakeU/NBSS/blob/main/models/io/norm.py

# I changed the name of the function to `recursive_normalization` to better reflect
# the implementation of the function.
def recursive_normalization(XrMag: Tensor, sliding_window_len: int = 250) -> Tensor:
    alpha = (sliding_window_len - 1) / (sliding_window_len + 1)
    mu = 0
    mu_list = []
    B, F, T = XrMag.shape
    XrMM = XrMag.mean(dim=1, keepdim=True).detach().cpu()  # [B,1,T]
    for t in range(T):
        if t < sliding_window_len:
            alpha_this = min((t - 1) / (t + 1), alpha)
        else:
            alpha_this = alpha
        mu = alpha_this * mu + (1 - alpha_this) * XrMM[..., t]
        mu_list.append(mu)

    XrMM = torch.stack(mu_list, dim=-1).to(XrMag.device)
    return XrMM
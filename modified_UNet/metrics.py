import torch
import torch.nn.functional as F

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100
    max_val = torch.max(target)
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr.item()

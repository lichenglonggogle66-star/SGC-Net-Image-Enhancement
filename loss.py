import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.color import rgb2lab
import numpy as np

class L2Loss(nn.Module):
    def forward(self, x, y):
        return F.mse_loss(x, y)

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return 1 - self.ssim(x, y)
    def ssim(self, x, y):
        C1 = 0.01**2
        C2 = 0.03**2
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        sigma_x = torch.var(x)
        sigma_y = torch.var(y)
        sigma_xy = torch.mean((x-mu_x)*(y-mu_y))
        return ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))

class ColorLoss(nn.Module):
    def forward(self, x, y):
        x = x.permute(0,2,3,1).cpu().detach().numpy()
        y = y.permute(0,2,3,1).cpu().detach().numpy()
        loss = 0
        for i in range(x.shape[0]):
            lab1 = rgb2lab(x[i])
            lab2 = rgb2lab(y[i])
            loss += np.mean(np.abs(lab1 - lab2))
        return torch.tensor(loss/x.shape[0], requires_grad=True)

class JointLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.1):
        super().__init__()
        self.l2 = L2Loss()
        self.ssim = SSIMLoss()
        self.color = ColorLoss()
        self.alpha = alpha
        self.beta = beta
    def forward(self, out, gt):
        l2 = self.l2(out, gt)
        ssim = self.ssim(out, gt)
        color = self.color(out, gt)
        return l2 + self.alpha * ssim + self.beta * color

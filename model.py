
---

## 3. model.py（论文真实模型，1:1 对应）
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CMCModule(nn.Module):
    def __init__(self, dim=512, n_heads=8):
        super().__init__()
        self.proj_v = nn.Linear(dim, dim)
        self.proj_t = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, vis_feat, txt_feat):
        vis_feat = self.proj_v(vis_feat).unsqueeze(1)
        txt_feat = self.proj_t(txt_feat).unsqueeze(1)
        attn_feat, _ = self.attn(vis_feat, txt_feat, txt_feat)
        gate = self.gate(torch.cat([vis_feat, attn_feat], dim=-1))
        calib = gate * attn_feat + (1 - gate) * vis_feat
        out = self.ffn(torch.cat([calib, vis_feat], dim=-1)).squeeze(1)
        return out

class CurveGenerator(nn.Module):
    def __init__(self, in_dim=512, num_params=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_params),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class SGCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_proj = nn.Linear(512, 512)
        self.cmc = CMCModule(dim=512)
        self.curve_head = CurveGenerator(512, 64)

    def apply_curve(self, img, params):
        b, c, h, w = img.shape
        params = params.view(b, 1, 1, -1)
        enhanced = img * params
        return enhanced

    def forward(self, img_feat, txt_feat):
        img_feat = self.clip_proj(img_feat)
        feat = self.cmc(img_feat, txt_feat)
        params = self.curve_head(feat)
        return params

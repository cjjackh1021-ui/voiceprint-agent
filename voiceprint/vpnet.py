"""VPNet: Conv1d frontend, dilated TCN, attentive statistics pooling.

The pieces follow the replication plan: the LAS-style conv frontend, Conv-TasNet
temporal blocks, and additive attention turned into a weighted mean and standard deviation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + 1e-5) + self.beta


class TCNBlock(nn.Module):
    def __init__(self, channels: int, hidden: int, kernel: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, hidden, 1)
        self.norm = GlobalLayerNorm(hidden)
        self.dw = nn.Conv1d(
            hidden, hidden, kernel, padding=dilation * (kernel // 2),
            dilation=dilation, groups=hidden)
        self.conv2 = nn.Conv1d(hidden, channels, 1)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm(self.conv1(x)))
        y = self.act(self.dw(y))
        return x + self.conv2(y)


class AttentiveStatsPool(nn.Module):
    def __init__(self, channels: int, attn: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(channels, attn, 1),
            nn.Tanh(),
            nn.Conv1d(attn, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.softmax(self.score(x), dim=-1)
        mean = torch.sum(weight * x, dim=-1)
        var = torch.sum(weight * (x - mean.unsqueeze(-1)).square(), dim=-1)
        return torch.cat([mean, torch.sqrt(var.clamp_min(1e-6))], dim=-1)


class VPNet(nn.Module):
    def __init__(self, n_mels: int = 80, channels: int = 128, emb: int = 192,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.tcn = nn.Sequential(*[TCNBlock(channels, channels, 3, d) for d in dilations])
        self.pool = AttentiveStatsPool(channels)
        self.proj = nn.Linear(channels * 2, emb)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.tcn(self.front(feats))
        return F.normalize(self.proj(self.pool(x)), dim=-1)


class AAMSoftmax(nn.Module):
    def __init__(self, emb: int, n_spk: int, scale: float = 30.0, margin: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_spk, emb))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embedding, dim=-1), F.normalize(self.weight, dim=-1))
        cosine = cosine.clamp(-1 + 1e-6, 1 - 1e-6)
        target = cosine[torch.arange(labels.size(0), device=labels.device), labels]
        phi = torch.cos(torch.acos(target) + self.margin)
        logits = cosine.clone()
        logits[torch.arange(labels.size(0), device=labels.device), labels] = phi
        return F.cross_entropy(self.scale * logits, labels)

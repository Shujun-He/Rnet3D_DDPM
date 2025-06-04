import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLN(nn.Module):
    def __init__(self, c, s_dim):
        super().__init__()
        self.norm_a = nn.LayerNorm(c, elementwise_affine=False)
        self.norm_s = nn.LayerNorm(s_dim, elementwise_affine=False)
        self.linear_scale = nn.Linear(s_dim, c)
        self.linear_shift = nn.Linear(s_dim, c, bias=False)

    def forward(self, a, s):
        # a: [B, c], s: [B, s_dim]
        a_norm = self.norm_a(a)
        s_norm = self.norm_s(s)
        scale = torch.sigmoid(self.linear_scale(s_norm))
        shift = self.linear_shift(s_norm)
        return scale * a_norm + shift

class ConditionedTransitionBlock(nn.Module):
    def __init__(self, c, s_dim, n=2):
        super().__init__()
        self.adaln = AdaLN(c, s_dim)
        self.linear1 = nn.Linear(c, n * c, bias=False)
        self.linear2 = nn.Linear(c, n * c, bias=False)
        self.output_gate = nn.Linear(s_dim, c)
        self.output_proj = nn.Linear(n * c, c, bias=False)

        # Initialize output_gate bias to -2 as in adaLN-Zero
        nn.init.constant_(self.output_gate.bias, -2.0)

    def forward(self, a, s):
        # a: [B, c], s: [B, s_dim]
        a = self.adaln(a, s)
        b = F.silu(self.linear1(a)) * self.linear2(a)  # SwiGLU
        gate = torch.sigmoid(self.output_gate(s))
        out = gate * self.output_proj(b)
        return out

B, c, s_dim = 32, 128, 64
x = torch.randn(B, c)
s = torch.randn(B, s_dim)

block = ConditionedTransitionBlock(c=c, s_dim=s_dim, n=2)
output = block(x, s)  # [B, c]
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DT import AdaLN

class LocalSelfAttention(nn.Module):
    def __init__(self, embed_dim, pair_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.pair2heads = nn.Sequential(nn.LayerNorm(pair_dim),
                                        nn.Linear(pair_dim, num_heads, bias = False))

        self.distance2heads = nn.Linear(1, num_heads, bias = False)

    def forward(self, x, local_attention_pair_rep, inverse_pair_distance, local_attention_pair_mask):
        B, L, D = x.shape
        H = self.num_heads
        W = self.window_size
        Dh = D // H


        # QKV projection
        qkv = self.qkv(x)  # (B, L, 3*D)
        qkv = qkv.reshape(B, L, 3, H, Dh).permute(2, 0, 3, 1, 4)  # (3, B, H, L, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, L, Dh)

        # Pad keys and values on the length dimension
        k = F.pad(k, (0, 0, W, W), mode='constant', value=0)  # (B, H, L+2W, Dh)
        v = F.pad(v, (0, 0, W, W), mode='constant', value=0)

        # Unfold to get local windows: (B, H, L, Dh, 2W+1)
        k = k.unfold(dimension=2, size=2 * W + 1, step=1)
        v = v.unfold(dimension=2, size=2 * W + 1, step=1)

        # print("k shape:", k.shape)  # Debugging line
        # exit()


        # Compute attention
        attn = torch.einsum('bhlc,bhlck->bhlk', q, k) / (Dh ** 0.5)  # (B, H, L, 2W+1)
        pair_bias = self.pair2heads(local_attention_pair_rep) + self.distance2heads(inverse_pair_distance)  # (B, L, 2W+1, H)
        pair_bias = pair_bias.permute(0, 3, 1, 2)

        attn = attn + pair_bias  # (B, H, L, 2W+1)

        attn = attn.masked_fill(local_attention_pair_mask == 1, float('-1e9'))

        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhlk,bhlck->bhlc', attn, v)  # (B, H, L, Dh)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


if __name__ == "__main__":
    # Parameters
    batch_size = 1
    seq_len = 256
    embed_dim = 128
    num_heads = 4
    window_size = 32
    pair_dim = 32
    res_len = 10

    # Create dummy input
    x = torch.randn(batch_size, seq_len, embed_dim).cuda()
    pair_rep = torch.randn(batch_size, res_len, res_len, pair_dim).cuda()

    all_atom_res_index=np.concatenate([[0]*20,[1]*23,[2]*25,[3]*27,[4]*29,[5]*22,[6]*20,[7]*23,[8]*37,[9]*30])

    # print("all_atom_res_index shape:", all_atom_res_index.shape)
    # exit()
    # Initialize model
    local_attn = LocalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size).cuda()

    # Forward pass
    out = local_attn(x).cuda()

    # Check output shape
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    assert out.shape == x.shape, "Output shape does not match input shape!"
    print("Test passed.")
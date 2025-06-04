import torch
from torch.utils.checkpoint import checkpoint

class SafeLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.norm = torch.nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        # Cast to float32 for numerical stability (even in autocast)
        orig_dtype = x.dtype
        x = x.float()
        x = self.norm(x)
        return x.to(orig_dtype)

class PairStack(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.LayerNorm(size),
            torch.nn.Linear(size, size)
        )

    def forward(self, x):
        # Issue also occurs if we unconditionally checkpoint()
        if torch.is_grad_enabled():
            return checkpoint(self.block, x, use_reentrant=False, debug=False)
        else:
            return self.block(x)


class Mod(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.stack = PairStack(size)
        self.linear = torch.nn.Linear(size, size)

    def forward(self, x):
        #with torch.set_grad_enabled(False):
        with torch.no_grad():
            self.stack.requires_grad_(False)
            x = self.stack(x).detach()
        self.stack.requires_grad_(True)
        x = self.stack(x)
        return self.linear(x)



def main():
    device = torch.device("cuda")
    size = 64

    m = Mod(size=size).to(device)
    x = torch.linspace(0, 1, 2 * 3 * size).reshape(2, 3, size).to(device)

    with torch.autocast(device.type, dtype=torch.bfloat16):
        output = m(x)

    loss = output.sum()
    loss.backward()

if __name__ == "__main__":
    main()
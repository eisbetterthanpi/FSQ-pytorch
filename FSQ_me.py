# @title FSQ me
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

def ste_round(x): return x.round().detach() + x - x.detach()

class FSQ(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.levels = torch.tensor(levels, device=device)
        self.basis = torch.cumprod(torch.tensor([*levels[1:], 1], device=device).flip(-1), dim=0).flip(-1)
        self.half_width = (self.levels-1)/2
        self.codebook_size = torch.prod(self.levels).item()
        self.codebook = self.indexes_to_codes(torch.arange(self.codebook_size, device=device))

    def forward(self, z, beta=1.0): # beta in (0,1). beta->0 => values more spread out
        offset = (self.levels+1) % 2 /2 # .5 if even, 0 if odd
        # bound = (F.sigmoid(2*z)-1/2) * (self.levels-beta) + offset
        bound = (F.tanh(z)/2) * (self.levels-beta) + offset
        quantized = ste_round(bound)
        return (quantized-offset) / self.half_width # split [-1,1]

    def codes_to_indexes(self, zhat):
        zhat = (zhat + 1) * self.half_width
        return (zhat * self.basis).sum(axis=-1).round().int()

    def indexes_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        codes = torch.remainder(indices//self.basis, self.levels)
        return codes / self.half_width - 1

fsq = FSQ(levels = [128,4,3,2])
# print(fsq.codebook)
batch_size, seq_len = 2, 4
# x = torch.linspace(-5,5,17).repeat(4,1).T # sig need larger variance to reach +-1
x = torch.linspace(-3,3,17).repeat(4,1).T
# x=la
la = fsq(x)
print(la)
lact = fsq.codes_to_indexes(la)
print(lact)
la = fsq.indexes_to_codes(lact)
print(la)

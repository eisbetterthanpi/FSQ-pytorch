# @title -1,1 code quant
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

def ste_round(x): return x.round().detach() + x - x.detach()

def soft_clamp_relu(x, lower=None, upper=None):
    if lower != None: x = lower+F.relu(x-lower)
    if upper != None: x = upper-F.relu(upper-x)
    return x

class codequant(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.levels = torch.tensor(levels, device=device)
        self.basis = torch.cumprod(torch.tensor([*levels[1:], 1], device=device).flip(-1), dim=0).flip(-1)
        self.half_width = (self.levels-1)/2
        self.codebook_size = torch.prod(self.levels).item()
        # self.codebook = self.indexes_to_codes(torch.arange(self.codebook_size, device=device))
    @property
    def codebook(self): return self.indexes_to_codes(torch.arange(self.codebook_size, device=device))

    def codes_to_indexes(self, zhat):
        zhat = soft_clamp_relu(zhat,-1,1)
        zhat = (zhat*-torch.pi/2).sin() # cos to unif
        zhat = (zhat + 1) * self.half_width
        return (zhat * self.basis).sum(axis=-1).round().int()

    def indexes_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        codes = torch.remainder(indices//self.basis, self.levels)
        # return codes / self.half_width - 1
        codes = codes / self.half_width - 1
        return soft_clamp_relu(codes,-1,1).asin()/torch.pi*2 # unif to cos

    def forward(self, z): # round code to nearest code
        offset = (self.levels+1) % 2 /2 # .5 if even, 0 if odd
        z = soft_clamp_relu(z,-1,1)
        z = (z*-torch.pi/2).sin() # cos to unif
        bound = z * self.half_width + offset
        quantized = ste_round(bound)
        # return (quantized-offset) / self.half_width # split [-1,1]
        zhat = (quantized-offset) / self.half_width # split [-1,1]
        return soft_clamp_relu(zhat,-1,1).asin()/torch.pi*2 # unif to cos

cq = codequant(levels = [128,4,3,2])
# print(cq.codebook)
batch_size, seq_len = 2, 4
# x = torch.linspace(-1.2,1.2,23).repeat(4,1).T
x = torch.linspace(-1.2,1.2,29).repeat(4,1).T
# x=la
la = cq(x)
print(la)
lact = cq.codes_to_indexes(la)
print(lact)
la = cq.indexes_to_codes(lact)
print(la)

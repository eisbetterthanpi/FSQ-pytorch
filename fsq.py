# @title FSQ torch
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"

def ste_round(x): return x.round().detach() + x - x.detach()

class FSQ(nn.Module): # https://colab.research.google.com/github/google-research/google-research/blob/master/fsq/fsq.ipynb
    def __init__(self, levels, eps = 1e-3):
        super().__init__()
        self.eps = eps
        self.levels = torch.tensor(levels, device=device)
        self.basis = torch.cat([torch.ones(1, device=device), torch.cumprod(self.levels[:-1], dim=0)]).long()
        self.num_dimensions = len(levels)
        self.codebook_size = torch.prod(self.levels).item()
        self.codebook = self.indexes_to_codes(torch.arange(self.codebook_size, device=device))

    def bound(self, z):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self.levels - 1) * (1 - self.eps) / 2
        offset = torch.where(self.levels % 2 == 1, 0.0, 0.5)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def forward(self, z):
        quantized = ste_round(self.bound(z))
        half_width = self.levels // 2 # Renormalize to [-1, 1]
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized): # Scale and shift to range [0, ..., L-1]
        half_width = self.levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self.levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat):
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * self.basis).sum(axis=-1).long()

    def indexes_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.fmod(indices // self.basis, self.levels)
        return self._scale_and_shift_inverse(codes_non_centered)

fsq = FSQ(levels = [3,3,2])

print(fsq.codebook)

batch_size, seq_len = 1, 1
x = torch.rand((batch_size, seq_len,3),device=device)

la = fsq(x)
print(la)
lact = fsq.codes_to_indexes(la)
print(lact)


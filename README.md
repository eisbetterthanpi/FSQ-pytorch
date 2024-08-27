# FSQ-pytorch

## Usage

```bash
fsq = FSQ(levels = [3,3,2])

print(fsq.codebook)

batch_size, seq_len = 1, 1
x = torch.rand((batch_size, seq_len,3),device=device)

la = fsq(x)
print(la)
lact = fsq.codes_to_indexes(la)
print(lact)
```


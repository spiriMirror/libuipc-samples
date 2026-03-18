import torch
import uipc.adapter.torch 

# create uipc buffer from torch (managed by torch)
tb = uipc.adapter.torch.buffer(dtype=torch.float32, device='cuda')
print(tb.buffer_view())
tb.resize(10)
print(tb.buffer_view())

# get torch tensor from uipc buffer
print(tb.torch())
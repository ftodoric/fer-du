import torch

x = torch.FloatTensor([[1], [2], [3]])
y = torch.FloatTensor([[1, 3, 3]])
print(torch.sum((x @ y), 0))

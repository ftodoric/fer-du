import torch

embedding = torch.nn.Embedding(10, 3)
x = torch.FloatTensor([[1], [2], [3]])
print(x)
print(x.squeeze(-1))

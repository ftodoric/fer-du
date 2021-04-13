import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# definiranje operacije
def f(x, a, b):
    return a * x + b


""" # definiranje varijabli i izgradnja dinamičnog
# računskog grafa s unaprijednim prolazom
a = torch.tensor(5., requires_grad=True)
b = torch.tensor(8., requires_grad=True)
x = torch.tensor(2.)
y = f(x, a, b)
s = a ** 2

# unatražni prolaz koji računa gradijent
# po svim tenzorima zadanim s requires_grad=True
y.backward()
s.backward()               # gradijent se akumulira
assert x.grad is None      # pytorch ne računa gradijente po x
assert a.grad == x + 2 * a  # dy/da + ds/da
assert b.grad == 1         # dy/db + ds/db

# ispis rezultata
print(f"y={y}, g_a={a.grad}, g_b={b.grad}") """

dataset = [(torch.randn(4, 4), torch.randint(5, size=())) for _ in range(25)]
dataset = [(x.numpy(), y.numpy()) for x, y in dataset]
loader = DataLoader(dataset, batch_size=8, shuffle=False,
                    num_workers=0, collate_fn=None, drop_last=False)
for x, y in loader:
    print(x.shape, y.shape)

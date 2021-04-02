import numpy as np

aa = np.random.rand(5, 5)
bb = np.random.rand(5, 5)

print(aa)
print(bb)

cc = ((aa > 0.5) & (bb > 0.5))
print(cc)

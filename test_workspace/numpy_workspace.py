import numpy as np

vec1 = np.random.randn(1, 3)
vec2 = np.random.randn(3, 1)

vec1 = [[1, 2, 3]]
vec2 = [[1], [2], [3]]

print(vec1 @ vec2)

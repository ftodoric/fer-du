import matplotlib.pyplot as plt
import numpy as np

a = np.linspace(0, 4, 50)
b = a**2

plt.plot(a, b, label="1")
plt.plot(b, a, label="-1")
plt.legend(loc="best")
plt.show()

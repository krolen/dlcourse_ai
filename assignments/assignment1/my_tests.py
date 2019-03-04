import numpy as np

arange = np.arange(0, 12)
print(arange)

arange_reshape = arange.reshape((4, 3))
print(arange_reshape)

print(arange_reshape[2, 1])

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])

np.linalg.norm((a - b), ord=1)

(a - b[0])

np.linalg.norm((a - b[0]), ord=1)

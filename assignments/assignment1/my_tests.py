import numpy as np

# arange = np.arange(0, 12)
# print(arange)
#
# arange_reshape = arange.reshape((4, 3))
# print(arange_reshape)
#
# print(arange_reshape[2, 1])
#
# a = np.array([1, 2, 3, 4])
# b = np.array([2, 3, 4, 5])
#
# np.linalg.norm((a - b), ord=1)
#
# (a - b[0])
#
# np.linalg.norm((a - b[0]), ord=1)

a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
b = np.array([[1, 10], [100, 1000], [10000, 100000]])

# print(a.repeat(2, axis=0))
# print(a.repeat(2, axis=1))
# print(a.transpose())
print(b[:, np.newaxis])
print(a.transpose())
print(a @ b.T)

# res0 = a.transpose() * b
# print(res0)
res = a * b[:, np.newaxis]
print(res)


def my_func(arr1, arr2):
    return arr1 * arr2 + 1


v_my_func = np.vectorize(my_func)

res2 = v_my_func(a, b[:, np.newaxis])

print(res2)


def one_test_distance(arr1, arr2):
    # return arr1 + arr2
    return np.linalg.norm((arr1 - arr2), ord=1)


# v_one_test_distance = np.vectorize(one_test_distance)
# v_one_test_distance(np.expand_dims(b, axis=1), a)

b1 = np.expand_dims(b, axis=1)

# print(one_test_distance(a.transpose().repeat(2, 1), b))

# b = np.array([[-1, 0, 1, 2], [-1, 3, 4, 5], [-1, 6, 7, 8]])
# print(b)

# print(b.reshape(1, 4, 3))
#
# a1 = np.arange(0, 60).reshape(5, 4, 3)
#
#
# def myprint(v):
#     print(v)
#     return v + 100
#
#
# a2 = np.apply_along_axis(myprint, 0, a1)
# print(a2)

def test(p1, p2):
    print(p1, p2)


v_test = np.vectorize(test)

res = v_test([[1, 2]], [[7, 8]])
print(res)

f1 = np.array([[0, 1, 2, 3, 4], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
print(f1)
f2 = np.array([[0, 10, 20, 30, 40], [60, 70, 80, 90, 100]])

f1 = np.array([[0, 1, 2, 3, 4], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
print(f1)

print(np.expand_dims(f1, 0))


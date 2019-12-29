import numpy as np

from RSG import RiemannianSubGradient

# data points span a 3-D subspace in a 4-D ambient space
X = np.array([[1, 2, 1, 1], [2, 4, 1, 2], [3, 6, 1, 1], [4, 8, 1, 2]])

RSG = RiemannianSubGradient()

# find a single direction that exactly orthogonal to the samples
RSG.c = 1
RSG.RSG_sphere(X)
print("==============(c=%d)\nB=" % RSG.c)
print(RSG.B)
print("Objective: %.4f" % RSG.loss_val)
print("Elapsed time: %.4fs" % RSG.elapsed_time)
print("num_iter: %d\n" % RSG.it)

# find two directions that "orthogonal" to the samples as much as possible
RSG.c = 2
RSG.RSG(X)
print("==============(c=%d)\nB=" % RSG.c)
print(RSG.B)
print("Objective: %.4f" % RSG.loss_val)
print("Elapsed time: %.4fs" % RSG.elapsed_time)
print("num_iter: %d\n" % RSG.it)

#  find three directions that "orthogonal" to the samples as much as possible
RSG.c = 3
RSG.RSG(X)
print("==============(c=%d)\nB=" % RSG.c)
print(RSG.B)
print("Objective: %.4f" % RSG.loss_val)
print("Elapsed time: %.4fs" % RSG.elapsed_time)
print("num_iter: %d\n" % RSG.it)

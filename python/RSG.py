import time
import warnings

import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import normalize


class RiemannianSubGradient:
    """Riemannian SubGradient solver
    
    The optimization objective for RiemannianSubGradient is the following
    least absolute distance problem:

        min_{B} ||X^T B||_{1,2}  s.t.  B^T B = I

    where:

        B : optimization variable with shape [n_features, n_dual_directions],
            and is constrained to have orthonormal columns

        X : data matrix with shape [n_features, n_samples]

        ||.||_{1,2} : mixed l1/l2 norm for any matrix A is defined by

            ||A||_{1,2} = \\sum_i ||row_i of A||_2

    One special case for this type of problem is the one on the sphere:

        min_{b} ||X^T b||_1  s.t.  ||b||_2 = 1

    In this case, we are only interested in finding a single dual direction
    that is orthogonal to the samples as much as possible.

    We solve the problem described above by Riemannian SubGradient (RSG) method, 
    which is proposed in our NeurIPS 2019 paper:

    Zhu, Z., Ding, T., Robinson, D.P., Tsakiris, M.C., & Vidal, R. (2019). 
    A Linearly Convergent Method for Non-Smooth Non-Convex Optimization on the 
    Grassmannian with Applications to Robust Subspace and Dictionary Learning.
    NeurIPS 2019.

    Please refer to the paper for details, and kindly cite our work 
    if you find it is useful.

    Parameters
    ----------
    mu_0 : float, optional
        The initial value of step size

    mu_min : float, optional
        The minimum value of step size that is allowed

    max_iter : int, optional
        The maximum number of iterations

    alpha : float, optional
        The line search paramter, which is chosen to be close to 0

    beta : float, optional
        The diminishing factor for step size

    c : int, optional
        The number of dual directions we aim to compute, which should
        satisfy 0 < c < num_features. If c == 1, the problem is on the
        sphere, and a single dual direction is computed.

    Attributes
    ----------
    B : array | ndarray, shape [n_features, c]
        computed optimization variable in the problem formulation

    loss_val: float
        final objective value

    it: int
        number of iterations performed

    elapsed_time: float
        elapsed time (in seconds) for running the algorithm

    Examples
    --------
        See demo.py
    """

    def __init__(
        self, mu_0=1e-2, mu_min=1e-15, max_iter=200, alpha=1e-3, beta=0.5, c=1
    ):
        self.mu_min = mu_min
        self.mu_0 = mu_0
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def RSG_sphere(self, X):
        """RSG solver for least absolute distance problem on the sphere:

            min_{b} ||X^T b||_1  s.t.  ||b||_2 = 1

        Parameters
        ----------
        X : ndarray matrix, [n_features, n_samples]
            Data
        """

        t_start = time.time()

        if self.c != 1:
            raise ValueError(
                "The problem is not on the sphere, please use the RSG function."
            )

        def loss(b):
            return LA.norm(X.T @ b, 1)

        eigs, evals = LA.eig(X @ X.T)
        b = evals[:, np.argmin(eigs)]

        mu = self.mu_0
        old_loss = loss(b)
        i = 0
        while mu > self.mu_min and i < self.max_iter:
            i += 1
            grad = X @ np.sign(X.T @ b)
            grad -= b * (grad.T.dot(b))
            grad_norm_square = LA.norm(grad) ** 2

            # modified line search
            b_next = b - mu * grad
            b_next /= LA.norm(b_next)
            while (
                loss(b_next) > old_loss - self.alpha * mu * grad_norm_square
                and mu > self.mu_min
            ):
                mu *= self.beta
                b_next = b - mu * grad
                b_next /= LA.norm(b_next)
            b = b_next
            old_loss = loss(b)

        self.B = b
        self.loss_val = old_loss
        self.it = i
        self.elapsed_time = time.time() - t_start

        return self

    def RSG(self, X):
        """RSG solver for group-wise least absolute distance problem:

            min_{B} ||X^T B||_{1,2}  s.t.  B^T B = I

        Parameters
        ----------
        X : ndarray matrix, [n_features, n_samples]
            Data
        """

        t_start = time.time()

        D = X.shape[0]

        if not (0 < self.c < D):
            raise ValueError("The problem is not well-defined.")

        if self.c == 1:
            warnings.warn(
                "The problem is on the sphere, RSG_sphere function is more efficient."
            )

        def loss(B):
            return np.sum(np.sqrt(np.sum((B.T @ X) ** 2, axis=0)))

        eigs, evals = LA.eigh(X @ X.T)
        B = evals[:, np.argsort(eigs)[: self.c]]

        mu = self.mu_0
        old_loss = loss(B)
        i = 0
        while mu > self.mu_min and i < self.max_iter:
            i += 1
            tmp = np.sqrt(np.sum((B.T @ X) ** 2, axis=0))
            indx = tmp > 0
            grad = X[:, indx] / np.tile(tmp[indx], (D, 1)) @ X[:, indx].T @ B
            grad -= B @ (B.T @ grad)
            grad_norm_square = LA.norm(grad) ** 2

            # modified line search
            B_next = normalize(B - mu * grad, axis=0)
            while (
                loss(B_next) > old_loss - self.alpha * mu * grad_norm_square
                and mu > self.mu_min
            ):
                mu *= self.beta
                B_next = normalize(B - mu * grad, axis=0)
            B = B_next
            old_loss = loss(B)

        self.B = B
        self.loss_val = old_loss
        self.it = i
        self.elapsed_time = time.time() - t_start

        return self

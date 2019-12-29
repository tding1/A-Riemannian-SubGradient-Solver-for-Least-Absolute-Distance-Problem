/** This header file contains the c++ implementation of the
  * Riemannian SubGradient (RSG) solver, in which we rely on two c++
  * scientific computing libraries, namely, Eigen and libigl, to
  * complete the relevant numerical tasks.
  *
  * The optimization objective is the following least absolute distance problem:
  *     
  *     min_{B} ||X^T B||_{1,2}  s.t.  B^T B = I
  * 
  * where:
  * 
  *     B : optimization variable with shape [n_features, n_dual_directions],
  *         and is constrained to have orthonormal columns
  *
  *     X : data matrix with shape [n_features, n_samples]
  *
  *     ||.||_{1,2} : mixed l1/l2 norm for any matrix A is defined by
  *
  *         ||A||_{1,2} = \\sum_i ||row_i of A||_2
  *
  * One special case for this type of problem is the one on the sphere:
  *
  *      min_{b} ||X^T b||_1  s.t.  ||b||_2 = 1
  *
  *  In this case, we are only interested in finding a single dual direction
  *  that is orthogonal to the samples as much as possible.
  *
  *  We solve the problem described above by RSG method, which is proposed 
  *  in our NeurIPS 2019 paper:
  *
  *  Zhu, Z., Ding, T., Robinson, D.P., Tsakiris, M.C., & Vidal, R. (2019). 
  *  A Linearly Convergent Method for Non-Smooth Non-Convex Optimization on the 
  *  Grassmannian with Applications to Robust Subspace and Dictionary Learning.
  *  NeurIPS 2019.
  *
  *  Please refer to the paper for details, and kindly cite our work 
  *  if you find it is useful.
  * 
  */
#ifndef RSG_H
#define RSG_H

#include "Eigen/Core"
#include "Eigen/Dense"
#include "igl/find.h"
#include "igl/orth.h"
#include <time.h>

using namespace Eigen;

/** Data type for RSG parameters.
  * 
  * mu_0      : initial value of step size
  * mu_min    : minimum value of step size that is allowed
  * max_iter  : maximum number of iterations
  * alpha     : line search paramter, which is chosen to be close to 0
  * beta      : diminishing factor for step size
  * c         : number of dual directions we aim to compute, which should
  *             satisfy 0 < c < num_features 
  *             If c == 1, the problem is on the sphere, and 
  *             a single dual direction is computed.
  */
typedef struct
{
    double mu_0 = 1e-2;
    double mu_min = 1e-15;
    int max_iter = 200;
    double alpha = 1e-3;
    double beta = 0.5;
    int c = 1;
} parms_rsg;

/** Data type for RSG outputs.
  * 
  * B_star        : matrix with shape [n_features, n_dual_directions]
  *                 computed optimization variable
  * loss_val      : scalar (double)
  *                 final objective value
  * iter          : scalar (int)
  *                 number of iterations performed
  * elapsed_time  : scalar (double)
  *                 elapsed time (in seconds) for running the algorithm
  */
typedef struct
{
    double loss_val;
    MatrixXd B_star;
    double elapsed_time;
    int iter;
} out_rsg;

double loss(const MatrixXd &X, const VectorXd &b)
{
    // compute ||X^T b||_1
    return (X.transpose() * b).lpNorm<1>();
}

double loss(const MatrixXd &X, const MatrixXd &B)
{
    // compute ||X^T B||_{1,2}
    return (B.transpose() * X).array().square().colwise().sum().array().sqrt().sum();
}

/** RSG solver for least absolute distance problem on the sphere:
  *       
  *     min_{b} ||X^T b||_1  s.t.  ||b||_2 = 1         
  */
void RSG_sphere(const parms_rsg &parms, const MatrixXd &X, out_rsg &out)
{
    clock_t tStart = clock();

    if (parms.c != 1)
    {
        fprintf(stderr, "Error: The problem is not on the sphere, please use the RSG solver.\n");
        return;
    }

    SelfAdjointEigenSolver<MatrixXd> es(X * X.transpose());
    VectorXd b = es.eigenvectors().col(0);

    double old_loss = loss(X, b);
    double mu = parms.mu_0;
    VectorXd b_next, grad;

    int i = 0;
    while (mu > parms.mu_min && i < parms.max_iter)
    {
        i++;
        grad = X * (VectorXd)(X.transpose() * b).array().sign();
        grad -= b * (grad.transpose() * b);
        double grad_norm_square = pow(grad.lpNorm<2>(), 2);
        b_next = (b - mu * grad).normalized();
        while (loss(X, b_next) > old_loss - parms.alpha * mu * grad_norm_square && mu > parms.mu_min)
        {
            mu *= parms.beta;
            b_next = (b - mu * grad).normalized();
        }
        b = b_next;
        old_loss = loss(X, b);
    }

    out.B_star = b;
    out.loss_val = old_loss;
    out.iter = i;
    out.elapsed_time = (double)(clock() - tStart) / CLOCKS_PER_SEC;
}

/** RSG solver for group-wise least absolute distance problem:
  *       
  *     min_{B} ||X^T B||_{1,2}  s.t.  B^T B = I    
  */
void RSG(const parms_rsg &parms, const MatrixXd &X, out_rsg &out)
{
    clock_t tStart = clock();

    const int D = X.rows();

    if (parms.c < 0 || parms.c >= D)
    {
        fprintf(stderr, "Error: The problem is not well-defined.\n");
        return;
    }

    if (parms.c == 1)
    {
        printf("Warning: The problem is on the sphere, RSG_sphere solver is more efficient.\n");
    }

    SelfAdjointEigenSolver<MatrixXd> es(X * X.transpose());
    MatrixXd B = es.eigenvectors().leftCols(parms.c);

    double old_loss = loss(X, B);
    double mu = parms.mu_0;
    MatrixXd B_next, grad;

    int i = 0;
    while (mu > parms.mu_min && i < parms.max_iter)
    {
        i++;
        ArrayXd tmp = (B.transpose() * X).array().square().colwise().sum().array().sqrt();
        ArrayXd indx;
        igl::find(tmp > 0, indx);

        grad = X(all, indx).array() / tmp(indx, 1).transpose().replicate(D, 1).array();
        grad *= X(all, indx).transpose() * B;
        grad -= B * (B.transpose() * grad);
        double grad_norm_square = pow(grad.lpNorm<2>(), 2);

        igl::orth(B - mu * grad, B_next);
        while (loss(X, B_next) > old_loss - parms.alpha * mu * grad_norm_square && mu > parms.mu_min)
        {
            mu *= parms.beta;
            igl::orth(B - mu * grad, B_next);
        }
        B = B_next;
        old_loss = loss(X, B);
    }

    out.B_star = B;
    out.loss_val = old_loss;
    out.iter = i;
    out.elapsed_time = (double)(clock() - tStart) / CLOCKS_PER_SEC;
}

#endif

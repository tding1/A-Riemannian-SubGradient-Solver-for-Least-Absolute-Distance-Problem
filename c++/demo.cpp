#include <iostream>
#include "RSG.h"

int main()
{
    // data points span a 3-D subspace in a 4-D ambient space
    MatrixXd X(4, 4);
    X << 1, 2, 1, 1,
        2, 4, 1, 2,
        3, 6, 1, 1,
        4, 8, 1, 2;

    parms_rsg parms;
    out_rsg out;

    // find a single direction that exactly orthogonal to the samples
    parms.c = 1;
    RSG_sphere(parms, X, out);
    printf("==============(c=%d)\nB=\n", parms.c);
    std::cout << out.B_star << std::endl;
    printf("Objective: %.4f\n", out.loss_val);
    printf("Elapsed time: %.4fs\n", out.elapsed_time);
    printf("num_iter: %d\n\n", out.iter);

    // find two directions that "orthogonal" to the samples as much as possible
    parms.c = 2;
    RSG(parms, X, out);
    printf("==============(c=%d)\nB=\n", parms.c);
    std::cout << out.B_star << std::endl;
    printf("Objective: %.4f\n", out.loss_val);
    printf("Elapsed time: %.4fs\n", out.elapsed_time);
    printf("num_iter: %d\n\n", out.iter);

    // find three directions that "orthogonal" to the samples as much as possible
    parms.c = 3;
    RSG(parms, X, out);
    printf("==============(c=%d)\nB=\n", parms.c);
    std::cout << out.B_star << std::endl;
    printf("Objective: %.4f\n", out.loss_val);
    printf("Elapsed time: %.4fs\n", out.elapsed_time);
    printf("num_iter: %d\n\n", out.iter);

    return 0;
}

function [B, loss_val, elapsed_time, iter] = RSG(X, parms)
% The Riemannian SubGradient (RSG) solver for group-wise 
% least absolute distance problem:
%
%     min_{B} ||X^T B||_{1,2}  s.t.  B^T B = I
% 
% where:
%
%     B : optimization variable with shape [n_features, n_dual_directions],
%         and is constrained to have orthonormal columns
%
%     X : data matrix with shape [n_features, n_samples]
% 
%     ||.||_{1,2} : mixed l1/l2 norm for any matrix A is defined by
% 
%         ||A||_{1,2} = \\sum_i ||row_i of A||_2
% 
% We solve the problem described above by RSG method, which is proposed in
% our NeurIPS 2019 paper:
% 
%     Zhu, Z., Ding, T., Robinson, D.P., Tsakiris, M.C., & Vidal, R. (2019). 
%     A Linearly Convergent Method for Non-Smooth Non-Convex Optimization on the 
%     Grassmannian with Applications to Robust Subspace and Dictionary Learning.
%     NeurIPS 2019.
% 
% Please refer to the paper for details, and kindly cite our work if you find it is useful.
% 
% PARAMETERS
% ----------
% 
%     See parms_rsg.m for details
% 
% OUTPUT
% ------
% 
%     B : matrix with shape [n_features, n_dual_directions]
%         computed optimization variable
% 
%     loss_val: scalar (float)
%         final objective value
% 
%     iter: scalar (int)
%         number of iterations performed
% 
%     elapsed_time: scalar (float)
%         elapsed time (in seconds) for running the algorithm
% 

t_start = tic;

D = size(X, 1);

if parms.c < 0 || parms.c >= D
    error('The problem is not well-defined.')
end

if parms.c == 1
    warning('The problem is on the sphere, RSG_sphere function is more efficient.')
end

loss = @(B) sum(sqrt(sum((B'*X).^2,1)));

[B,~] = eigs(X * X', parms.c, 'SM');

mu = parms.mu_0;
old_loss = loss(B);
i = 0;
while mu > parms.mu_min && i < parms.max_iter
    i = i + 1;
    
    tmp = sqrt(sum((B'*X).^2,1)); indx = tmp>0;
    grad = (X(:,indx)./repmat(tmp(indx),D,1))*X(:,indx)'*B;
    grad = grad - B*(B'*grad);
    grad_norm_square = norm(grad)^2;

    % modified line search
    tmp = orth(B - mu * grad);
    while loss(tmp) > old_loss - parms.alpha * mu * grad_norm_square && mu > parms.mu_min
        mu = mu * parms.beta;
        tmp = orth(B - mu * grad);
    end
    B = tmp;
    old_loss = loss(B);
end

loss_val = old_loss;
iter = i;
elapsed_time = toc(t_start);

end
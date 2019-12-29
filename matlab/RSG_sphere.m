function [b, loss_val, elapsed_time, iter] = RSG_sphere(X, parms)
% The Riemannian SubGradient (RSG) solver for least absolute distance 
% problem on the sphere:
% 
%     min_{b} ||X^T b||_1  s.t.  ||b||_2 = 1
% 
% where:
%
%     b : optimization variable with shape [n_features, 1],
%         and is constrained to be on the sphere
%
%     X : data matrix with shape [n_features, n_samples]
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
%     b : vector with shape [n_features, 1]
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

if parms.c ~= 1
    error('The problem is not on the sphere, please use the RSG function.')
end

loss = @(b) norm(X' * b, 1);

% initializationie
[b,~] = eigs(X * X',1,'SM');

mu = parms.mu_0;
old_loss = loss(b);
i = 0;
while mu > parms.mu_min && i < parms.max_iter
    i = i + 1;
    grad = X * sign(X' * b);
    grad = grad - b * (grad' * b);
    grad_norm_square = norm(grad)^2;

    % modified line search
    tmp = normc(b - mu * grad);
    while loss(tmp) > old_loss - parms.alpha * mu * grad_norm_square && mu > parms.mu_min
        mu = mu * parms.beta;
        tmp = normc(b - mu * grad);
    end
    b = tmp;
    old_loss = loss(b);
end

loss_val = old_loss;
iter = i;
elapsed_time = toc(t_start);

end
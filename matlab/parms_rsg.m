function parms = parms_rsg
% Parameter specification for the Riemannian SubGradient solver, see
% 
%          RSG.m  and  RSG_sphere.m  for the implementation
% 
% PARAMETERS
% ----------
%
%   mu_0 : initial value of step size
%
%   mu_min : minimum value of step size that is allowed
%
%   max_iter : maximum number of iterations
%
%   alpha : line search paramter, which is chosen to be close to 0
%
%   beta : diminishing factor for step size
%
%   c : number of dual directions we aim to compute, which should
%       satisfy 0 < c < num_features 
%       If c == 1, the problem is on the sphere, and 
%       a single dual direction is computed.

    parms.mu_0 = 1e-2;
    parms.mu_min = 1e-15;
    parms.max_iter = 200;
    parms.alpha = 1e-3;
    parms.beta = 0.5;
    parms.c = 1;
end
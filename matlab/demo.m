clc
clear

% data points span a 3-D subspace in a 4-D ambient space
X = [1 2 1 1; 
     2 4 1 2; 
     3 6 1 1;
     4 8 1 2];

parms = parms_rsg;

% find a single direction that exactly orthogonal to the samples
parms.c = 1;
[B, f, t, iter] = RSG_sphere(X, parms);
fprintf('==============(c=%d)\nB=\n', parms.c)
disp(B)
fprintf('Objective: %.4f\n', f)
fprintf('Elapsed time: %.4fs\n', t)
fprintf('num_iter: %d\n\n', iter)

% find two directions that "orthogonal" to the samples as much as possible
parms.c = 2;
[B, f, t, iter] = RSG(X, parms);
fprintf('==============(c=%d)\nB=\n', parms.c)
disp(B)
fprintf('Objective: %.4f\n', f)
fprintf('Elapsed time: %.4fs\n', t)
fprintf('num_iter: %d\n\n', iter)

% find three directions that "orthogonal" to the samples as much as possible
parms.c = 3;
[B, f, t, iter] = RSG(X, parms);
fprintf('==============(c=%d)\nB=\n', parms.c)
disp(B)
fprintf('Objective: %.4f\n', f)
fprintf('Elapsed time: %.4fs\n', t)
fprintf('num_iter: %d\n\n', iter)

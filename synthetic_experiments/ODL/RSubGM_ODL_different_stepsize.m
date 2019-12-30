
close all; clear all;
randn('seed',2018);rand('seed',2018)

%% setup the data
theta = .3;   % sparsity level
D = 70;   % dimension
p = 1.5;   % sample complexity (as power of n)
success_vec = zeros(length(p), 1);
m = round(10*D^p);    % number of measurements
Q = randU(D);     % a uniformly random orthogonal matrix
X = randn(D, m).*(rand(D, m) <= theta);   % iid Bern-Gaussian model
Xtilde = Q*X;
obj = @(b)norm(Xtilde'*b,1);
% random initialization
bo = normc(randn(D,1));

%% geometrical constant
mu_o = 1e1;
beta = .9;
Niter = 200;
b = bo;
for i = 1:Niter
    mu = mu_o*beta^(i);
    grad = sum( repmat(sign(b'*Xtilde),D,1).*Xtilde, 2)/m;grad = grad - b*(b'*grad);
    b = normc(b - mu*grad);
    temp = Q'*b;  [~,indx] = max(abs(temp));
    dist_geometrical(i) = sqrt(norm(temp)^2 - 2*temp(indx)*sign(temp(indx))+1);
    
end
%% constant step size
b = bo;
mu_o = .1;
Niter = 1000;
for i = 1:Niter
    mu = mu_o;
    grad = sum( repmat(sign(b'*Xtilde),D,1).*Xtilde, 2)/m;grad = grad - b*(b'*grad);
    b = normc(b - mu*grad);
    temp = Q'*b;  [~,indx] = max(abs(temp));
    dist_constant(i) = sqrt(norm(temp)^2 - 2*temp(indx)*sign(temp(indx))+1);
end
%% linear decay: mu_k = mu_0/k
b = bo; mu_o = 10;
for i = 1:Niter
    mu = mu_o/i;
    grad = sum( repmat(sign(b'*Xtilde),D,1).*Xtilde, 2)/m;grad = grad - b*(b'*grad);
    b = normc(b - mu*grad);
    temp = Q'*b;  [~,indx] = max(abs(temp));
    dist_decaylinear(i) = sqrt(norm(temp)^2 - 2*temp(indx)*sign(temp(indx))+1);
end
%% mu_k = mu_0/k^{3/8}
b = bo;
mu_o = 1;
for i = 1:Niter
    mu = mu_o/i^(3/8);
    grad = sum( repmat(sign(b'*Xtilde),D,1).*Xtilde, 2)/m; rgrad = grad - (b'*grad)*b;
    b = normc(b - mu*rgrad);
    temp = Q'*b;  [~,indx] = max(abs(temp));
    dist_decayroot(i) = sqrt(norm(temp)^2 - 2*temp(indx)*sign(temp(indx))+1);
end
%%
fontsize = 26;
plotStyle = {'b-','k:','m--','r-','g:'};
figure
semilogy(0:length(dist_constant)-1,dist_constant,plotStyle{1},'linewidth',2);
legendInfo{1} = ['$\mu_k = 0.1$']; hold on
semilogy(0:length(dist_decaylinear)-1,dist_decaylinear,plotStyle{2},'linewidth',2);
legendInfo{2} = ['$\mu_k = 10/{k}$'];
semilogy(0:length(dist_decayroot)-1,dist_decayroot,plotStyle{3},'linewidth',2);
legendInfo{3} = ['$\mu_k = 1/{k^{3/8}}$'];
semilogy(0:length(dist_geometrical)-1,dist_geometrical,plotStyle{4},'linewidth',2);
legendInfo{4} = ['$\mu_k = 10\times 0.9^k$'];
ylim([1e-8,max(dist_constant)*1.2])
legend(legendInfo,'Interpreter','LaTex','Location','Best')
xlabel('iteration','FontSize',fontsize,'FontName','Times New Roman');
ylabel('dist','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
set(gca,'YDir','normal')
set(gca, ...
    'LineWidth' , 2                     , ...
    'FontSize'  , fontsize              , ...
    'FontName'  , 'Times New Roman'         );
set(gcf, 'Color', 'white');

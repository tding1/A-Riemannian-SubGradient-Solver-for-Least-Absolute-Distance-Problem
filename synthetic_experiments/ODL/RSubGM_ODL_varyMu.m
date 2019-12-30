close all; clear all;
randn('seed',2018);rand('seed',2018)
D = 30;  N = 500;

% generate data
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

%%
maxiter = 2e2;
mu_list = [1e0 1e1 1e2 1e3];
beta = 0.9;


for i_mu = 1:length(mu_list)
    mu_o = mu_list(i_mu);
    %bo = normc(randn(D,1));
    b = bo;
    i = 0;
    while i<= maxiter
        i = i+1;
        grad = sum( repmat(sign(b'*Xtilde),D,1).*Xtilde, 2)/m;grad = grad - b*(b'*grad);
        
        mu = mu_o*beta^(i);
        b = normc(b - mu*grad);
         temp = Q'*b;  [~,indx] = max(abs(temp));
        angle(i,i_mu) = sqrt(norm(temp)^2 - 2*temp(indx)*sign(temp(indx))+1);
    end
end

%%
fontsize = 30;
plotStyle = {'b-','g:','k--','r-','b:','r--','k-','k:','k--'};
figure
for i_mu = 1:length(mu_list)
        semilogy(0:length(find(angle(:,i_mu)>0))-1,angle(find(angle(:,i_mu)>0),i_mu),plotStyle{i_mu},'linewidth',2);
        legendInfo{i_mu} = ['\mu_0 = ' num2str(mu_list(i_mu))];
        
        hold on
end
ylim([min(min(min(angle)))*0.99,max(max(max(angle)))*2])
xlim([0 size(angle,1)])
legend(legendInfo,'Location','Best')
xlabel('iteration','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
ylabel('dist','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
set(gca,'YDir','normal')
set(gca, ...
    'LineWidth' , 2                     , ...
    'FontSize'  , fontsize              , ...
    'FontName'  , 'Times New Roman'         );
set(gcf, 'Color', 'white');
close all; clear all;
randn('seed',2018);rand('seed',2018)

%% setup the data
D = 100; %ambient dimension
N = 1500; % number of inliers
ratio = 1 ./ (1 ./ 0.7 - 1); % outlier ratio
M = floor(N * ratio); % number of outliers
d = 90; % subspace dimension
c = D -d;
X = [normc( randn(d,N) );zeros(D-d,N)];
O = normc(randn(D,M));
Xtilde = [X O];
obj = @(B) sum(sqrt(sum((B'*Xtilde).^2,1)));
% initialization
[Bo,~] = eigs(Xtilde*Xtilde',c,'SM');

%% geometrical constant
%%% line search to determine initial step size

mu_o = 1e-2;
beta = .8;
Niter = 300;
B = Bo;
for i = 1:Niter
    mu = mu_o*beta^(i);
    temp = sqrt(sum((B'*Xtilde).^2,1)); indx = temp>0;
    grad = (Xtilde(:,indx)./repmat(temp(indx),D,1))*Xtilde(:,indx)'*B;
    grad = grad - B*(B'*grad);
    B = orth(B - mu*grad); 
    dist_geometrical(i) = norm(B(1:d),'fro');
end


%% constant step size
B = Bo;
mu_o = 1e-4;
for i = 1:Niter
    mu = mu_o;
    temp = sqrt(sum((B'*Xtilde).^2,1)); indx = temp>0;
    grad = (Xtilde(:,indx)./repmat(temp(indx),D,1))*Xtilde(:,indx)'*B;
    grad = grad - B*(B'*grad);
    B = orth(B - mu*grad);
    dist_constant(i) = norm(B(1:d),'fro');
end
%% line decay
B = Bo; mu_o = 1e-2;
for i = 1:Niter
    mu = mu_o/i;
    temp = sqrt(sum((B'*Xtilde).^2,1)); indx = temp>0;
    grad = (Xtilde(:,indx)./repmat(temp(indx),D,1))*Xtilde(:,indx)'*B;
    grad = grad - B*(B'*grad);
    B = orth(B - mu*grad); 
    dist_decaylinear(i) = norm(B(1:d),'fro');
end

%% root k decay
B = Bo;
mu_o = 1e-3;
for i = 1:Niter
    mu = mu_o/sqrt(i);
    temp = sqrt(sum((B'*Xtilde).^2,1)); indx = temp>0;
    grad = (Xtilde(:,indx)./repmat(temp(indx),D,1))*Xtilde(:,indx)'*B;
    grad = grad - B*(B'*grad);
    B = orth(B - mu*grad);
    dist_decayroot(i) = norm(B(1:d),'fro');
end

%%
fontsize = 26;
plotStyle = {'b-','k:','m--','r-','g:'};
figure
semilogy(0:length(dist_constant)-1,dist_constant,plotStyle{1},'linewidth',2);
legendInfo{1} = ['$\mu_k = 10^{-4}$']; hold on
semilogy(0:length(dist_decaylinear)-1,dist_decaylinear,plotStyle{2},'linewidth',2);
legendInfo{2} = ['$\mu_k = {10^{-2}}/{k}$'];
semilogy(0:length(dist_decayroot)-1,dist_decayroot,plotStyle{3},'linewidth',2);
legendInfo{3} = ['$\mu_k = {10^{-3}}/{\sqrt{k}}$'];
semilogy(0:length(dist_geometrical)-1,dist_geometrical,plotStyle{4},'linewidth',2);
legendInfo{4} = ['$\mu_k = {10^{-2}}\times 0.8^k$'];
ylim([1e-14,max(dist_constant)*1.2])
legend(legendInfo,'Interpreter','LaTex','Location','Best')
xlabel('iteration','FontSize',fontsize,'FontName','Times New Roman');
ylabel('dist','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
set(gca,'YDir','normal')
set(gca, ...
    'LineWidth' , 2                     , ...
    'FontSize'  , fontsize              , ...
    'FontName'  , 'Times New Roman'         );
set(gcf, 'Color', 'white');

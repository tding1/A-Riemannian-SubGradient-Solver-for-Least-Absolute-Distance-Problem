close all; clear all;
randn('seed',2018);rand('seed',2018)
D = 100;  N = 1500;

ratio = 1 ./ (1 ./ 0.7 - 1);
M = floor(N * ratio);

d = 90;
c = D -d;
maxiter = 2e2;
beta_list = [0.5 0.6 0.7 0.8 0.9];
mu_o = 1e-2;

X = [normc( randn(d,N) );zeros(D-d,N)];
O = normc(randn(D,M));
Xtilde = [X O];

% initialization
[Bo,~] = eigs(Xtilde*Xtilde',c,'SM');
for i_beta = 1:length(beta_list)
    beta = beta_list(i_beta);
    B = Bo;
    i = 1;
    
    dist(1,i_beta) = norm(B(1:d),'fro');
    while i<= maxiter
        i = i+1;
         mu = mu_o*beta^(i);
    temp = sqrt(sum((B'*Xtilde).^2,1)); indx = temp>0;
    grad = (Xtilde(:,indx)./repmat(temp(indx),D,1))*Xtilde(:,indx)'*B;
    grad = grad - B*(B'*grad);
    B = orth(B - mu*grad); 
    dist(i,i_beta) = norm(B(1:d),'fro');
    end
end

%%
fontsize = 30;
plotStyle = {'b-','g:','k--','r-','b:','r--','k-','k:','k--'};
figure
for i_beta = 1:length(beta_list)
    semilogy(0:length(find(dist(:,i_beta)>0))-1,dist(find(dist(:,i_beta)>0),i_beta),plotStyle{i_beta},'linewidth',3);
    legendInfo{i_beta} = ['\beta = ' num2str(beta_list(i_beta))];
    
    hold on
end
ylim([1e-12,1])
xlim([0 size(dist,1)])
legend(legendInfo,'Location','Best')
xlabel('iteration','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
ylabel('dist','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
set(gca,'YDir','normal')
set(gca, ...
    'LineWidth' , 2                     , ...
    'FontSize'  , fontsize              , ...
    'FontName'  , 'Times New Roman'         );
set(gcf, 'Color', 'white');

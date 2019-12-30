close all; clear all;
randn('seed',2018);rand('seed',2018)
D = 30;  N = 500;
d = 25;

MNratio = 0.1:0.1:0.9;
ratio = 1 ./ (1 ./ MNratio - 1);
M_list = floor(N * ratio);

maxiter = 1e2;


beta = .8;

eps_ls = 0.01; alpha = 0.001; beta_ls = 0.5;


for id = 1:length(M_list)
    M = M_list(id);
    X = [normc( randn(d,N) );zeros(D-d,N)];
    O = normc(randn(D,M));
    Xtilde = [X O];
    obj = @(B) sum(sqrt(sum((B'*Xtilde).^2,1)));
    % initialization
    c = D -d;
    [Bo,~] = eigs(Xtilde*Xtilde',c,'SM');
    B = Bo;
    
    %%% line search to determine initial step size
    temp = sqrt(sum((B'*Xtilde).^2,1)); indx = temp>0;
    grad = (Xtilde(:,indx)./repmat(temp(indx),D,1))*Xtilde(:,indx)'*B;
    grad = grad - B*(B'*grad);
    grad_norm = norm(grad,'fro')^2;
    eps = eps_ls;
    obj_old = obj(B);
    
    while obj( orth(B - eps*grad) )> obj_old - alpha*eps*grad_norm
        eps = eps*beta_ls;
    end
    
    
    eps_o = eps;
    i = 1;
    value(1,id) = obj(B);obj_old = value(1,id);
    dist(1,id) = norm(B(1:d),'fro');
    while i<= maxiter
        i = i+1;
        temp = sqrt(sum((B'*Xtilde).^2,1)); indx = temp>0;
        grad = (Xtilde(:,indx)./repmat(temp(indx),D,1))*Xtilde(:,indx)'*B;
        grad = grad - B*(B'*grad);
        eps = eps_o*beta^(i);
        B = orth(B - eps*grad);
        value(i,id) = obj(B);obj_old = value(i,id);
        dist(i,id) = norm(B(1:d),'fro');
    end
end


plotStyle = {'b-','b:','b--','r-','r:','r--','k-','k:','k--'};
fontsize = 22;
figure
for id = 1:length(M_list)
    semilogy(0:length(find(dist(:,id)>0))-1,dist(find(dist(:,id)>0),id),plotStyle{id},'linewidth',2);
    legendInfo{id} = ['\gamma = ' num2str(MNratio(id))];
    hold on
end
ylim([min(min(dist))*0.99,max(max(dist))*2])
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
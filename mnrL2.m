function [B, B0] = mnrL2(X,Y,lambda, param)

[n, k] = size(X);
[~, q] = size(Y);
numClasses = length(unique(Y));
gtol = 1e-4;

%% do cgd for B
B_not_converged = true;
niter = 0;
Binit = rand(k+1,numClasses);
B = Binit(2:end,:);
Bold = B;
B0 = B(1,:);
B0old = B0;
alpha = 0.0001;
normGold = inf;
while B_not_converged
    %Compute gradient
    [dfdB, dfdB0] = Bgrad(X, Y, B, B0, numClasses, k, n,lambda, param);
    %check if converged
    normG = norm([dfdB0; dfdB], 'fro')^2;
    if (norm([dfdB0; dfdB], 'fro')^2 < gtol) || niter > 1000
        B_not_converged = false;
    elseif normG > normGold
        B = Bold;
        B0 = B0old;
        B_not_converged = false;
    end
    %first iter of conjugate gradient
    if niter == 0
        % if first iter do optimal gd update
        S =  -dfdB;
        S0 = -dfdB0;
        %                 alpha = fminsearch(@(t) cost_fun(L, B + t*S, B0 + t*S0,...
        %                     X, Ymask, Xnorm, n, lambda), 0.1)
        B = B + alpha*S;
        B0 = B0 + alpha*S0;
        Sold = S;
        S0old = S0;
        dfdBold = dfdB;
        dfdB0old = dfdB0;
    %all other iters of conjugate gradient
    else
        %compute conjugate direction using Polak-Ribière
        %                 PR = max(0, sum(-dfdB.*(-dfdB + dfdBold), 'all')./ sum(-dfdB.*-dfdB, 'all'));
        %                 PR0 = max(0, sum(-dfdB0.*(-dfdB0 + dfdB0old), 'all')./ sum(-dfdB0.*-dfdB0,'all'));
        fr = (norm([dfdB0;dfdB], 'fro')/ norm([dfdB0old;dfdBold], 'fro'))^2;
        S = -dfdB + Sold*fr;
        S0 = -dfdB0 + S0old*fr;
        %                 [alpha, val] = fminsearch(@(t) cost_fun(L, B + t*S, B0 + t*S0,...
        %                                             X, Ymask, Xnorm, n, lambda), 0.1 )
        Bnew = B + alpha*S;
        Bold = B; B = Bnew;
        B0new = B0 + alpha*S0;
        B0old = B0; B0 = B0new;
        Sold = S;
        S0old = S0;
        dfdBold = dfdB;
        dfdB0old = dfdB0;
        niter = niter+1;
    end
    normGold = normG;
    niter = niter+1;
    
    
end


end

function [dfdB, dfdB0] = Bgrad(X, Y, B, B0, numClasses, k, n, lambda, param)
%compute gradient
dfdB = zeros(k, numClasses);
dfdB0 = zeros(1, numClasses);
for j = 1:numClasses
    Xj = X(Y==j, :); % nj x p
    expM = exp(Xj*B + B0);
    weights = expM(:,j).*((1- sum(expM,2))./sum(expM,2));
    dfdB(:,j) = -(1/n)*(1-lambda)*sum((Xj.*weights)', 2) + 2*param*B(:,j);
    dfdB0(j) = -(1/n)*(1-lambda)*sum(weights) + 2*param*B0(j);
end
end


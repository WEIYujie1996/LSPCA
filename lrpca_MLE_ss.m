function [Zu, Yu, Zl, L, B] = lrpca_MLE_ss(Xu, Xl, Yl, k, Linit)

% Inputs:
%       X: (n x p) data matrix columns are features rows are
%       observations
%
%       Y: (n x 1) Categorical Response Variables (1, 2, ...,
%       numClasses)
%
%       gamma: tuning parameter
%
%       sigma: gaussian kernel parameter
%
%       k: desired number of reduced dimensions
%
%       Linit: (pxk) initial guess at a subspace
%           -default: pass in L0 = 0 and first k principle
%           components will be used
%
%       numIter: number of iterations to run the optimization
%       program
%
%       maxSubIter: maximum number of iterations to solve for L
%       during each outer iteration
%
%
% Outputs:
%
%       Z: (n x k) dimension reduced form of X; A = X*L'
%
%       L: (p x k) matrix with colspanspan equal to the desired subspace
%
%       B: (k x numClasses) coefficients mapping reduced X to Y
%
%       B0: (k x 1) bias of the coefficients
%

%store dimensions:
[nl, p] = size(Xl);
[nu, ~] = size(Xu);
n = nu + nl;
[~, q] = size(Yl);
X = [Xl; Xu];
%norms
Xnorm = norm(X, 'fro');
Ynorm = norm(Yl, 'fro');
numClasses = length(unique(Yl));
Ymask = zeros(nl,numClasses); Ymask(sub2ind(size(Ymask), (1:nl)', Yl)) = 1;

% initialize L0 by PCA of X, and B0 by L0
if sum(abs(Linit), 'all') == 0
    Linit = pca(X);
    Linit = Linit(:,1:k);
end

%solve the problem using CG on the grassmann manifold
L = Linit;
Binit = mnrfit(Xl*L,Ymask, 'interactions', 'on');
Binit = [Binit, zeros(k+1,1)];
B0 = Binit(1,:);
B = Binit(2:end,:);
var_x = (n*(p-k))^-1 * ((norm(X, 'fro')^2-norm(X*L, 'fro')^2));
alpha = max((n*k)^-1 * norm(X*L, 'fro')^2 - var_x, 0);
eta = sqrt(var_x + alpha) - sqrt(var_x);
gamma = (var_x + eta) / eta;
niter = 0;
notConverged = true;
fstar = inf;
while notConverged
    %% Update old vars
    Lprev = L;
    fstarprev = fstar;
    
    %% L step
    % set up the optimization subproblem in manopt
    warning('off', 'manopt:getHessian:approx')
    warning('off', 'manopt:getgradient:approx')
    manifold = grassmannfactory(p, k, 1);
    problem.M = manifold;
    problem.cost  = @(Ltilde) cost_fun(Ltilde, B, B0, X, Xl, Ymask, Xnorm, n, nl, p, k, var_x, alpha);
    problem.egrad = @(Ltilde) Lgrad(Ltilde, B, B0, X, Xl, Yl, Xnorm, numClasses, nl, p, k, var_x, gamma);
    options.verbosity = 0;
    %options.minstepsize = 1e-12;
    options.stopfun = @mystopfun;
    [L, fstar, ~, options] = conjugategradient(problem, L, options);
    %[L, fstar, ~, options] = steepestdescent(problem, L, options);

    %% B step
    B = mnrfit(Xl*L,Ymask, 'interactions', 'on');
    B = [B, zeros(k+1,1)];
    B0 = B(1,:);
    B = B(2:end,:);
    
    %% update var_x
    if alpha>0
        var_x = (n*(p-k))^-1 * ((norm(X, 'fro')^2-norm(X*L, 'fro')^2));
    else
        var_x = (n*p)^-1 * norm(X, 'fro')^2;
    end
    
    %% update alpha
    alpha = max((n*k)^-1 * norm(X*L, 'fro')^2 - var_x, 0);
    eta = sqrt(var_x + alpha) - sqrt(var_x);
    gamma = (var_x + eta) / eta;
    
    %% test for overall convergence
    niter = niter+1;
    subspace_discrepancy = 1 - detsim(Lprev', L');
    if (subspace_discrepancy < 1e-16 || niter>1000 || (fstar - fstarprev)^2 < 1e-16)
        notConverged = false;
%         subspace_discrepancy
%         (fstar - fstarprev)^2
%         niter
    end
    
end

% set the output variables
Zl = Xl*L;
Zu = Xu*L;
B = [B0;B];
[~, Yu] = max(Zu*B(2:end,:) + B(1,:), [], 2);
end

function f = cost_fun(L, B, B0, X, Xl, Ymask, Xnorm, n, nl, p, k, var_x, alpha)
tmp = (Xl*L)*B + B0;
f1 = (1/Xnorm^2)*((1/var_x)*((norm(X, 'fro')^2-(alpha/(var_x+alpha))*norm(X*L, 'fro')^2)) + n*(p-k)*log(var_x) + n*k*log(var_x+alpha));
f2 = -(2/nl)*sum((tmp - logsumexp(tmp)).*Ymask, 'all');
f =  f1 + f2;
end

function g = Lgrad(L, B, B0, X, Xl, Yl, Xnorm, numClasses, nl, p, k, var_x, gamma)
g = zeros(p,k);
for j = 1:numClasses
    Xj = Xl(Yl==j, :);
    bj = B(:,j);
    bj0 = B0(j);
    [nj, ~] = size(Xj);
    for i = 1:nj
        xi = Xj(i,:)';
        tmp = xi'*L*B + B0;
        weights = exp(tmp - logsumexp(tmp, 2));
        dLdij = -(2/nl)*xi*(bj - sum(B.*weights, 2))';
        g = g + dLdij; % add and repeat for next class
    end
end
g = g - 2*(1/var_x)*(1/Xnorm^2)*(1/gamma)*(X'*(X*L)) + (1/var_x)*(1/Xnorm^2)*(1/gamma^2)*((L*((L'*X')*X))*L + (((X')*X)*L)*L'*L); %add derivative for PCA term
end

%multiplied f2 by 2 to offset 1/2 in pca term.

function stopnow = mystopfun(problem, x, info, last)
stopnow1 = (last >= 3 && info(last-2).cost - info(last).cost < 1e-10);
stopnow2 = info(last).gradnorm <= 1e-10;
stopnow3 = info(last).stepsize <= 1e-10;
stopnow = (stopnow1 && stopnow3) || stopnow2;
end

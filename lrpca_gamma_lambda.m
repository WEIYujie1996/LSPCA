function [ Z, L, B] = lrpca_gamma_lambda(X, Y, lambda, gamma, k, Linit)

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
[n, p] = size(X);

%useful variables
Xnorm = norm(X, 'fro');
Ynorm = norm(Y, 'fro');
numClasses = length(unique(Y));
Ymask = zeros(n,numClasses); Ymask(sub2ind(size(Ymask), (1:n)', Y)) = 1;

% initialize L0 by PCA of X, and B0 by L0
if sum(abs(Linit), 'all') == 0
    Linit = pca(X);
    Linit = Linit(:,1:k);
end

%solve the problem using CG on the grassmann manifold
L = Linit;
Binit = mnrfit(X*L,Ymask, 'interactions', 'on');
Binit = [Binit, zeros(k+1,1)];
B0 = Binit(1,:);
B = Binit(2:end,:);
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
    problem.cost  = @(Ltilde) cost_fun(Ltilde, B, B0, X, Ymask, Xnorm, n, gamma, lambda);
    problem.egrad = @(Ltilde) Lgrad(Ltilde, B, B0, X, Y, Xnorm, numClasses, n, p, k, gamma, lambda);
    options.verbosity = 0;
    %options.minstepsize = 1e-12;
    options.stopfun = @mystopfun;
    [L, fstar, ~, options] = conjugategradient(problem, L, options);
    %[L, fstar, ~, options] = steepestdescent(problem, L, options);

    
    
    %% B step
    B = mnrfit(X*L,Ymask, 'interactions', 'on');
    B = [B, zeros(k+1,1)];
    B0 = B(1,:);
    B = B(2:end,:);
    
    
    %% test for overall convergence
    niter = niter+1;
    subspace_discrepancy = 1 - detsim(Lprev', L');
    if subspace_discrepancy < 1e-16 || niter>1000 || (fstar - fstarprev)^2 < 1e-16
        notConverged = false;
    end
    
end

% set the output variables
Z = X*L;
B = [B0;B];
end

function f = cost_fun(L, B, B0, X, Ymask, Xnorm, n, gamma, lambda)
tmp = (X*L)*B + B0;
% num = exp((X*L)*B + B0);
%denom = sum(exp((X*L)*B + B0), 2);
f1 = lambda*(1/Xnorm^2)*(2/gamma)*norm(gamma*X - (X*L)*L', 'fro')^2;
%f2 = -(1/n)*sum(log(num./denom).*Ymask, 'all');
f2 = -(1/n)*sum((tmp - logsumexp(tmp)).*Ymask, 'all');
f =  f1 + f2;
end

function g = Lgrad(L, B, B0, X, Y, Xnorm, numClasses, n, p, k, gamma, lambda)
g = zeros(p,k);
for j = 1:numClasses
    Xj = X(Y==j, :);
    bj = B(:,j);
    bj0 = B0(j);
    [nj, ~] = size(Xj);
    for i = 1:nj
        xi = Xj(i,:)';
        tmp = xi'*L*B + B0;
        weights = exp(tmp - logsumexp(tmp, 2));
        dLdij = (1/n)*xi*(bj - sum(B.*weights, 2))';
        g = g - (1-gamma)*dLdij; % add and repeat for next class
    end
end
g = g + lambda*(1/Xnorm^2)*(2/gamma)*( 2*L*(L'*(X'*(X*L))) + 2*X'*(X*L*(L'*L)) -4*gamma*X'*(X*L) ); %add derivative for PCA term
end

function stopnow = mystopfun(problem, x, info, last)
stopnow1 = (last >= 3 && info(last-2).cost - info(last).cost < 1e-10);
stopnow2 = info(last).gradnorm <= 1e-10;
stopnow3 = info(last).stepsize <= 1e-10;
stopnow = (stopnow1 && stopnow3) || stopnow2;
end

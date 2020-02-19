function [Zu, Yu, Zl, L, B, K] = klspca_sub_ss(Xu, Xl, Yl, lambda, sigma, k, L0, Kinit, sstol)
% Inputs:
%       X: (n x p) data matrix columns are features rows are
%       observations
%
%       Y: (n x q) Response Variables
%
%       k: desired number of reduced dimensions
%
%       L0: (p x k) initial guess at a subspace
%           -default: pass in L0 = 0 and first k principle
%           components will be used
%
%
% Outputs:
%
%       Z: (n x k) dimension reduced form of X; A = X*L'
%
%       L: (k x p) matrix with rowspan equal to the desired subspace
%
%       B: (k x q) regression coefficients mapping reduced X to Y
%           i.e. Y = X*L'*B
%

%create full matrix
X = [Xl; Xu];
%The kernel procedure is exactly the same as the regular procedure but with
%a kernel matrix in place of X
[nl, p] = size(Xl);
[nu, ~] = size(Xu);
n = nu + nl;
[~, q] = size(Yl);
if sigma == 0 && sum(abs(Kinit),'all') == 0 %to specify using a linear kernel (faster if n < p)
    X = X*X';
    Xl = X(1:nl,:);
    Xu = X(nl+1:end,:);
elseif sum(abs(Kinit),'all') == 0
    X = gaussian_kernel(X, X, sigma);
    Xl = X(1:nl,:);
    Xu = X(nl+1:end,:);
else
    X = Kinit;
    Xl = X(1:nl,:);
    Xu = X(nl+1:end,:);
end

%norms
Xnorm = norm(X, 'fro');
Ynorm = norm(Yl, 'fro');

%anonymous function for calculating Bi from Li and X
calc_B = @(X, L) (X*L) \ Yl;

% initialize L0 by PCA of X, and B0 by L0
if (sum(sum(L0 ~=0)) == 0)
    L = pca(X,'NumComponents',k);
else
    L = L0;
end


%solve the problem using manopt on the grassmann manifold
% set up the optimization subproblem in manopt
warning('off', 'manopt:getHessian:approx')
warning('off', 'manopt:getgradient:approx')
manifold = grassmannfactory(p, k, 1);
%manifold = stiefelfactory(p, k);
problem.M = manifold;
problem.cost  = @(L) (1/Xnorm^2)*lambda*norm(X - X*L*L', 'fro')^2 + (1-lambda)*(1/Ynorm^2)*norm(Yl - (Xl*L)*((Xl*L)\Yl), 'fro')^2;
problem.egrad = @(L) (-2*(1/Xnorm^2)*lambda*((L'*X')*X) - 2*(1-lambda)*(1/Ynorm^2)*((Xl*L)\Yl)*((Yl'*(eye(n)-(X*L)*pinv(X*L))*X)))';
options.verbosity = 0;
options.stopfun = @mystopfun;
% solve the subproblem for a number of iterations over the steifel
% manifold
%[Lopt, optcost, info, options] = barzilaiborwein(problem, L0, options);
[L, ~, ~, ~] = conjugategradient(problem, L, options);    

% calculate reduced dimension data
B = calc_B(Xl, L);
Zl = Xl*L;
Zu = Xu*L;
Yu = Zu*B;
K = X;
end

function stopnow = mystopfun(problem, x, info, last)
stopnow1 = (last >= 10 && info(last-9).cost - info(last).cost < 1e-8);
stopnow2 = info(last).gradnorm <= 1e-8;
stopnow3 = info(last).stepsize <= 1e-15;
stopnow = (stopnow1 && stopnow3) || stopnow2;
end


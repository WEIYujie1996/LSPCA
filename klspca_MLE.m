function [Z, L, B, K] = klspca_MLE(X, Y, sigma, k, L0, Kinit, sstol)
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
% Outputs:
%
%       Z: (n x k) dimension reduced form of X; A = X*L'
%
%       L: (k x p) matrix with rowspan equal to the desired subspace
%
%       B: (k x q) regression coefficients mapping reduced X to Y
%           i.e. Y = X*L'*B


%create kernel matrix
if sigma == 0 && sum(abs(Kinit),'all') == 0 %to specify using a linear kernel (faster if n < p)
    X = X*X';
elseif sum(abs(Kinit),'all') == 0
    X = gaussian_kernel(X, X, sigma);
else
    X = Kinit;
end

%store dimensions:
[n, p] = size(X);
[~, q] = size(Y);

%anonymous function for calculating Bi from Li and X
calc_B = @(X, L) (X*L) \ Y;

% initialize L0 by PCA of X, and B0 by L0
if (sum(sum(L0 ~=0)) == 0)
    L = pca(X,'NumComponents',k);
else
    L = L0;
end
% initialize the other optimization variables
B = calc_B(X, L);
var_x = (n*(p-k))^-1 * ((norm(X, 'fro')^2-norm(X*L, 'fro')^2));
alpha = max((n*k)^-1 * norm(X*L, 'fro')^2 - var_x, 0);
var_y = (n*q)^-1 * norm(Y - X*(L*B), 'fro')^2;

niter = 0;
notConverged = true;
fstar = inf;
while notConverged
    
    % Update old vars
    Lprev = L;
    fstarprev = fstar;
    
    % set up the optimization subproblem in manopt
    warning('off', 'manopt:getHessian:approx')
    warning('off', 'manopt:getgradient:approx')
    manifold = grassmannfactory(p, k, 1);
    problem.M = manifold;
    problem.cost  = @(L) 0.5*((1/var_y)*norm(Y - X*(L*B), 'fro')^2 + (1/var_x)*((norm(X, 'fro')^2-(alpha/(var_x+alpha))*norm(X*L, 'fro')^2)) + (p-k)*log(var_x) + k*log(var_x+alpha) + q*log(var_y));
    problem.egrad = @(L) (1/var_y)*(X'*(Y-X*(L*B)))*B' - (alpha/(var_x+alpha))*(X'*(X*L));
    options.verbosity = 0;
    options.stopfun = @mystopfun;
    
    % solve the subproblem for a number of iterations over the steifel
    % manifold
    %[Lopt, optcost, info, options] = barzilaiborwein(problem, Ltr, options);
    [L, fstar, ~, options] = conjugategradient(problem, L, options);
    
    %update B
    B = calc_B(X, L);

    %update var_x
    var_x = (n*(p-k))^-1 * ((norm(X, 'fro')^2-norm(X*L, 'fro')^2));
    
    %update alpha
    alpha = max((n*k)^-1 * norm(X*L, 'fro')^2 - var_x, 0);
    
    %update var_y
    var_y = (n*q)^-1 * norm(Y - X*(L*B), 'fro')^2;
    
    %% test for overall convergence
    niter = niter+1;
    subspace_discrepancy = 1 - detsim(Lprev', L');
    if subspace_discrepancy < sstol || niter>500 || (fstar - fstarprev)^2 < sstol
        notConverged = false;
    end
    
end
% set the output variables
Z = X*L;
K = X;
end

function stopnow = mystopfun(problem, x, info, last)
stopnow1 = (last >= 3 && info(last-2).cost - info(last).cost < 1e-6);
stopnow2 = info(last).gradnorm <= 1e-6;
stopnow3 = info(last).stepsize <= 1e-8;
stopnow = (stopnow1 && stopnow3) || stopnow2;
end
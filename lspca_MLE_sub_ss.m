function [Zu, Yu, Zl, L, B] = lspca_MLE_sub_ss(Xu, Xl, Yl, k, L0, sstol)
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

%store dimensions:
[nl, p] = size(Xl);
[nu, p] = size(Xu);
n = nu + nl;
[~, q] = size(Yl);
X = [Xl; Xu];

%norms
Xnorm = norm(X, 'fro');
Ynorm = norm(Yl, 'fro');

%anonymous function for calculating Bi from Li and X
calc_B = @(X, L) (Xl*L) \ Yl;

% initialize L0 by PCA of X, and B0 by L0
if (sum(sum(L0 ~=0)) == 0)
    L = pca(X,'NumComponents',k);
else
    L = L0;
end
% initialize the other optimization variables
B = calc_B(Xl, L);
var_x = (n*(p-k))^-1 * ((norm(X, 'fro')^2-norm(X*L, 'fro')^2));
alpha = max((n*k)^-1 * norm(X*L, 'fro')^2 - var_x, 0);
eta = sqrt(var_x + alpha) - sqrt(var_x);
gamma = (var_x + eta) / eta;
var_y = (n*q)^-1 * norm(Yl - Xl*(L*B), 'fro')^2;

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
    problem.cost  = @(L) 0.5*( (1/Ynorm^2)*((1/var_y)*norm(Yl - Xl*(L*(Xl*L \ Yl)), 'fro')^2 + + n*q*log(var_y)) + (1/Xnorm^2)*((1/var_x)*((norm(X, 'fro')^2-(alpha/(var_x+alpha))*norm(X*L, 'fro')^2)) + n*(p-k)*log(var_x) + n*k*log(var_x+alpha)));
    problem.egrad = @(L) -(1/var_y)*(1/Ynorm^2)*(Xl'*(Yl-Xl*(L*(Xl*L \ Yl))))*(Xl*L \ Yl)' - 2*(1/var_x)*(1/Xnorm^2)*(1/gamma)*(X'*(X*L)) + (1/var_x)*(1/Xnorm^2)*(1/gamma^2)*((L*((L'*X')*X))*L + (((X')*X)*L)*L'*L);
    options.verbosity = 0;
    options.stopfun = @mystopfun;
    options.maxiter = 2000;
    
    % solve the subproblem for a number of iterations over the steifel
    % manifold
    %[L, fstar, info, ~] = barzilaiborwein(problem, L, options);
    [L, fstar, info, ~] = conjugategradient(problem, L, options);
    %[L, fstar, info, ~] = trustregions(problem, L, options);
%     info(end).iter;
%     info(end).gradnorm;
%     fstar;
    
    %update B
    B = calc_B(Xl, L);

    %update var_x
    if alpha>0
        var_x = (n*(p-k))^-1 * ((norm(X, 'fro')^2-norm(X*L, 'fro')^2));
    else
        var_x = (n*p)^-1 * norm(X, 'fro')^2;
    end
    
    %update alpha
    alpha = max((n*k)^-1 * norm(X*L, 'fro')^2 - var_x, 0);
    eta = sqrt(var_x + alpha) - sqrt(var_x);
    gamma = (var_x + eta) / eta;
    
    %update var_y
    var_y = (nl*q)^-1 * norm(Yl - Xl*(L*B), 'fro')^2;
    
    %% test for overall convergence
    niter = niter+1;
    subspace_discrepancy = 1 - detsim(Lprev', L');
    if subspace_discrepancy < sstol || niter>500 || (fstar - fstarprev)^2 < sstol
        notConverged = false;
    end
    
end
% calculate reduced dimension data
B = calc_B(Xl, L);
Zl = Xl*L;
Zu = Xu*L;
Yu = Zu*B;
end

function stopnow = mystopfun(problem, x, info, last)
stopnow1 = (last >= 10 && info(last-9).cost - info(last).cost < 1e-8);
stopnow2 = info(last).gradnorm <= 1e-8;
stopnow3 = info(last).stepsize <= 1e-15;
stopnow = (stopnow1 && stopnow3) || stopnow2;
end


function [Z, L, B, K] = ilspca_gamma(X, Y, lambda, gamma, sigma, k, Kinit)


% NOTE: we are using the convention that the the data points lie in the
% rows of X, therefore our convention for Stiefel manifold is
% L*L'= I, while the definition in manopt is L'*L = I, so we must
% solve the problem in therms of L'

% Inputs:
%       X: (n x p) data matrix columns are features rows are
%       observations
%
%       Y: (n x q) Response Variables
%
%       lambda: tuning parameter (higher lambda -> more empasis on
%       capturing response varable relationship)
%
%       k: desired number of reduced dimensions
%
%       L0: initial guess at a subspace
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
%       A: (n x k) dimension reduced form of X; A = X*L'
%
%       L: (k x p) matrix with rowspan equal to the desired subspace
%
%       B: (k x q) regression coefficients mapping reduced X to Y
%           i.e. Y = X*L'*B
%
%       PCAvariationCaputred: total variation captured by the PCA
%       term of the objective function on a scale of [0, 1]
%
%       LSvariationCaputred: total variation captured by the Least
%       Squares term of the objective function on a scale of [0, 1]

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

%deflation matrix
Xd = X;

%norms
Xnorm = norm(X, 'fro');
Ynorm = norm(Y, 'fro');

%anonymous function for calculating Bi from Li and X
calc_B = @(X, Li) (X*Li) \ Y;


Lopt = zeros(p, k);
%solve the problem using manopt on the stiefel manifold
for i = 1:k
%    size(Xd)
    L = pca(Xd, 'Numcomponents', 1);
    B = calc_B(Xd, L);
    
    % set up the optimization subproblem in manopt
    warning('off', 'manopt:getHessian:approx')
    warning('off', 'manopt:getgradient:approx')
    manifold = grassmannfactory(p, 1, 1);
    %manifold = stiefelfactory(p, k);
    problem.M = manifold;
%    size(L)
    size(B)
    problem.cost  = @(L) (1/Xnorm^2)*lambda*norm(gamma*Xd - Xd*L*L', 'fro')^2 + (1/Ynorm^2)*gamma*norm(Y - (Xd*L)*B, 'fro')^2;
%     problem.egrad = @(Ltr) (2*(1/Xnorm^2)*lambda*(1-2*gamma)*((Ltr'*Xd')*Xd) - 2*(1/Ynorm^2)*gamma*(pinv(Xd*Ltr)*Y)*((Y'*(eye(n)-(Xd*Ltr)*pinv(Xd*Ltr))*Xd)))';
    problem.egrad = @(L) (2*(1/Xnorm^2)*lambda*(1-2*gamma)*((L'*Xd')*Xd) - 2*(1/Ynorm^2)*gamma*((Y-Xd*(L*B))*B')*L')';
    options.verbosity = 0;
    options.stopfun = @mystopfun;
    
    % solve the subproblem for a number of iterations over the steifel
    % manifold
    %[Lopt, optcost, info, options] = barzilaiborwein(problem, Ltr, options);
    [L, optcost, info, options] = conjugategradient(problem, L, options);
    
    %calculate B
    B = calc_B(Xd, L);
    Lopt(:,i) = L;
    Xd = Xd - (Xd*L)*L';
    Xnorm = norm(Xd, 'fro');
end
% set the output variables
L = Lopt;
Z = X*L;
B = calc_B(X, L);
K = X;
end

function stopnow = mystopfun(problem, x, info, last)
stopnow1 = (last >= 3 && info(last-2).cost - info(last).cost < 1e-6);
stopnow2 = info(last).gradnorm <= 1e-6;
stopnow3 = info(last).stepsize <= 1e-8;
stopnow = (stopnow1 && stopnow3) || stopnow2;
end


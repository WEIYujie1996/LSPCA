function [Zu, Yu, Zl, L, B, K] = klspca_gamma_ss(Xu, Xl, Yl, lambda, gamma, sigma, k, L0, K0)


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


%create full matrix
X = [Xl; Xu];
%The kernel procedure is exactly the same as the regular procedure but with
%a kernel matrix in place of X
[nl, p] = size(Xl);
[nu, ~] = size(Xu);
if sigma == 0 && Kinit==0 %to specify using a linear kernel (faster if n < p)
    X = X*X';
    Xl = X(1:nl,:);
    Xu = X(nl+1:end,:);
elseif K0 == 0
    X = gaussian_kernel(X, X, sigma);
    Xl = X(1:nl,:);
    Xu = X(nl+1:end,:);
else
    X = K0;
    Xl = X(1:nl,:);
    Xu = X(nl+1:end,:);
end
% deflation matrix
Xd = X;
Xld = Xl;
%useful variables
Xnorm = norm(X, 'fro');
Ynorm = norm(Yl, 'fro');
%store dimensions:
[nl, p] = size(Xl); %reset dimensions to match kernel problem
[nu, ~] = size(Xu);
n = nu + nl;
nratio = nl/n;
[~, q] = size(Yl);

%anonymous function for calculating Bi from Li and X
calc_B = @(X, Li) (X*Li') \ Yl;

% initialize L0 by PCA of X, and B0 by L0
if (sum(sum(L0 ~=0)) == 0 || lambda == 0)
    L0 = pca(X);
    L0 = L0(:, 1:k); % see the note below about convention for orientation of L
else
    L0 = L0';
end
% initialize the other optimization variables

%solve the problem using manopt on the grassmann manifold
% set up the optimization subproblem in manopt
warning('off', 'manopt:getHessian:approx')
warning('off', 'manopt:getgradient:approx')
manifold = grassmannfactory(p, k, 1);
%manifold = stiefelfactory(p, k);
problem.M = manifold;
problem.cost  = @(Ltr) (1/Xnorm^2)*lambda*norm(gamma*X - X*Ltr*Ltr', 'fro')^2 + (1/Ynorm^2)*gamma*norm(Yl - (Xl*Ltr)*(pinv(Xl*Ltr)*Yl), 'fro')^2;
problem.egrad = @(Ltr) (2*(1/Xnorm^2)*lambda*(1-2*gamma)*((Ltr'*X')*X) - 2*(1/Ynorm^2)*gamma*(pinv(Xl*Ltr)*Yl)*((Yl'*(eye(nl)-(Xl*Ltr)*pinv(Xl*Ltr))*Xl)))';
options.verbosity = 0;
options.stopfun = @mystopfun;
% solve the subproblem for a number of iterations over the steifel
% manifold
%[Lopt, optcost, info, options] = barzilaiborwein(problem, Ltr, options);
[Lopt, ~, ~, ~] = conjugategradient(problem, L0, options);

% set the output variables
L = Lopt';
B = calc_B(Xl, L);
Zl = Xl*L';
Zu = Xu*L';
Yu = Zu*B;
K = X;
end

function stopnow = mystopfun(problem, x, info, last)
stopnow1 = (last >= 3 && info(last-2).cost - info(last).cost < 1e-6);
stopnow2 = info(last).gradnorm <= 1e-6;
stopnow3 = info(last).stepsize <= 1e-8;
stopnow = (stopnow1 && stopnow3) || stopnow2;
end

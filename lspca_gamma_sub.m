function [Z, L, B] = lspca_gamma_sub(X, Y, lambda, gamma, k, L0)


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
        %       L0: (p x k) initial guess at a subspace
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




%store dimensions:
[n, p] = size(X);
[~, q] = size(Y);

%norms
Xnorm = norm(X, 'fro');
Ynorm = norm(Y, 'fro');

%anonymous function for calculating Bi from Li and X
calc_B = @(X, L) (X*L) \ Y;

% initialize L0 by PCA of X, and B0 by L0
if (sum(sum(L0 ~=0)) == 0 || lambda == 0)
    L0 = pca(X);
    L0 = L0(:, 1:k); % see the note below about convention for orientation of L
end

%solve the problem using manopt on the grassmann manifold
% set up the optimization subproblem in manopt
warning('off', 'manopt:getHessian:approx')
warning('off', 'manopt:getgradient:approx')
manifold = grassmannfactory(p, k, 1);
%manifold = stiefelfactory(p, k);
problem.M = manifold;
problem.cost  = @(L) (1/Xnorm^2)*lambda*norm(X - gamma^-1*X*L*L', 'fro')^2 + (1-lambda)*(1/Ynorm^2)*norm(Y - (X*L)*(pinv(X*L)*Y), 'fro')^2;
problem.egrad = @(L) (2*(1/Xnorm^2)*lambda*(gamma^-2 - 2/gamma)*((L'*X')*X) - 2*(1-lambda)*(1/Ynorm^2)*(pinv(X*L)*Y)*((Y'*(eye(n)-(X*L)*pinv(X*L))*X)))';
options.verbosity = 0;
options.stopfun = @mystopfun;
% solve the subproblem for a number of iterations over the steifel
% manifold
%[Lopt, optcost, info, options] = barzilaiborwein(problem, L0, options);
[L, ~, ~, ~] = conjugategradient(problem, L0, options);

% set the output variables
B = calc_B(X, L);
Z = X*L;

end

function stopnow = mystopfun(problem, x, info, last)
        stopnow1 = (last >= 3 && info(last-2).cost - info(last).cost < 1e-6);
        stopnow2 = info(last).gradnorm <= 1e-6;
        stopnow3 = info(last).stepsize <= 1e-8;
        stopnow = (stopnow1 && stopnow3) || stopnow2;
end


function [Mrrr_ridge, Lrrr_ridge] = rrr_ridge(X,Y,k,lambda)
%reduced rank ridge regression as defined in:
%   Mukherjee & Zhu 2011,
%   'Reduced Rank Ridge Regression and Its Kernel Extensions'
%
% inputs:
%           X: nxp matrix of independent variables
%           Y: nxq matrix of dependent variables
%           k: rank of RRR estimator
%           lambda: hyperparameter for ridge term
%
% outputs:
%           Mrrr_ridge: ridge RRR soln such that Y ~ X*Mrrr
%           Lrrr: kxp matrix with orthonormal rows spanning dimension
%           reduction subspace for ridge RRR

if lambda==0
    [Mrrr_ridge, Lrrr_ridge] = rrr(X,Y,k);
else
    
    [n,p] = size(X);
    [~,q] = size(Y);
    
    X_star = [X; sqrt(lambda)*eye(p)];
    Y_star = [Y; zeros(p,q)];
    M_star_ridge = X_star \ Y_star; % Ridge regression estimator
    Y_star_ridge = X_star * M_star_ridge;
    [~,~,v] = svds(Y_star_ridge, k);
    Mrrr_ridge = (M_star_ridge*v)*v'; % ridge RRR estimator
    [Lrrr_transpose, ~, ~] = svds(Mrrr_ridge, k);
    Lrrr_ridge = Lrrr_transpose'; %basis for ridge RRR dimension reduction subspace
end
end


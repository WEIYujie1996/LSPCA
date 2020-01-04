function [Mrrr, Lrrr] = rrr(X,Y,k)
%todo: add maximum likelihood estimate option

%reduced rank regression
%
% inputs:
%           X: nxp matrix of independent variables
%           Y: nxq matrix of dependent variables
%           k: rank of RRR estimator
%           ml_est: boolean flag to use maximum likelihood estimate
%
% outputs:
%           Mrrr: RRR soln such that Y ~ X*Mrrr
%           Lrrr: kxp matrix with orthonormal rows spanning dimension
%           reduction subspace for RRR


Mols = X \ Y; %least squares solution
[~,~,v] = svds(X*Mols, k); 
Mrrr = Mols*v*v'; %RRR solution
[Lrrr_transpose, ~, ~] = svds(Mrrr, k);
Lrrr = Lrrr_transpose'; %basis for RRR dimension reduction subspace
end


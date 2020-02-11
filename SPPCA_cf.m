function [Z,L,B,W_x,W_y,var_x,var_y] = SPPCA_cf(X,Y,k)
%Implementation of SPPCA as described in:
%Yu, Shipeng, Kai Yu, Volker Tresp, Hans-Peter Kriegel, and Mingrui Wu. 
%"Supervised probabilistic principal component analysis." In Proceedings 
%of the 12th ACM SIGKDD international conference on Knowledge discovery 
%and data mining, pp. 464-473. ACM, 2006.

%Inputs: X : centered input data, nxp array
%        Y : centered response variables, nxq array
%        k : desired reduced dimension k < p
%        thresh : threshold for stopping the EM algorithm

%rng(0);
[n, p] = size(X);
[~, q] = size(Y);
norm_x = norm(X, 'fro')^2;
norm_y = norm(Y, 'fro')^2;

%form normalized sample covariance matrix for (x,y), using MLE for
%variances based on isotropic Gaussian model
S_x = X'*X;
var_x = sum(diag(S_x))/(n*p);
S_y = Y'*Y;
var_y = sum(diag(S_y))/(n*q);
S_xy = (Y'*X)';
S = [S_x/var_x, S_xy/sqrt(var_x*var_y); S_xy'/sqrt(var_x*var_y), S_y/var_y];

%take truncated SVD
[U,S_k,~] = svds(S,k);
U_x = U(1:p,:);
U_y = U(p+1:end,:);

%form factor loading matrices
W_x = sqrt(var_x)*U_x*sqrt(S_k-eye(k));
W_y = sqrt(var_y)*U_y*sqrt(S_k-eye(k));

%form embedding matrix
L = ((1/sqrt(var_x))*sqrt(diag(1/diag(S_k-eye(k))))*pinv(U_x'*U_x + diag(1/diag(S_k-eye(k))))*U_x')';
Z = X*L;
B = pinv(Z)*Y;

end


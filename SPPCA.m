function [Z,L,B,W_x,W_y,var_x,var_y] = SPPCA(X,Y,k,thresh, varargin)
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

%the params we are trying to estimate are W_x \in R^pxk, W_y \in R^qxk,
%var_x \in R, var_y \in R

% create variables for current and previous values of estimated parameters
% and sufficient statistics at each iteration
W_x_prev = zeros(p,k);
W_y_prev = zeros(q,k);
var_x_prev = 0;
var_y_prev = 0;

if length(varargin) >0
W_x = varargin{1};
W_y = varargin{2};
else
W_x = randn(p,k);
W_y = randn(q,k);
end
var_x = 1;
var_y = 1;

z_is = zeros(k,n);
zzt_is = zeros(k,k,n);

iter = 0;
while (norm(abs(W_x - W_x_prev), 'fro')^2 + norm(abs(W_y - W_y_prev), 'fro')^2 > thresh)
    
    % update A matrix
    A = W_x'*W_x/var_x + W_y'*W_y/var_y + eye(k);
    A_inv = inv(A);

    % Expectation step
    z_is = A_inv*( W_x'*X'/var_x + W_y'*Y'/var_y);
    for jj = 1:n
       zzt_is(:,:,jj) =  A_inv + z_is(:,jj)*z_is(:,jj)';
    end
    
    % Maximization step
    W_x_prev = W_x;
    W_y_prev = W_y;
    var_x_prev = var_x;
    var_y_prev = var_y;
    
    C = sum(zzt_is, 3); C_inv = inv(C);
    W_x = X'*z_is'*C_inv;
    W_y = Y'*z_is'*C_inv;
    var_x = (1/(p*n))*(norm_x + trace(W_x'*W_x*C) - 2*trace((X*W_x)*z_is));
    var_y = (1/(q*n))*(norm_y + trace(W_y'*W_y*C) - 2*trace((Y*W_y)*z_is));
    
    %debug stuff
    %iter = iter+1
    %dif_wx = norm( abs(W_x - W_x_prev), 'fro')   
    %dif_wy =  norm(abs(W_y - W_y_prev), 'fro')
    %trace((X*W_x)*z_is)
    %trace((Y*W_y)*z_is)

end

Z = z_is';
L = inv(W_x'*W_x + var_x*eye(k))*W_x';

%L = L ./ vecnorm(L,2,2);
B = pinv(Z)*Y;

end


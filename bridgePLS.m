function [Z, L, beta] = bridgePLS(X,Y,alpha, k)

H = alpha*X'*X + (1-alpha)*X'*Y*Y'*X;
[L, ~, ~] = svds(H, k);
L = L';
size(L)

Z = X*L';
beta = Z \ Y;

end


function [Z U] = SPCA(X, Y, k)
%Implement SPCA as in Barshan (2011)
    [p,n] = size(X);
    [q,~] = size(Y);
    H = eye(n)-(1/n)*(ones(n,n));
    Q = (X*(H*Y'))*((Y*H)*X');
    %Q = (X*Y')*(Y*X');
    [V,D] = svds(Q, k);
    %U = real(V(:, 1:k));
    %U = U ./ vecnorm(U,2,1);
    U = V';
    Z = (U*X)';
end


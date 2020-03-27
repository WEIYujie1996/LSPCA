function K = gaussian_kernel(X1, X2, sigma)
sX1 = sum(X1.^2,2);
sX2 = sum(X2.^2,2);
K = exp((bsxfun(@minus,bsxfun(@minus,2*X1*X2',sX1),sX2'))/(2*sigma^2));
K = K - mean(K);
K = (K' - mean(K'))'; %centered kernel matrix
end


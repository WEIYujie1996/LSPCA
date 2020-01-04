function [K, sigma, s] = gpHeuristic(X, k, sigma, provar)

K = gaussian_kernel(X,X,sigma);
[~, ~, ~, ~, var] = pca(K,'NumComponents', k);
s = sum(var(1:k))/100;
ss = [s];
sigmas = [sigma];
while (s < provar-0.01 || s > provar+0.01)
    if s < provar
        sigma = 1.1*sigma;
    else
        sigma = 0.9*sigma;
    end
    K = gaussian_kernel(X,X,sigma);
    [~, ~, ~, ~, var] = pca(K,'NumComponents', k);
    s = sum(var(1:k))/100;
    if abs(sigmas(end)-sigma)< 1e-3 || abs(ss(end)-s)< 1e-3
        sprintf(strcat('proportion variance achieved was s=', num2str(s)))
        break
    end
    sigmas(end+1) = sigma;
    ss(end+1) = s;
end
end


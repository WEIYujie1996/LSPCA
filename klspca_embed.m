function K = klspca_embed(Xtest, Xtrain, Lnormalized, sigma)
    K = gaussian_kernel(Xtest, Xtrain, sigma)*Lnormalized;
end
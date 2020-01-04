function Ztest = klspca_embed(Xtest, Xtrain, Lnormalized, sigma)
    Ztest = gaussian_kernel(Xtest, Xtrain, sigma)*Lnormalized';
end
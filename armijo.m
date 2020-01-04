function [newL] = armijo(func, L, Lgrad, alpha0, maxiter)
    fval = func(L);
    alpha = alpha0;
    gnorm = norm(Lgrad, 'fro')^2;
    newL = L - alpha*Lgrad;
    newval = func(newL);
    niter = 0;
    while fval - newval < alpha*0.99*gnorm
        alpha = alpha*0.5;
        newL = L - alpha*Lgrad;
        newval = func(newL);
        niter = niter + 1;
        if niter>maxiter
            break
        end
    end
end


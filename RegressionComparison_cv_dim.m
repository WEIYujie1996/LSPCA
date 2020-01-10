
%% setup and load data
%rng(0);
ks = 2:10;
load(strcat(dataset, '.mat'));
[n, p] = size(X);
[~, q] = size(Y);


% create same splits to use every time and center
kfold = 10;
cvidx = crossvalind('kfold',n,kfold);
for l = 1:kfold
    Xtrain = X(cvidx~=l, :); %lth training set
    Ytrain = Y(cvidx~=l, :);
    Xtest = X(cvidx==l, :); %lth testing set
    Ytest = Y(cvidx==l, :);
    [Xtrain,Xtest,Ytrain,Ytest] = center_data(Xtrain,Xtest,Ytrain,Ytest,'regression');
    Xtrains{l} = Xtrain; %lth centered training set
    Ytrains{l} = Ytrain;
    Xtests{l} = Xtest; %lth centered testing set
    Ytests{l} = Ytest;
end



for t = 1:length(ks) %dimensionality of reduced data
    k = ks(t)
    
    %% PCA
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xtrain = Xtrains{l};
        Ytrain = Ytrains{l};
        [ntrain, ~] = size(Ytrain);
        Xtest = Xtests{l};
        Ytest = Ytests{l};
        [ntest, ~] = size(Ytest);
        
        tic
        [Lpca, Zpca] = pca(Xtrain, 'NumComponents', k);
        Zpcas{l,t} = Zpca;
        Lpca = Lpca';
        % compute embedding for test data
        pcaXtest = Xtest*Lpca';
        pcaXtests{l,t} = pcaXtest;
        PCAtimes(l,t,t) = toc;
        % compute error
        B = Zpca \ Ytrain;
        PCRYtrain = Zpca*B;
        PCRYtest = pcaXtest*B;
        mse = norm(PCRYtest - Ytest, 'fro')^2 /ntest;
        PCArates(l,t) = mse ;
        PCArates_train(l,t) = norm(PCRYtrain - Ytrain, 'fro')^2 / ntrain;
        PCAvar(l,t) = norm(Xtest*Lpca', 'fro') / norm(Xtest, 'fro');
        PCAvar_train(l,t) = norm(Xtrain*Lpca', 'fro') / norm(Xtrain, 'fro');
    end
    
    %% PLS
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xtrain = Xtrains{l};
        Ytrain = Ytrains{l};
        [ntrain, ~] = size(Ytrain);
        Xtest = Xtests{l};
        Ytest = Ytests{l};
        [ntest, ~] = size(Ytest);
        %solve for basis
        [Xloadings,Yloadings,Xscores,Yscores,betaPLS,pctVar,~,stats] = plsregress(Xtrain,Ytrain,k);
        Lpls = orth(stats.W)';
        Lpls = Lpls(1:k, :);
        % predict
        PLSYtest = [ones(ntest,1) Xtest]*betaPLS;
        PLSYtrain = [ones(ntrain,1) Xtrain]*betaPLS;
        % compute error
        mse = norm(PLSYtest - Ytest, 'fro')^2;
        PLSrates(l,t) = mse / ntest;
        PLSrates_train(l,t) = norm(PLSYtrain - Ytrain, 'fro')^2 / ntrain;
        PLSvar(l,t) = norm(Xtest*Lpls', 'fro') / norm(Xtrain, 'fro');
        PLSvar_train(l,t) = sum(pctVar(1,:));
    end
    
    
    %% LSPCA
    
    %Lambda = 5;
    Lambdas = linspace(1,5,5);
    %Gammas = logspace(log10(3), log10(0.5), 51);
    Gammas = [linspace(1, 0.7, 10), logspace(log10(0.7),log10(0.5058), 30)];
    
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xtrain = Xtrains{l};
        Ytrain = Ytrains{l};
        [ntrain, ~] = size(Ytrain);
        Xtest = Xtests{l};
        Ytest = Ytests{l};
        [ntest, ~] = size(Ytest);
        
        %solve
        for jj = 1:length(Lambdas)
            Lambda = Lambdas(jj)
            for ii = 1:length(Gammas)
                Gamma = Gammas(ii);
                if ii == 1
                    [Zlspca, Llspca, B] = lspca_gamma_sub(Xtrain, Ytrain, Lambda, Gamma, k, 0);
                    %[Zlspca, Llspca, B] = ilspca_gamma(Xtrain, Ytrain, Lambda, Gamma, k);

                    
                else
                    [Zlspca, Llspca, B] = lspca_gamma_sub(Xtrain, Ytrain, Lambda, Gamma, k, Llspca);
                    %[Zlspca, Llspca, B] = ilspca_gamma(Xtrain, Ytrain, Lambda, Gamma, k);
                end
                %Llspca = Llspca';
                Ls{l,t,ii, jj} = Llspca;
                %predict
                LSPCAXtest = Xtest*Llspca';
                LSPCAXtrain = Xtrain*Llspca';
                LSPCAYtest = LSPCAXtest*B;
                LSPCAYtrain = LSPCAXtrain*B;
                lspca_mbd_test{l,t,ii,jj} =LSPCAXtest;
                lspca_mbd_train{l,t,ii,jj} = Zlspca;
                mse = norm(LSPCAYtest - Ytest, 'fro')^2 / ntest;
                % compute error
                train_err = norm(Ytrain - LSPCAYtrain, 'fro')^2 / ntrain;
                LSPCArates(l,t, ii, jj) = mse ;
                LSPCArates_train(l,t, ii, jj) = train_err;
                LSPCAvar(l,t, ii, jj) = norm(LSPCAXtest, 'fro') / norm(Xtest, 'fro');
                LSPCAvar_train(l,t, ii, jj) = norm(Zlspca, 'fro') / norm(Xtrain, 'fro');
            end
        end
    end
    
    %% kLSPCA
    %Lambda = 5;
    Lambdas = linspace(1,5,5);
    %Gammas = logspace(log10(3), log10(0.5), 51);
    Gammas = [linspace(1, 0.7, 10), logspace(log10(0.7),log10(0.5058), 30)];
    
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xtrain = Xtrains{l};
        Ytrain = Ytrains{l};
        [ntrain, ~] = size(Ytrain);
        Xtest = Xtests{l};
        Ytest = Ytests{l};
        [ntest, ~] = size(Ytest);
        
        for kk = 1:length(sigmas)
            sigma = sigmas(kk)
            for jj = 1:length(Lambdas)
                Lambda = Lambdas(jj);
                for ii = 1:length(Gammas)
                    Gamma = Gammas(ii);
                    if ii == 1
                        %[ Zklspca, Lorth, B, Klspca] = klspca_gamma(Xtrain, Ytrain, Lambda, Gamma, sigma, k, 0);
                        [ Zklspca, Lorth, B, Klspca] = klspca_gamma_sub(Xtrain, Ytrain, Lambda, Gamma, sigma, k, 0, 0);
                    else
                        %[ Zklspca, Lorth, B, Klspca] = klspca_gamma(Xtrain, Ytrain, Lambda, Gamma, sigma, k, Klspca);
                        [ Zklspca, Lorth, B, Klspca] = klspca_gamma_sub(Xtrain, Ytrain, Lambda, Gamma, sigma, k, Lorth, Klspca);
                    end
                    %Lorth = Lorth';
                    embedFunc = @(data) klspca_embed(data, Xtrain, Lorth, sigma);
                    kLs{l,t,ii,jj,kk} = Lorth;
                    kLSPCAXtest = embedFunc(Xtest);
                    klspca_mbd_test{l,t,ii,jj,kk} = kLSPCAXtest;
                    kLSPCAXtrain = Zklspca;
                    klspca_mbd_train{l,t,ii,jj,kk} = Zklspca;
                    kLSPCAYtest = kLSPCAXtest*B;
                    kLSPCAYtrain = kLSPCAXtrain*B;
                    mse = norm(kLSPCAYtest - Ytest, 'fro')^2 / ntest;
                    train_err = norm(Ytrain - kLSPCAYtrain, 'fro')^2 / ntrain;
                    kLSPCArates(l,t,ii,jj,kk) = mse ;
                    kLSPCArates_train(l,t,ii,jj,kk) = train_err;
                    kLSPCAvar(l,t,ii,jj,kk) = norm(kLSPCAXtest, 'fro') / norm(gaussian_kernel(Xtest,Xtrain,sigma), 'fro');
                    kLSPCAvar_train(l,t,ii,jj,kk) = norm(Zklspca, 'fro') / norm(Klspca, 'fro');
                end
            end
        end
    end
    
    %% KPCA
    for l = 1:kfold
        test_num = l;
        % get lth fold
        Xtrain = Xtrains{l};
        Ytrain = Ytrains{l};
        [ntrain, ~] = size(Ytrain);
        Xtest = Xtests{l};
        Ytest = Ytests{l};
        [ntest, ~] = size(Ytest);
        for jj = 1:length(sigmas)
            sigma = sigmas(jj);
            K = gaussian_kernel(Xtrain, Xtrain, sigma);
            Ktest = gaussian_kernel(Xtest, Xtrain, sigma);
            tic
            [Lkpca, Zkpca] = pca(K, 'NumComponents', k);
            kPCAtimes(l,t,t,jj) = toc;
            Zkpcas{l,t,jj} = Zkpca;
            Lkpca = Lkpca';
            % compute embedding for test data
            kpcaXtest = Ktest*Lkpca';
            kpcaXtests{l,t,jj} = kpcaXtest;
            % compute error
            B = Zkpca \ Ytrain;
            kpcaYtest = kpcaXtest*B;
            kpcaYtrain = Zkpca*B;
            mse = norm(kpcaYtest - Ytest, 'fro')^2 / ntest;
            train_err = norm(Ytrain - kpcaYtrain, 'fro')^2 / ntrain;
            kPCArates(l,t,jj) = mse;
            kPCArates_train(l,t,jj) = train_err;
            kPCAvar(l,t,jj) = norm(kpcaXtest, 'fro') / norm(Ktest, 'fro');
            kPCAvar_train(l,t,jj) = norm(Zkpca, 'fro') / norm(K, 'fro');
        end
    end
    
    %% R4
    
    if ~strcmp(dataset, 'DLBCL')
        params = logspace(-1, 6, 101);
        params(1) = 0; %edge case is just RRR
        ridge_solns = [];
        for l = 1:kfold
            test_num = l;
            % get lth fold
            Xtrain = Xtrains{l};
            Ytrain = Ytrains{l};
            [ntrain, ~] = size(Ytrain);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            for pp = 1:length(params)
                param = params(pp);
                [Mrrr, Lrrr] = rrr_ridge(Xtrain, Ytrain, k, param);
                ridge_solns(:,:,pp) = Lrrr;
            end
            ridge_rrr_solns{l,t} = ridge_solns;
            Ls_rrr = ridge_rrr_solns{l,t};
            [~, ~, numsols] = size(Ls_rrr);
            for tt = 1:numsols
                b = (Xtrain*Ls_rrr(:,:,tt)') \ Ytrain;
                ridge_rrr_err = norm(Ytest - Xtest*Ls_rrr(:,:,tt)'*b, 'fro')^2 / ntest;
                ridge_rrr_err_train = norm(Ytrain - Xtrain*Ls_rrr(:,:,tt)'*b, 'fro')^2 / ntrain;
                ridge_rrr_var = norm(Xtest*Ls_rrr(:,:,tt)', 'fro') / norm(Xtest, 'fro');
                ridge_rrr_var_train = norm(Xtrain*Ls_rrr(:,:,tt)', 'fro') / norm(Xtrain, 'fro');
                ridge_rrr_rates(l,t, tt) = ridge_rrr_err;
                ridge_rrr_rates_train(l,t,tt) = ridge_rrr_err_train;
                ridge_rrrvars(l,t,tt) = ridge_rrr_var;
                ridge_rrrvars_train(l,t, tt) = ridge_rrr_var_train;
                Lridge_rrrs{l, t, tt} = Ls_rrr(:,:,tt);
            end
        end
    else
        ridge_rrr_rates(l,t, 1) = nan;
        ridge_rrr_rates_train(l,t,1) = nan;
        ridge_rrrvars(l,t,1) = nan;
        ridge_rrrvars_train(l,t,1) = nan;
    end
    
    %% ISPCA
    for l = 1:kfold
        test_num = l;
        % get lth fold
        Xtrain = Xtrains{l};
        Ytrain = Ytrains{l};
        [ntrain, ~] = size(Ytrain);
        Xtest = Xtests{l};
        Ytest = Ytests{l};
        [ntest, ~] = size(Ytest);
        %find basis
        tic
        [Zispca, Lispca, B] = ISPCA(Xtrain,Ytrain,k);
        ISPCAtimes(l,t) = toc;
        % predict
        ISPCAXtest = Xtest*Lispca';
        ISPCAYtest = ISPCAXtest*B;
        ISPCAYtrain = Zispca*B;
        % compute error
        mse = norm(ISPCAYtest - Ytest, 'fro')^2 / ntest;
        ISPCArates(l,t) = mse ;
        ISPCArates_train(l,t) = norm(ISPCAYtrain - Ytrain, 'fro')^2 / ntrain;
        ISPCAvar(l,t) = norm(Xtest*Lispca', 'fro') / norm(Xtest, 'fro');
        ISPCAvar_train(l,t) = norm(Xtrain*Lispca', 'fro') / norm(Xtrain, 'fro');
    end
    
    %% SPPCA
    if ~strcmp(dataset, 'DLBCL')
        for l = 1:kfold
            test_num = l;
            % get lth fold
            Xtrain = Xtrains{l};
            Ytrain = Ytrains{l};
            [ntrain, ~] = size(Ytrain);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            tic
            SPPCAXtest = {};
            SPPCAYtest = {};
            SPPCAYtrain = {};
            sppca_err = [];
            % solve
            for count = 1:10 %do 10 initializations and take the best b/c ends up in bad local minima a lot
                [Zsppca, Lsppca, B, W_x, W_y, var_x, var_y] = SPPCA(Xtrain,Ytrain,k,exp(-10), randn(p,k), randn(q,k));
                Zsppcas{count} = Zsppca;
                Lsppcas{count} = Lsppca;
                SPPCAXtest{count} = Xtest*Lsppca';
                SPPCAYtest{count} = SPPCAXtest{count}*B;
                SPPCAYtrain{count} = Zsppca*B;
                sppca_err(count) =  norm(SPPCAYtrain{count} - Ytrain, 'fro')^2;
            end
            [~, loc] = min(sppca_err);
            Zsppca = Zsppcas{loc};
            Lsppca = orth(Lsppcas{loc}')';
            % Predict
            SPPCAXtest = SPPCAXtest{loc};
            SPPCAYtest = SPPCAYtest{loc};
            SPPCAYtrain = SPPCAYtrain{loc};
            SPPCAtimes(l,t) = toc;
            % compute error
            mse = norm(SPPCAYtest - Ytest, 'fro')^2/ ntest;
            SPPCArates(l,t) = mse ;
            SPPCArates_train(l,t) = norm(SPPCAYtrain - Ytrain, 'fro')^2 / ntrain;
            SPPCAvar(l,t) = norm(Xtest*Lsppca', 'fro') / norm(Xtest, 'fro');
            SPPCAvar_train(l,t) = norm(Xtrain*Lsppca', 'fro') / norm(Xtrain, 'fro');
        end
    else
        SPPCArates(l,t) = nan ;
        SPPCArates_train(l,t) = nan;
        SPPCAvar(l,t) = nan;
        SPPCAvar_train(l,t) = nan;
    end
    
    
    %% Barshan
    
    for l = 1:kfold
        test_num = l;
        % get lth fold
        Xtrain = Xtrains{l};
        Ytrain = Ytrains{l};
        [ntrain, ~] = size(Ytrain);
        Xtest = Xtests{l};
        Ytest = Ytests{l};
        [ntest, ~] = size(Ytest);
        
        %learn basis
        [Zspca Lspca] = SPCA(Xtrain', Ytrain', k);
        spcaXtest = Xtest*Lspca';
        % predict
        betaSPCA = Zspca \Ytrain;
        SPCAYtest = spcaXtest * betaSPCA;
        SPCAYtrain = Zspca*betaSPCA;
        %compute error
        mse = norm(SPCAYtest - Ytest, 'fro')^2/ ntest;
        SPCArates(l,t) = mse ;
        SPCArates_train(l,t) = norm(SPCAYtrain - Ytrain, 'fro')^2 / ntrain;
        SPCAvar(l,t) = norm(Xtest*Lspca', 'fro') / norm(Xtest, 'fro');
        SPCAvar_train(l,t) = norm(Xtrain*Lspca', 'fro') / norm(Xtrain, 'fro');
    end
    
    %% Perform Barshan's KSPCA based 2D embedding
    for l = 1:kfold
        test_num = l;
        % get lth fold
        Xtrain = Xtrains{l};
        Ytrain = Ytrains{l};
        [ntrain, ~] = size(Ytrain);
        Xtest = Xtests{l};
        Ytest = Ytests{l};
        [ntest, ~] = size(Ytest);
        
        for jj = 1:length(sigmas)
            sigma = sigmas(jj);
            % calc with best param on full training set
            barshparam.ktype_y = 'linear';
            barshparam.kparam_y = 1;
            barshparam.ktype_x = 'rbf';
            barshparam.kparam_x = sigma;
            [Zkspca, Lkspca] = KSPCA(Xtrain', Ytrain', k, barshparam);
            Zkspca = Zkspca';
            %do regression in learned basis
            betakSPCA = Zkspca \ Ytrain;
            Ktest = gaussian_kernel(Xtest, Xtrain, sigma);
            K = gaussian_kernel(Xtrain, Xtrain, sigma);
            kspcaXtest = Ktest*Lkspca;
            kSPCAYtest = kspcaXtest*betakSPCA;
            kSPCAYtrain = Zkspca*betakSPCA;
            kspca_mbd_test{l,t,jj} = kspcaXtest;
            kspca_mbd_train{l,t,jj} = Zkspca;
            % compute error
            mse = norm(kSPCAYtest - Ytest, 'fro')^2 / ntest;
            kSPCArates(l,t,jj) = mse;
            kSPCArates_train(l,t,jj) = norm(kSPCAYtrain - Ytrain, 'fro')^2  / ntrain;
            kSPCAvar(l,t,jj) = norm(kspcaXtest, 'fro') / norm(Ktest, 'fro');
            kSPCAvar_train(l,t,jj) = norm(Zkspca, 'fro') / norm(K, 'fro');
        end
    end
    
    
    %% Sup SVD
    for l = 1:kfold
        test_num = l;
        % get lth fold
        Xtrain = Xtrains{l};
        Ytrain = Ytrains{l};
        [ntrain, ~] = size(Ytrain);
        Xtest = Xtests{l};
        Ytest = Ytests{l};
        [ntest, ~] = size(Ytest);
        % solve
        [~,V,Zssvd,~,~]=SupPCA(Ytrain,Xtrain,k);
        Lssvd = V';
        % Predict
        Bssvd = Zssvd \ Ytrain;
        SSVDXtest = Xtest*Lssvd';
        SSVDYtest = SSVDXtest*Bssvd;
        SSVDYtrain = Zssvd*Bssvd;
        %compute error
        mse = norm(SSVDYtest - Ytest, 'fro')^2 / ntest;
        SSVDrates(l,t) = mse ;
        SSVDrates_train(l,t) = norm(SSVDYtrain - Ytrain, 'fro')^2 / ntrain;
        SSVDvar(l,t) = norm(SSVDXtest, 'fro') / norm(Xtest, 'fro');
        SSVDvar_train(l,t) = norm(Zssvd, 'fro') / norm(Xtrain, 'fro');
    end
    
    
end

%% save all data
save(strcat(dataset, '_results'))

%% compute avg performance for each k

%means
avgPCA = mean(PCArates);
avgPCA_train = mean(PCArates_train);
avgkPCA = mean(kPCArates);
avgkPCA_train = mean(kPCArates_train);
avgPLS = mean(PLSrates);
avgPLS_train = mean(PLSrates_train);
avgLSPCA = mean(LSPCArates, 1);
avgLSPCA_train = mean(LSPCArates_train, 1);
avgkLSPCA = mean(kLSPCArates, 1);
avgkLSPCA_train = mean(kLSPCArates_train, 1);
avgSPCA = mean(SPCArates);
avgSPCA_train = mean(SPCArates_train);
avgkSPCA = mean(kSPCArates);
avgkSPCA_train = mean(kSPCArates_train);
avgISPCA = mean(ISPCArates);
avgISPCA_train = mean(ISPCArates_train);
avgSPPCA = mean(SPPCArates);
avgSPPCA_train = mean(SPPCArates_train);
avgR4 = mean(ridge_rrr_rates, 1);
avgR4_train = mean(ridge_rrr_rates_train, 1);
avgSSVD = mean(SSVDrates);
avgSSVD_train = mean(SSVDrates_train);

avgPCAvar = mean(PCAvar);
avgkPCAvar = mean(kPCAvar);
avgPLSvar = mean(PLSvar);
avgLSPCAvar = mean(LSPCAvar, 1);
avgkLSPCAvar = mean(kLSPCAvar, 1);
avgSPCAvar = mean(SPCAvar);
avgkSPCAvar = mean(kSPCAvar);
avgISPCAvar = mean(ISPCAvar);
avgSPPCAvar = mean(SPPCAvar);
avgR4var = mean(ridge_rrrvars, 1);
avgSSVDvar = mean(SSVDvar);


avgPCAvar_train = mean(PCAvar_train);
avgkPCAvar_train = mean(kPCAvar_train);
avgPLSvar_train = mean(PLSvar_train);
avgLSPCAvar_train = mean(LSPCAvar_train, 1);
avgkLSPCAvar_train = mean(kLSPCAvar_train, 1);
avgSPCAvar_train = mean(SPCAvar_train);
avgkSPCAvar_train = mean(kSPCAvar_train);
avgISPCAvar_train = mean(ISPCAvar_train);
avgSPPCAvar_train = mean(SPPCAvar_train);
avgR4var_train = mean(ridge_rrrvars_train, 1);
avgSSVDvar_train = mean(SSVDvar_train);


%% Print results over all subspace dimensions
loc = find(avgPCA==min(avgPCA,[],'all'),1,'last');
[~,kloc] = ind2sub(size(avgPCA), loc);
k = ks(kloc);
m = mean(PCArates(:,kloc));
v = mean(PCAvar(:,kloc));
sm = std(PCArates(:,kloc));
sv = std(PCAvar(:,kloc));
sprintf('PCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

loc = find(avgLSPCA==min(avgLSPCA,[],'all'),1,'last');
[~,kloc,gamloc] = ind2sub(size(avgLSPCA), loc);
k = ks(kloc);
m = mean(LSPCArates(:,kloc,gamloc), 1);
v = mean(LSPCAvar(:,kloc,gamloc), 1);
sm = std(LSPCArates(:,kloc,gamloc), 1);
sv = std(LSPCAvar(:,kloc,gamloc), 1);
sprintf('LSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)


loc = find(avgkLSPCA==min(avgkLSPCA,[],'all'),1,'last');
[~,klock,gamlock,siglock] = ind2sub(size(avgkLSPCA), loc);
k = ks(klock);
m = mean(kLSPCArates(:,klock,gamlock,siglock), 1);
v = mean(kLSPCAvar(:,klock,gamlock,siglock), 1);
sm = std(kLSPCArates(:,klock,gamlock,siglock), 1);
sv = std(kLSPCAvar(:,klock,gamlock,siglock), 1);
sprintf('kLSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)


loc = find(avgISPCA==min(avgISPCA,[],'all'),1,'last');
[~,kloc] = ind2sub(size(avgISPCA), loc);
k = ks(kloc);
m = mean(ISPCArates(:,kloc));
v = mean(ISPCAvar(:,kloc));
sm = std(ISPCArates(:,kloc));
sv = std(ISPCAvar(:,kloc));
sprintf('ISPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

loc = find(avgSPPCA==min(avgSPPCA,[],'all'),1,'last');
[~,kloc] = ind2sub(size(avgSPPCA), loc);
k = ks(kloc);
m = mean(SPPCArates(:,kloc));
v = mean(SPPCAvar(:,kloc));
sm = std(SPPCArates(:,kloc));
sv = std(SPPCAvar(:,kloc));
sprintf('SPPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

loc = find(avgSPCA==min(avgSPCA,[],'all'),1,'last');
[~,kloc] = ind2sub(size(avgSPCA), loc);
k = ks(kloc);
m = mean(SPCArates(:,kloc));
v = mean(SPCAvar(:,kloc));
sm = std(SPCArates(:,kloc));
sv = std(SPCAvar(:,kloc));
sprintf('Barshanerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

loc = find(avgkSPCA==min(avgkSPCA,[],'all'),1,'last');
[~,kloc,sigloc] = ind2sub(size(avgkSPCA), loc);
k = ks(kloc);
m = mean(kSPCArates(:,kloc,sigloc));
v = mean(kSPCAvar(:,kloc,sigloc));
sm = std(kSPCArates(:,kloc,sigloc));
sv = std(kSPCAvar(:,kloc,sigloc));
sprintf('kBarshanerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

loc = find(avgSSVD==min(avgSSVD,[],'all'),1,'last');
[~,kloc] = ind2sub(size(avgSSVD), loc);
k = ks(kloc);
m = mean(SSVDrates(:,kloc));
v = mean(SSVDvar(:,kloc));
sm = std(SSVDrates(:,kloc));
sv = std(SSVDvar(:,kloc));
sprintf('SSVD: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

loc = find(avgR4==min(avgR4,[],'all'),1,'last');
[~,kloc,~] = ind2sub(size(avgR4), loc);
k = ks(kloc);
loc = 1; % RRR with parameter value 0
m = mean(ridge_rrr_rates(:,kloc,loc), 1);
v = mean(ridge_rrrvars(:,kloc,loc), 1);
sm = std(ridge_rrr_rates(:,kloc,loc), 1);
sv = std(ridge_rrrvars(:,kloc,loc), 1);
sprintf('RRR: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

loc = find(avgR4==min(avgR4,[],'all'),1,'last');
[~,kloc,locr4] = ind2sub(size(avgR4), loc);
k = ks(kloc);
m = mean(ridge_rrr_rates(:,kloc,locr4), 1);
v = mean(ridge_rrrvars(:,kloc,locr4), 1);
sm = std(ridge_rrr_rates(:,kloc,locr4), 1);
sv = std(ridge_rrrvars(:,kloc,locr4), 1);
sprintf('R4: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)



%% print mean performance with std errors


m = mean(PCArates);
v = mean(PCAvar);
sm = std(PCArates);
sv = std(PCAvar);
sprintf('PCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

[~, gamloc] = min(min(avgLSPCA,[], 3),[], 2);
[~, lamloc] = min(min(avgLSPCA,[], 2),[], 3);
m = mean(LSPCArates(:,:,gamloc,lamloc), 1);
v = mean(LSPCAvar(:,:,gamloc,lamloc), 1);
sm = std(LSPCArates(:,:,gamloc,lamloc), 1);
sv = std(LSPCAvar(:,:,gamloc,lamloc), 1);
sprintf('LSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

[~, gamloc] = min(min(min(avgkLSPCA,[], 4),[], 3), [], 2);
[~, lamlock] = min(min(min(avgkLSPCA,[], 2),[], 4), [], 3);
[~, siglock] = min(min(min(avgkLSPCA,[], 2),[], 3), [], 4);
m = mean(kLSPCArates(:,:,gamloc,lamlock,siglock), 1);
v = mean(kLSPCAvar(:,:,gamloc,lamlock,siglock), 1);
sm = std(kLSPCArates(:,:,gamloc,lamlock,siglock), 1);
sv = std(kLSPCAvar(:,:,gamloc,lamlock,siglock), 1);
sprintf('kLSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(ISPCArates);
v = mean(ISPCAvar);
sm = std(ISPCArates);
sv = std(ISPCAvar);
sprintf('ISPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(SPPCArates);
v = mean(SPPCAvar);
sm = std(SPPCArates);
sv = std(SPPCAvar);
sprintf('SPPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(SPCArates);
v = mean(SPCAvar);
sm = std(SPCArates);
sv = std(SPCAvar);
sprintf('Barshanerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

[~, locb] = min(avgkSPCA,[], 2);
m = mean(kSPCArates(:,:,locb));
v = mean(kSPCAvar(:,:,locb));
sm = std(kSPCArates(:,:,locb));
sv = std(kSPCAvar(:,:,locb));                                                                            
sprintf('kBarshanerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(SSVDrates);
v = mean(SSVDvar);
sm = std(SSVDrates);
sv = std(SSVDvar);
sprintf('SSVD: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

loc = 1; % RRR with parameter value 0
m = mean(ridge_rrr_rates(:,loc), 1);
v = mean(ridge_rrrvars(:,loc), 1);
sm = std(ridge_rrr_rates(:,loc), 1);
sv = std(ridge_rrrvars(:,loc), 1);
sprintf('RRR: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

[~, locr4] = min(avgR4_train);
m = mean(ridge_rrr_rates(:,locr4), 1);
v = mean(ridge_rrrvars(:,locr4), 1);
sm = std(ridge_rrr_rates(:,locr4), 1);
sv = std(ridge_rrrvars(:,locr4), 1);
sprintf('R4: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)



%%% plot error - var tradeoff curves
% 
% 
% for t = 1:length(ks)
%     figure()
%     hold on
%     plot(avgPCAvar_train(t), avgPCA(t), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
%     %plot(avgkPCAvar_train(t), avgkPCA(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgLSPCAvar_train(t,:,lamloc), avgLSPCA(t, :,lamloc), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgkLSPCAvar_train(t,:,lamlock,siglock), avgkLSPCA(t, :,lamlock,siglock), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgISPCAvar_train(t), avgISPCA(t), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPPCAvar_train(t), avgSPPCA(t), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPCAvar_train(t), avgSPCA(t), '+', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgkSPCAvar_train(t,locb), avgkSPCA(t), '>', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgR4var_train(t,2:end), avgR4(t, 2:end), ':', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgR4var_train(t,1), avgR4(t, 1), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgPLSvar_train(t), avgPLS(t), 'h', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSSVDvar_train(t), avgSSVD(t), 'd', 'MarkerSize', 20, 'LineWidth', 2)
%     
%     xlabel('Variation Explained', 'fontsize', 25)
%     %title('Test', 'fontsize', 25)
%     ylabel('MSE', 'fontsize', 25)
%     %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
%     set(gca, 'fontsize', 25)
%     lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'Location', 'best'); lgd.FontSize = 15;
%     %lgd = legend('LSPCA', 'R4', 'PLS', 'SPPCA', 'Barshan', 'SSVD', 'PCA', 'Location', 'southeast'); lgd.FontSize = 15;
%     %ylim([0, 0.12])
%     %set(gca, 'YScale', 'log')
%     xlim([0,1])
% end
% saveas(gcf, strcat(dataset, 'multi_obj_gamma.jpg'))
% 
% 
% %% plot training error - var tradeoff curves
% for t = 1:length(ks)
%     figure()
%     hold on
%     plot(avgPCAvar_train(t), avgPCA_train(t), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
%     %    plot(avgkPCAvar_train(t), avgkPCA_train(t), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgLSPCAvar_train(t,:,lamloc), avgLSPCA_train(t, :,lamloc), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgkLSPCAvar_train(t,:,lamlock,siglock), avgkLSPCA_train(t, :,lamlock,siglock), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgISPCAvar_train(t), avgISPCA_train(t), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPPCAvar_train(t), avgSPPCA_train(t), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPCAvar_train(t), avgSPCA_train(t), '+', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgkSPCAvar_train(t), avgkSPCA_train(t), '>', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgR4var_train(t,2:end), avgR4_train(t, 2:end), ':', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgR4var_train(t,1), avgR4_train(t, 1), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgPLSvar_train(t), avgPLS_train(t), 'h', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSSVDvar_train(t), avgSSVD_train(t), 'd', 'MarkerSize', 20, 'LineWidth', 2)
%     
%     xlabel('Variation Explained', 'fontsize', 25)
%     %title('Train', 'fontsize', 25)
%     ylabel('MSE', 'fontsize', 25)
%     %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
%     set(gca, 'fontsize', 25)
%     lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'Location', 'best'); lgd.FontSize = 15;
%     %ylim([0, 0.12])
%     %set(gca, 'YScale', 'log')
%     xlim([0,1])
% end
% saveas(gcf, strcat(dataset, 'multi_obj_train_gamma.jpg'))
% 
% 
% 




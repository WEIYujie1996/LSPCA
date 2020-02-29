numExps = 10;
for dd = 1:numExps
    %% setup and load data
    load(strcat(dataset, '.mat'));
    [n, p] = size(X);
    [~, q] = size(Y);
%     ks = 2:min(10, p-1);
    ks = 2;
    
    %holdout an independent test set
    proportion = 0.2;
    nhold = floor(n*proportion); % 20%
    idxhold = ~crossvalind('HoldOut',n,proportion);
    Xhold = X(idxhold, :);
    X = X(~idxhold, :);
    Yhold = Y(idxhold, :);
    Y = Y(~idxhold, :);
    n = n - nhold; %this is our new n
    
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
    
    % store the independent test set, and corresponding training set (non holdout data)
    % at the end of the train  and test cell arrays for convenience of
    % implementation
    [Xtrain,Xtest,Ytrain,Ytest] = center_data(X,Xhold,Y,Yhold,'regression');
    Xtrains{kfold+1} = Xtrain; %lth centered training set
    Ytrains{kfold+1} = Ytrain;
    Xtests{kfold+1} = Xtest; %lth centered testing set
    Ytests{kfold+1} = Ytest;
    % this way we will end up evaluating the independent set for every possible
    % model, which will take longer, but this is just easier. At the end, we
    % will choose the best model by looking only at the tests corresponding to
    % l = 1:kfold, and l=kfold+1 will give us the independent test error
    % corresponding to that model
    
    
    for t = 1:length(ks) %dimensionality of reduced data
        k = ks(t)
        
        %% PCA
        for l = 1:kfold+1
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
        for l = 1:kfold+1
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
        
        Lambdas = fliplr(logspace(-3, 0, 51));
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xtrain = Xtrains{l};
            Ytrain = Ytrains{l};
            [ntrain, ~] = size(Ytrain);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            for ii = 1:length(Lambdas)
                Lambda = Lambdas(ii);
                
                if ii == 1
                    [Zlspca, Llspca, B] = lspca_sub(Xtrain, Ytrain, Lambda, k, 0);
                else
                    [Zlspca, Llspca, B] = lspca_sub(Xtrain, Ytrain, Lambda, k, Llspca);
                end
                Ls{l,t,ii} = Llspca;
                %predict
                LSPCAXtest = Xtest*Llspca;
                LSPCAXtrain = Xtrain*Llspca;
                LSPCAYtest = LSPCAXtest*B;
                LSPCAYtrain = LSPCAXtrain*B;
                lspca_mbd_test{l,t,ii} =LSPCAXtest;
                lspca_mbd_train{l,t,ii} = Zlspca;
                mse = norm(LSPCAYtest - Ytest, 'fro')^2 / ntest;
                % compute error
                train_err = norm(Ytrain - LSPCAYtrain, 'fro')^2 / ntrain;
                LSPCArates(l,t,ii) = mse ;
                LSPCArates_train(l,t, ii) = train_err;
                LSPCAvar(l,t, ii) = norm(LSPCAXtest, 'fro') / norm(Xtest, 'fro');
                LSPCAvar_train(l,t, ii) = norm(Zlspca, 'fro') / norm(Xtrain, 'fro');
            end
        end
        
        %% LSPCA (MLE)
        
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xtrain = Xtrains{l};
            Ytrain = Ytrains{l};
            [ntrain, ~] = size(Ytrain);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            %solve
            Linit = orth(randn(p,k));
            %        Linit = 0;
            %         V = pca(Xtrain);
            %        Linit = V(:,1:2);
            %         c = 0.5;
            %         Linit = orth((1-c)*V(:,1:2) + c*randn(p,k));
            [Zlspca, Llspca, B, var_x, var_y, alpha] = lspca_MLE_sub(Xtrain, Ytrain, k, Linit, 1e-8);
            mle_Ls{l,t} = Llspca;
            %predict
            LSPCAXtest = Xtest*Llspca;
            LSPCAYtest = LSPCAXtest*B;
            LSPCAYtrain = Zlspca*B;
            mle_lspca_mbd_test{l,t} =LSPCAXtest;
            mle_lspca_mbd_train{l,t} = Zlspca;
            mse = norm(LSPCAYtest - Ytest, 'fro')^2 / ntest;
            % compute error
            train_err = norm(Ytrain - LSPCAYtrain, 'fro')^2 / ntrain;
            mle_LSPCArates(l,t) = mse ;
            mle_LSPCArates_train(l,t) = train_err;
            mle_LSPCAvar(l,t) = norm(LSPCAXtest, 'fro') / norm(Xtest, 'fro');
            mle_LSPCAvar_train(l,t) = norm(Zlspca, 'fro') / norm(Xtrain, 'fro');
        end
        
        %% kLSPCA
        Lambdas = fliplr(logspace(-3, 0, 51));
        
        for l = 1:kfold+1
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
                for ii = 1:length(Lambdas)
                    Lambda = Lambdas(ii);
                    
                    if ii == 1
                        [ Zklspca, Lorth, B, Klspca] = klspca_sub(Xtrain, Ytrain, Lambda, sigma, k, 0, 0);
                    else
                        [ Zklspca, Lorth, B, Klspca] = klspca_sub(Xtrain, Ytrain, Lambda, sigma, k, Lorth, Klspca);
                    end
                    embedFunc = @(data) klspca_embed(data, Xtrain, Lorth, sigma);
                    kLs{l,t,ii,kk} = Lorth;
                    kLSPCAXtest = embedFunc(Xtest);
                    klspca_mbd_test{l,t,ii,kk} = kLSPCAXtest;
                    kLSPCAXtrain = Zklspca;
                    klspca_mbd_train{l,t,ii,kk} = Zklspca;
                    kLSPCAYtest = kLSPCAXtest*B;
                    kLSPCAYtrain = kLSPCAXtrain*B;
                    mse = norm(kLSPCAYtest - Ytest, 'fro')^2 / ntest;
                    train_err = norm(Ytrain - kLSPCAYtrain, 'fro')^2 / ntrain;
                    kLSPCArates(l,t,ii,kk) = mse ;
                    kLSPCArates_train(l,t,ii,kk) = train_err;
                    kLSPCAvar(l,t,ii,kk) = norm(kLSPCAXtest, 'fro') / norm(gaussian_kernel(Xtest,Xtrain,sigma), 'fro');
                    kLSPCAvar_train(l,t,ii,kk) = norm(Zklspca, 'fro') / norm(Klspca, 'fro');
                end
            end
            
        end
        
        %% kLSPCA (MLE)
        
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xtrain = Xtrains{l};
            Ytrain = Ytrains{l};
            [ntrain, ~] = size(Ytrain);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            Linit = orth(randn(ntrain,k));
            %Linit = 0;
            for ss = 1:length(sigmas)
                sigma = sigmas(ss)
                [Zklspca, Lorth, B, Klspca] = klspca_MLE_sub(Xtrain, Ytrain, sigma, k, Linit, 0, 1e-10);
                embedFunc = @(data) klspca_embed(data, Xtrain, Lorth, sigma);
                mle_kLs{l,t,ss} = Lorth;
                kLSPCAXtest = embedFunc(Xtest);
                mle_klspca_mbd_test{l,t,ss} = kLSPCAXtest;
                mle_klspca_mbd_train{l,t,ss} = Zklspca;
                kLSPCAYtest = kLSPCAXtest*B;
                kLSPCAYtrain = Zklspca*B;
                mse = norm(kLSPCAYtest - Ytest, 'fro')^2 / ntest;
                train_err = norm(Ytrain - kLSPCAYtrain, 'fro')^2 / ntrain;
                mle_kLSPCArates(l,t,ss) = mse ;
                mle_kLSPCArates_train(l,t,ss) = train_err;
                mle_kLSPCAvar(l,t,ss) = norm(kLSPCAXtest, 'fro') / norm(gaussian_kernel(Xtest,Xtrain,sigma), 'fro');
                mle_kLSPCAvar_train(l,t,ss) = norm(Zklspca, 'fro') / norm(Klspca, 'fro');
            end
        end
        
        
        %% KPCA
        for l = 1:kfold+1
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
            for l = 1:kfold+1
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
        for l = 1:kfold+1
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
        for l = 1:kfold+1
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
            for count = 1%:10 %do 10 initializations and take the best b/c ends up in bad local minima a lot
                [Zsppca, Lsppca, B, W_x, W_y, var_x, var_y] = SPPCA(Xtrain,Ytrain,k);
                Zsppcas{count} = Zsppca;
                Lsppcas{count} = Lsppca;
                SPPCAXtest{count} = Xtest*Lsppca;
                SPPCAYtest{count} = SPPCAXtest{count}*B;
                SPPCAYtrain{count} = Zsppca*B;
                sppca_err(count) =  norm(SPPCAYtrain{count} - Ytrain, 'fro')^2;
            end
            [~, loc] = min(sppca_err);
            Zsppca = Zsppcas{loc};
            Lsppca = orth(Lsppcas{loc});
            % Predict
            SPPCAXtest = SPPCAXtest{loc};
            SPPCAYtest = SPPCAYtest{loc};
            SPPCAYtrain = SPPCAYtrain{loc};
            SPPCAtimes(l,t) = toc;
            % compute error
            mse = norm(SPPCAYtest - Ytest, 'fro')^2/ ntest;
            SPPCArates(l,t) = mse ;
            SPPCArates_train(l,t)= norm(SPPCAYtrain - Ytrain, 'fro')^2 / ntrain;
            SPPCAvar(l,t) = norm(Xtest*Lsppca, 'fro') / norm(Xtest, 'fro');
            SPPCAvar_train(l,t) = norm(Xtrain*Lsppca, 'fro') / norm(Xtrain, 'fro');
        end
        
        
        %% Barshan
        
        for l = 1:kfold+1
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
        for l = 1:kfold+1
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
        for l = 1:kfold+1
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
    
    %% compute avg performance for each k
    
    %means
    avgPCA = mean(PCArates(1:end-1,:));
    avgPCA_train = mean(PCArates_train(1:end-1,:));
    avgkPCA = mean(kPCArates(1:end-1,:,:));
    avgkPCA_train = mean(kPCArates_train(1:end-1,:,:));
    avgPLS = mean(PLSrates(1:end-1,:));
    avgPLS_train = mean(PLSrates_train(1:end-1,:));
    avgLSPCA = mean(LSPCArates(1:end-1,:,:), 1);
    avgLSPCA_train = mean(LSPCArates_train(1:end-1,:,:), 1);
    avgkLSPCA = mean(kLSPCArates(1:end-1,:,:,:), 1);
    avgkLSPCA_train = mean(kLSPCArates_train(1:end-1,:,:,:), 1);
    avgLSPCAmle = mean(mle_LSPCArates(1:end-1,:,:), 1);
    avgLSPCAmle_train = mean(mle_LSPCArates_train(1:end-1,:,:), 1);
    avgkLSPCAmle = mean(mle_kLSPCArates(1:end-1,:,:,:), 1);
    avgkLSPCAmle_train = mean(mle_kLSPCArates_train(1:end-1,:,:,:), 1);
    avgSPCA = mean(SPCArates(1:end-1,:));
    avgSPCA_train = mean(SPCArates_train(1:end-1,:));
    avgkSPCA = mean(kSPCArates(1:end-1,:,:));
    avgkSPCA_train = mean(kSPCArates_train(1:end-1,:,:));
    avgISPCA = mean(ISPCArates(1:end-1,:));
    avgISPCA_train = mean(ISPCArates_train(1:end-1,:));
    avgSPPCA = mean(SPPCArates(1:end-1,:));
    avgSPPCA_train = mean(SPPCArates_train(1:end-1,:));
    avgR4 = mean(ridge_rrr_rates(1:end-1,:,:), 1);
    avgR4_train = mean(ridge_rrr_rates_train(1:end-1,:,:), 1);
    avgSSVD = mean(SSVDrates(1:end-1,:));
    avgSSVD_train = mean(SSVDrates_train(1:end-1,:));
    
    avgPCAvar = mean(PCAvar(1:end-1,:));
    avgkPCAvar = mean(kPCAvar(1:end-1,:,:));
    avgPLSvar = mean(PLSvar(1:end-1,:));
    avgLSPCAvar = mean(LSPCAvar(1:end-1,:,:), 1);
    avgkLSPCAvar = mean(kLSPCAvar(1:end-1,:,:,:), 1);
    avgLSPCAmlevar = mean(mle_LSPCAvar(1:end-1,:,:), 1);
    avgkLSPCAmlevar = mean(mle_kLSPCAvar(1:end-1,:,:,:), 1);
    avgSPCAvar = mean(SPCAvar(1:end-1,:));
    avgkSPCAvar = mean(kSPCAvar(1:end-1,:,:));
    avgISPCAvar = mean(ISPCAvar(1:end-1,:));
    avgSPPCAvar = mean(SPPCAvar(1:end-1,:));
    avgR4var = mean(ridge_rrrvars(1:end-1,:,:), 1);
    avgSSVDvar = mean(SSVDvar(1:end-1,:));
    
    avgPCAvar_train = mean(PCAvar_train(1:end-1,:));
    avgkPCAvar_train = mean(kPCAvar_train(1:end-1,:,:));
    avgPLSvar_train = mean(PLSvar_train(1:end-1,:));
    avgLSPCAvar_train = mean(LSPCAvar_train(1:end-1,:,:), 1);
    avgkLSPCAvar_train = mean(kLSPCAvar_train(1:end-1,:,:,:), 1);
    avgLSPCAmlevar_train = mean(mle_LSPCAvar_train(1:end-1,:,:), 1);
    avgkLSPCAmlevar_train = mean(mle_kLSPCAvar_train(1:end-1,:,:,:), 1);
    avgSPCAvar_train = mean(SPCAvar_train(1:end-1,:));
    avgkSPCAvar_train = mean(kSPCAvar_train(1:end-1,:,:));
    avgISPCAvar_train = mean(ISPCAvar_train(1:end-1,:));
    avgSPPCAvar_train = mean(SPPCAvar_train(1:end-1,:));
    avgR4var_train = mean(ridge_rrrvars_train(1:end-1,:,:), 1);
    avgSSVDvar_train = mean(SSVDvar_train(1:end-1,:));
    
    %% Calc performance for best model and store
    
    % cv over subspace dim
    loc = find(avgPCA==min(avgPCA,[],'all'),1,'last');
    [~,kloc] = ind2sub(size(avgPCA), loc);
    kpca = ks(kloc);
    PCAval(dd) = PCArates(end,kloc);
    PCAvalVar(dd) = PCAvar(end,kloc);
    
    loc = find(avgkPCA==min(avgkPCA,[],'all'),1,'last');
    [~,klock,siglock] = ind2sub(size(avgkPCA), loc);
    kPCAval(dd) = kPCArates(end,klock,siglock);
    kPCAvalVar(dd) = kPCAvar(end,klock,siglock);
    
    loc = find(avgLSPCA==min(avgLSPCA,[],'all'),1,'last');
    [~,kloc,lamloc] = ind2sub(size(avgLSPCA), loc);
    klspca = ks(kloc);
    LSPCAval(dd) = LSPCArates(end,kloc,lamloc);
    LSPCAvalVar(dd) = LSPCAvar(end,kloc,lamloc);
    
    
    loc = find(avgkLSPCA==min(avgkLSPCA,[],'all'),1,'last');
    [~,klock,lamlock,siglock] = ind2sub(size(avgkLSPCA), loc);
    kklspca = ks(klock);
    kLSPCAval(dd) = kLSPCArates(end,klock,lamlock,siglock);
    kLSPCAvalVar(dd) = kLSPCAvar(end,klock,lamlock,siglock);
    
    loc = find(avgLSPCAmle==min(avgLSPCAmle,[],'all'),1,'last');
    [~,kloc] = ind2sub(size(avgLSPCAmle), loc);
    klspcamle = ks(kloc);
    mle_LSPCAval(dd) = mle_LSPCArates(end,kloc);
    mle_LSPCAvalVar(dd) = mle_LSPCAvar(end,kloc);
    
    
    loc = find(avgkLSPCAmle==min(avgkLSPCAmle,[],'all'),1,'last');
    [~,klock,siglock] = ind2sub(size(avgkLSPCAmle), loc);
    kklspca = ks(klock);
    mle_kLSPCAval(dd) = mle_kLSPCArates(end,klock,siglock);
    mle_kLSPCAvalVar(dd) = mle_kLSPCAvar(end,klock,siglock);
    
    
    loc = find(avgISPCA==min(avgISPCA,[],'all'),1,'last');
    [~,kloc] = ind2sub(size(avgISPCA), loc);
    kispca = ks(kloc);
    ISPCAval(dd) = ISPCArates(end,kloc);
    ISPCAvalVar(dd) = ISPCAvar(end,kloc);
    
    loc = find(avgSPPCA==min(avgSPPCA,[],'all'),1,'last');
    [~,kloc] = ind2sub(size(avgSPPCA), loc);
    ksppca = ks(kloc);
    SPPCAval(dd) = SPPCArates(end,kloc);
    SPPCAvalVar(dd) = SPPCAvar(end,kloc);
    
    loc = find(avgSPCA==min(avgSPCA,[],'all'),1,'last');
    [~,kloc] = ind2sub(size(avgSPCA), loc);
    kspca = ks(kloc);
    SPCAval(dd) = SPCArates(end,kloc);
    SPCAvalVar(dd) = SPCAvar(end,kloc);
    
    loc = find(avgkSPCA==min(avgkSPCA,[],'all'),1,'last');
    [~,kloc,sigloc] = ind2sub(size(avgkSPCA), loc);
    kkspca = ks(kloc);
    kSPCAval(dd) = kSPCArates(end,kloc,sigloc);
    kSPCAvalVar(dd) = kSPCAvar(end,kloc,sigloc);
    
    loc = find(avgPLS==min(avgPLS,[],'all'),1,'last');
    [~,kloc] = ind2sub(size(avgPLS), loc);
    kpls = ks(kloc);
    PLSval(dd) = PLSrates(end,kloc);
    PLSvalVar(dd) = PLSvar(end,kloc);
    
    loc = find(avgSSVD==min(avgSSVD,[],'all'),1,'last');
    [~,kloc] = ind2sub(size(avgSSVD), loc);
    kssvd = ks(kloc);
    SSVDval(dd) = SSVDrates(end,kloc);
    SSVDvalVar(dd) = SSVDvar(end,kloc);
    
    loc = find(avgR4==min(avgR4,[],'all'),1,'last');
    [~,kloc,~] = ind2sub(size(avgR4), loc);
    krrr = ks(kloc);
    loc = 1; % RRR with parameter value 0
    RRRval(dd) = ridge_rrr_rates(end,kloc,loc);
    RRRvalVar(dd) = ridge_rrrvars(end,kloc,loc);
    
    loc = find(avgR4==min(avgR4,[],'all'),1,'last');
    [~,kloc,locr4] = ind2sub(size(avgR4), loc);
    kr4 = ks(kloc);
    R4val(dd) = ridge_rrr_rates(end,kloc,locr4);
    R4valVar(dd) = ridge_rrrvars(end,kloc,locr4);
    
    
    %fixed subspace dimension k=2
    
    kloc=1; %k=2
    
    kpca = ks(kloc);
    PCAval_fixed(dd) = PCArates(end,kloc);
    PCAvalVar_fixed(dd) = PCAvar(end,kloc);
    
    loc = find(avgkPCA(:,kloc,:)==min(avgkPCA(:,kloc,:),[],'all'),1,'last');
    [~,~,sigloc] = ind2sub(size(avgkPCA(:,kloc,:)), loc);
    kkpca = ks(kloc);
    kPCAval_fixed(dd) = kPCArates(end,kloc,sigloc);
    kPCAvalVar_fixed(dd) = kPCAvar(end,kloc,sigloc);
    
    loc = find(avgLSPCA(:,kloc,:)==min(avgLSPCA(:,kloc,:),[],'all'),1,'last');
    [~,~,lamloc] = ind2sub(size(avgLSPCA(:,kloc,:)), loc);
    klspca = ks(kloc);
    LSPCAval_fixed(dd) = LSPCArates(end,kloc,lamloc);
    LSPCAvalVar_fixed(dd) = LSPCAvar(end,kloc,lamloc);
    
    loc = find(avgkLSPCA(:,kloc,:,:)==min(avgkLSPCA(:,kloc,:,:),[],'all'),1,'last');
    [~,~,lamlock,siglock] = ind2sub(size(avgkLSPCA(:,kloc,:,:)), loc);
    klock=kloc;
    kklspca = ks(klock);
    kLSPCAval_fixed(dd) = kLSPCArates(end,klock,lamlock,siglock);
    kLSPCAvalVar_fixed(dd) = kLSPCAvar(end,klock,lamlock,siglock);
    
    klspca = ks(kloc);
    mle_LSPCAval_fixed(dd) = mle_LSPCArates(end,kloc);
    mle_LSPCAvalVar_fixed(dd) = mle_LSPCAvar(end,kloc);
    
    loc = find(avgkLSPCAmle(:,kloc,:)==min(avgkLSPCAmle(:,kloc,:),[],'all'),1,'last');
    [~,~,sigloc] = ind2sub(size(avgkLSPCAmle(:,kloc,:)), loc);
    kklspca = ks(kloc);
    mle_kLSPCAval_fixed(dd) = mle_kLSPCArates(end,kloc,sigloc);
    mle_kLSPCAvalVar_fixed(dd) = mle_kLSPCAvar(end,kloc,sigloc);
    
    kispca = ks(kloc);
    ISPCAval_fixed(dd) = ISPCArates(end,kloc);
    ISPCAvalVar_fixed(dd) = ISPCAvar(end,kloc);
    
    ksppca = ks(kloc);
    SPPCAval_fixed(dd) = SPPCArates(end,kloc);
    SPPCAvalVar_fixed(dd) = SPPCAvar(end,kloc);
    
    kspca = ks(kloc);
    SPCAval_fixed(dd) = SPCArates(end,kloc);
    SPCAvalVar_fixed(dd) = SPCAvar(end,kloc);
    
    loc = find(avgkSPCA(:,kloc,:)==min(avgkSPCA(:,kloc,:),[],'all'),1,'last');
    [~,~,sigloc] = ind2sub(size(avgkSPCA(:,kloc,:)), loc);
    kkspca = ks(kloc);
    kSPCAval_fixed(dd) = kSPCArates(end,kloc,sigloc);
    kSPCAvalVar_fixed(dd) = kSPCAvar(end,kloc,sigloc);
    
    kpls = ks(kloc);
    PLSval_fixed(dd) = PLSrates(end,kloc);
    PLSvalVar_fixed(dd) = PLSvar(end,kloc);
    
    kssvd = ks(kloc);
    SSVDval_fixed(dd) = SSVDrates(end,kloc);
    SSVDvalVar_fixed(dd) = SSVDvar(end,kloc);
    
    krrr = ks(kloc);
    loc = 1; % RRR with parameter value 0
    RRRval_fixed(dd) = ridge_rrr_rates(end,kloc,loc);
    RRRvalVar_fixed(dd) = ridge_rrrvars(end,kloc,loc);
    
    loc = find(avgR4(:,kloc,:)==min(avgR4(:,kloc,:),[],'all'),1,'last');
    [~,~,locr4] = ind2sub(size(avgR4(:,kloc,:)), loc);
    kr4 = ks(kloc);
    R4val_fixed(dd) = ridge_rrr_rates(end,kloc,locr4);
    R4valVar_fixed(dd) = ridge_rrrvars(end,kloc,locr4);
    
    % track full values
    LSPCAval_track(dd,:,:,:) = LSPCArates;
    LSPCAvar_track(dd,:,:,:) = LSPCAvar;
    LSPCAval_track_train(dd,:,:,:) = LSPCArates_train;
    LSPCAvar_track_train(dd,:,:,:) = LSPCAvar_train;
    kLSPCAval_track(dd,:,:,:) = kLSPCArates;
    kLSPCAvar_track(dd,:,:,:) = kLSPCAvar;
    kLSPCAval_track_train(dd,:,:,:) = kLSPCArates_train;
    kLSPCAvar_track_train(dd,:,:,:) = kLSPCAvar_train;
    
end

%% save all data
save(strcat(dataset, '_results_dim'))

%% Print results over all subspace dimensions
m = mean(PCAval);
v = mean(PCAvalVar);
sm = std(PCAval);
sv = std(PCAvalVar);
sprintf('PCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(kPCAval);
v = mean(kPCAvalVar);
sm = std(kPCAval);
sv = std(kPCAvalVar);
sprintf('kPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)


m = mean(ISPCAval);
v = mean(ISPCAvalVar);
sm = std(ISPCAval);
sv = std(ISPCAvalVar);
sprintf('ISPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(SPPCAval);
v = mean(SPPCAvalVar);
sm = std(SPPCAval);
sv = std(SPPCAvalVar);
sprintf('SPPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(SPCAval);
v = mean(SPCAvalVar);
sm = std(SPCAval);
sv = std(SPCAvalVar);
sprintf('Barshanerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(SSVDval);
v = mean(SSVDvalVar);
sm = std(SSVDval);
sv = std(SSVDvalVar);
sprintf('SSVD: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(PLSval);
v = mean(PLSvalVar);
sm = std(PLSval);
sv = std(PLSvalVar);
sprintf('PLS: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(RRRval);
v = mean(RRRvalVar);
sm = std(RRRval);
sv = std(RRRvalVar);
sprintf('RRR: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(R4val);
v = mean(R4valVar);
sm = std(R4val);
sv = std(R4valVar);
sprintf('R4: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(LSPCAval);
v = mean(LSPCAvalVar);
sm = std(LSPCAval);
sv = std(LSPCAvalVar);
sprintf('LSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(mle_LSPCAval);
v = mean(mle_LSPCAvalVar);
sm = std(mle_LSPCAval);
sv = std(mle_LSPCAvalVar);
sprintf('mle_LSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(kSPCAval);
v = mean(kSPCAvalVar);
sm = std(kSPCAval);
sv = std(kSPCAvalVar);
sprintf('kBarshanerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(kLSPCAval);
v = mean(kLSPCAvalVar);
sm = std(kLSPCAval);
sv = std(kLSPCAvalVar);
sprintf('kLSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(mle_kLSPCAval);
v = mean(mle_kLSPCAvalVar);
sm = std(mle_kLSPCAval);
sv = std(mle_kLSPCAvalVar);
sprintf('mle_kLSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
%% Print results with k=2

kloc=1; %k=2
k = ks(kloc);

kpca = ks(kloc);
m = mean(PCAval_fixed);
v = mean(PCAvalVar_fixed);
sm = std(PCAval_fixed);
sv = std(PCAvalVar_fixed);
sprintf('PCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

% m = mean(kPCAval_fixed);
% v = mean(kPCAvalVar_fixed);
% sm = std(kPCAval_fixed);
% sv = std(kPCAvalVar_fixed);
% sprintf('kPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(kPCAval_fixed);
v = mean(kPCAvalVar_fixed);
sm = std(kPCAval_fixed);
sv = std(kPCAvalVar_fixed);
sprintf('kPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(ISPCAval_fixed);
v = mean(ISPCAvalVar_fixed);
sm = std(ISPCAval_fixed);
sv = std(ISPCAvalVar_fixed);
sprintf('ISPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(SPPCAval_fixed);
v = mean(SPPCAvalVar_fixed);
sm = std(SPPCAval_fixed);
sv = std(SPPCAvalVar_fixed);
sprintf('SPPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(SPCAval_fixed);
v = mean(SPCAvalVar_fixed);
sm = std(SPCAval_fixed);
sv = std(SPCAvalVar_fixed);
sprintf('Barshanerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(SSVDval_fixed);
v = mean(SSVDvalVar_fixed);
sm = std(SSVDval_fixed);
sv = std(SSVDvalVar_fixed);
sprintf('SSVD: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(PLSval_fixed);
v = mean(PLSvalVar_fixed);
sm = std(PLSval_fixed);
sv = std(PLSvalVar_fixed);
sprintf('PLS: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(RRRval_fixed);
v = mean(RRRvalVar_fixed);
sm = std(RRRval_fixed);
sv = std(RRRvalVar_fixed);
sprintf('RRR: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(R4val_fixed);
v = mean(R4valVar_fixed);
sm = std(R4val_fixed);
sv = std(R4valVar_fixed);
sprintf('R4: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(LSPCAval_fixed);
v = mean(LSPCAvalVar_fixed);
sm = std(LSPCAval_fixed);
sv = std(LSPCAvalVar_fixed);
sprintf('LSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(mle_LSPCAval_fixed);
v = mean(mle_LSPCAvalVar_fixed);
sm = std(mle_LSPCAval_fixed);
sv = std(mle_LSPCAvalVar_fixed);
sprintf('mle_LSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(kSPCAval_fixed);
v = mean(kSPCAvalVar_fixed);
sm = std(kSPCAval_fixed);
sv = std(kSPCAvalVar_fixed);
sprintf('kBarshanerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(kLSPCAval_fixed);
v = mean(kLSPCAvalVar_fixed);
sm = std(kLSPCAval_fixed);
sv = std(kLSPCAvalVar_fixed);
sprintf('kLSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(mle_kLSPCAval_fixed);
v = mean(mle_kLSPCAvalVar_fixed);
sm = std(mle_kLSPCAval_fixed);
sv = std(mle_kLSPCAvalVar_fixed);
sprintf('kLSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)


% %% plot error - var tradeoff curves
% 
% % for t = 1:length(ks)
% figure()
% hold on
% plot(mean(PCAvalVar_fixed), mean(PCAval_fixed), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(mean(kPCAvalVar_fixed), mean(kPCAval_fixed), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% plot(squeeze(LSPCAvar(end,2,:)), squeeze(LSPCArates(end,2,:)), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(squeeze(kLSPCAvar(end,2,:)), squeeze(kLSPCArates(end,2,:)), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(mle_LSPCAvalVar_fixed), mean(mle_LSPCAval_fixed), '*', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(mle_kLSPCAvalVar_fixed), mean(mle_kLSPCAval_fixed), '*', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(ISPCAvalVar_fixed), mean(ISPCAval_fixed), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(SPPCAvalVar_fixed), mean(SPPCAval_fixed), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(SPCAvalVar_fixed), mean(SPCAval_fixed), '+', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(kSPCAvalVar_fixed), mean(kSPCAval_fixed), '>', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(RRRvalVar_fixed), mean(RRRval_fixed), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(PLSvalVar_fixed), mean(PLSval_fixed), 'h', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(SSVDvalVar_fixed), mean(SSVDval_fixed), 'd', 'MarkerSize', 20, 'LineWidth', 2)
% 
% xlabel('Variation Explained', 'fontsize', 25)
% %title('Test', 'fontsize', 25)
% ylabel('MSE', 'fontsize', 25)
% %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
% set(gca, 'fontsize', 25)
% lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'LSPCA (MLE)', 'kLSPCA (MLE)', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'Location', 'best'); lgd.FontSize = 15;
% %lgd = legend('LSPCA', 'R4', 'PLS', 'SPPCA', 'Barshan', 'SSVD', 'PCA', 'Location', 'southeast'); lgd.FontSize = 15;
% %ylim([0, 0.12])
% %set(gca, 'YScale', 'log')
% xlim([0,1])
% 
% %% plot error - var tradeoff curves
% 
% % for t = 1:length(ks)
% figure()
% hold on
% plot(mean(PCAvalVar), mean(PCAval), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(mean(kPCAvalVar), mean(kPCAval), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(squeeze(LSPCAvar(end,2,:)), squeeze(LSPCArates(end,2,:)), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% % plot(squeeze(kLSPCAvar(end,2,:)), squeeze(kLSPCArates(end,2,:)), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(LSPCAvalVar), mean(LSPCAval), '*', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(kLSPCAvalVar), mean(kLSPCAval), '*', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(mle_LSPCAvalVar), mean(mle_LSPCAval), '*', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(mle_kLSPCAvalVar), mean(mle_kLSPCAval), '*', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(ISPCAvalVar), mean(ISPCAval), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(SPPCAvalVar), mean(SPPCAval), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(SPCAvalVar), mean(SPCAval), '+', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(kSPCAvalVar), mean(kSPCAval), '>', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(RRRvalVar), mean(RRRval), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(PLSvalVar), mean(PLSval), 'h', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(SSVDvalVar), mean(SSVDval), 'd', 'MarkerSize', 20, 'LineWidth', 2)
% 
% xlabel('Variation Explained', 'fontsize', 25)
% %title('Test', 'fontsize', 25)
% ylabel('MSE', 'fontsize', 25)
% %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
% set(gca, 'fontsize', 25)
% lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'LSPCA (MLE)', 'kLSPCA (MLE)', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'Location', 'best'); lgd.FontSize = 15;
% %lgd = legend('LSPCA', 'R4', 'PLS', 'SPPCA', 'Barshan', 'SSVD', 'PCA', 'Location', 'southeast'); lgd.FontSize = 15;
% %ylim([0, 0.12])
% %set(gca, 'YScale', 'log')
% xlim([0,1])

% 
% %% add MLE results to plot
% avgLSPCA = mean(LSPCArates, 1);
% avgLSPCA_train = mean(LSPCArates_train, 1);
% avgkLSPCA = mean(kLSPCArates, 1);
% avgkLSPCA_train = mean(kLSPCArates_train, 1);
% avgLSPCAvar = mean(LSPCAvar, 1);
% avgkLSPCAvar = mean(kLSPCAvar, 1);
% avgLSPCAvar_train = mean(LSPCAvar_train, 1);
% avgkLSPCAvar_train = mean(kLSPCAvar_train, 1);
% hold on
% plot(avgLSPCAvar(klspca), avgLSPCA(klspca), 'b*', 'LineWidth', 2, 'MarkerSize', 20)
% [m, loc] = min(avgkLSPCA(1,kklspca,:));
% plot(avgkLSPCAvar(1,kklspca,loc), m, 'r*', 'LineWidth', 2, 'MarkerSize', 20)
% lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'LSPCA (CV)', 'kLSPCA (CV)', 'Location', 'best'); lgd.FontSize = 15;
% 
% 
% %% training error and var
% t = find(ks==klspca);
% temp = avgLSPCA(:,t,:,:);
% temp = reshape(temp, [length(Gammas), length(Lambdas)]);
% temp_train = reshape(avgLSPCA_train(:,t,:,:), [length(Gammas), length(Lambdas)]);
% tempv = avgLSPCAvar(:,t,:,:);
% tempv = reshape(tempv, [length(Gammas), length(Lambdas)]);
% tempv_train = reshape(avgLSPCAvar_train(:,t,:,:), [length(Gammas), length(Lambdas)]);
% [mLSPCA, I] = min(temp);
% I = sub2ind(size(temp),I,1:length(I));
% vLSPCA = tempv(I);
% mLSPCA_train = temp_train(I);
% vLSPCA_train = tempv_train(I);
% 
% t = find(ks==kklspca);
% temp = avgkLSPCA(:,t,:,:);
% temp = reshape(temp, [length(Gammas), length(Lambdas), length(sigmas)]);
% temp_train = reshape(avgkLSPCA_train(:,t,:,:), [length(Gammas), length(Lambdas), length(sigmas)]);
% tempv = avgkLSPCAvar(:,t,:,:);
% tempv = reshape(tempv, [length(Gammas), length(Lambdas), length(sigmas)]);
% tempv_train = reshape(avgkLSPCAvar_train(:,t,:,:), [length(Gammas), length(Lambdas), length(sigmas)]);
% [res, I1] = min(temp, [], 1);
% [~,a,b] = size(I1);
% for i=1:a
%     for j=1:b
%         tempvv(i,j) = tempv(I1(1,i,j),i,j);
%         tempmm_train(i,j) = temp_train(I1(1,i,j),i,j);
%         tempvv_train(i,j) = tempv_train(I1(1,i,j),i,j);
%     end
% end
% [mkLSPCA, I2] = min(res, [], 3);
% [~,c] = size(I2);
% for i=1:c
%     vkLSPCA(i) = tempvv(i,I2(1,i));
%     mkLSPCA_train(i) = tempmm_train(i,10);
%     vkLSPCA_train(i) = tempvv_train(i,10);
% end
% 
% 
% % for t = 1:length(ks)
% figure()
% hold on
% plot(avgPCAvar_train(1,kpca-1), avgPCA_train(1,kpca-1), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% %plot(avgkPCAvar_train(1,t), avgkPCA(1,t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% plot(vLSPCA_train(:), mLSPCA_train(:), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% % plot(lambda_avgLSPCAvar_train(1,t,:), lambda_avgLSPCA(1,t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(vkLSPCA_train(:), mkLSPCA_train(:), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(avgISPCAvar_train(1,kispca-1), avgISPCA_train(1,kispca-1), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgSPPCAvar_train(1,ksppca-1), avgSPPCA_train(1,ksppca-1), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgSPCAvar_train(1,kspca-1), avgSPCA_train(1,kspca-1), '+', 'MarkerSize', 20, 'LineWidth', 2)
% loc = find(avgkSPCA(:,kkspca-1,:)==min(avgkSPCA(:,kkspca-1,:),[],'all'),1,'last');
% [~,~,sigloc] = ind2sub(size(avgkSPCA(:,kkspca-1,:)), loc);
% plot(avgkSPCAvar_train(1,kkspca-1,sigloc), avgkSPCA_train(1,kkspca-1,sigloc), '>', 'MarkerSize', 20, 'LineWidth', 2)
% x=avgR4var_train(1,kr4-1,2:end); y = avgR4_train(1,kr4-1, 2:end);
% plot(x(:), y(:), ':', 'LineWidth', 2, 'MarkerSize', 20)
% plot(avgR4var_train(1,krrr-1,1), avgR4_train(1,krrr-1, 1), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgPLSvar_train(1,kpls-1), avgPLS_train(1,kpls-1), 'h', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgSSVDvar_train(1,kssvd-1), avgSSVD_train(1,kssvd-1), 'd', 'MarkerSize', 20, 'LineWidth', 2)
% 
% xlabel('Variation Explained', 'fontsize', 25)
% %title('Test', 'fontsize', 25)
% ylabel('MSE', 'fontsize', 25)
% %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
% set(gca, 'fontsize', 25)
% lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'Location', 'best'); lgd.FontSize = 15;
% %lgd = legend('LSPCA', 'R4', 'PLS', 'SPPCA', 'Barshan', 'SSVD', 'PCA', 'Location', 'southeast'); lgd.FontSize = 15;
% %ylim([0, 0.12])
% %set(gca, 'YScale', 'log')
% xlim([0,1])
% 
% 
% %% add MLE results to plot
% %% add MLE results to plot
% avgLSPCA = mean(LSPCArates, 1);
% avgLSPCA_train = mean(LSPCArates_train, 1);
% avgkLSPCA = mean(kLSPCArates, 1);
% avgkLSPCA_train = mean(kLSPCArates_train, 1);
% avgLSPCAvar = mean(LSPCAvar, 1);
% avgkLSPCAvar = mean(kLSPCAvar, 1);
% avgLSPCAvar_train = mean(LSPCAvar_train, 1);
% avgkLSPCAvar_train = mean(kLSPCAvar_train, 1);
% hold on
% plot(avgLSPCAvar_train(klspca-1), avgLSPCA_train(klspca-1), 'b*', 'LineWidth', 2, 'MarkerSize', 20)
% [m, loc] = min(avgkLSPCA_train(1,kklspca-1,:));
% plot(avgkLSPCAvar_train(1,kklspca-1,loc), m, 'r*', 'LineWidth', 2, 'MarkerSize', 20)
% lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'LSPCA (CV)', 'kLSPCA (CV)', 'Location', 'best'); lgd.FontSize = 15;
% 
% % end
% % saveas(gcf, strcat(dataset, 'multi_obj_gamma.jpg'))
% %
% %
% % %% plot training error - var tradeoff curves
% % for t = 1:length(ks)
% %     figure()
% %     hold on
% %     plot(avgPCAvar_train(t), avgPCA_train(t), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% %     %    plot(avgkPCAvar_train(t), avgkPCA_train(t), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgLSPCAvar_train(t,:,lamloc), avgLSPCA_train(t, :,lamloc), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% %     plot(avgkLSPCAvar_train(t,:,lamlock,siglock), avgkLSPCA_train(t, :,lamlock,siglock), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% %     plot(avgISPCAvar_train(t), avgISPCA_train(t), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgSPPCAvar_train(t), avgSPPCA_train(t), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgSPCAvar_train(t), avgSPCA_train(t), '+', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgkSPCAvar_train(t), avgkSPCA_train(t), '>', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgR4var_train(t,2:end), avgR4_train(t, 2:end), ':', 'LineWidth', 2, 'MarkerSize', 20)
% %     plot(avgR4var_train(t,1), avgR4_train(t, 1), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgPLSvar_train(t), avgPLS_train(t), 'h', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgSSVDvar_train(t), avgSSVD_train(t), 'd', 'MarkerSize', 20, 'LineWidth', 2)
% %
% %     xlabel('Variation Explained', 'fontsize', 25)
% %     %title('Train', 'fontsize', 25)
% %     ylabel('MSE', 'fontsize', 25)
% %     %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
% %     set(gca, 'fontsize', 25)
% %     lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'Location', 'best'); lgd.FontSize = 15;
% %     %ylim([0, 0.12])
% %     %set(gca, 'YScale', 'log')
% %     xlim([0,1])
% % end
% % saveas(gcf, strcat(dataset, 'multi_obj_train_gamma.jpg'))
% %
% %
% %
% 
% 


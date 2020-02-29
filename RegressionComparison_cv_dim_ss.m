numExps = 10;
for dd = 1:numExps
    %% setup and load data
    load(strcat(dataset, '.mat'));
    [n, p] = size(X);
    [~, q] = size(Y);
    %ks = 2:min(10, p-1);
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
        Xlab = X(cvidx==l, :); %lth labled set
        Ylab = Y(cvidx==l, :);
        %randomly select a fold to holdout, this is our test set for
        %inductive semisupervised learning
        testIdx = randi(kfold);
        while testIdx == l
           testIdx = randi(kfold);
        end
        Xtest = X(cvidx==testIdx, :); %lth test set
        Ytest = Y(cvidx==testIdx, :);
        Xunlab = X((cvidx~=l & cvidx~=testIdx), :); %lth unlabled set
        Yunlab = Y((cvidx~=l & cvidx~=testIdx), :);
        [Xlab,Xunlab,Xtest,Ylab,Yunlab,Ytest] = ss_center_data(Xlab,Xunlab,Xtest,Ylab,Yunlab,Ytest,'regression');
        Xlabs{l} = Xlab; %lth centered labled set
        Ylabs{l} = Ylab;
        Xunlabs{l} = Xunlab; %lth centered unlabled set
        Yunlabs{l} = Yunlab;
        Xtests{l} = Xtest; %lth centered test set
        Ytests{l} = Ytest;
    end  
    
    % store the independent test set, and corresponding labing set (non holdout data)
    % at the end of the lab  and test cell arrays for convenience of
    % implementation

    %pick 5% of data for labeled data and use the rest as unlabeled
    labIdx = crossvalind('kfold',n,kfold);
    Xlab = X(labIdx==1,:); Ylab = Y(labIdx==1,:);
    Xunlab = X(labIdx~=1,:); Yunlab = Y(labIdx~=1,:);
    %center the labeled, unlabeled, and holdout data
    [Xlab,Xunlab,Xhold,Ylab,Yunlab,Yhold] = ss_center_data(Xlab,Xunlab,Xhold,Ylab,Yunlab,Yhold,'regression');
    Xlabs{kfold+1} = Xlab; %lth centered labeled set
    Ylabs{kfold+1} = Ylab;
    Xunlabs{kfold+1} = Xunlab; %lth centered unlabeled set
    Yunlabs{kfold+1} = Yunlab;
    Xtests{kfold+1} = Xhold; %lth centered testing set
    Ytests{kfold+1} = Yhold;

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
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            Xunlab = Xunlabs{l};
            Yunlab = Yunlabs{l};
            [nlab, ~] = size(Ylab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            tic
            [Lpca, Zpca] = pca([Xlab; Xunlab], 'NumComponents', k);
            Zpcas{l,t} = Zpca;
            % compute embedding for test data
            pcaXtest = Xtest*Lpca;
            pcaXtests{l,t} = pcaXtest;
            PCAtimes(l,t,t) = toc;
            % compute error
            B = (Xlab*Lpca) \ Ylab;
            PCRYlab = (Xlab*Lpca)*B;
            PCRYtest = pcaXtest*B;
            mse = norm(PCRYtest - Ytest, 'fro')^2 /ntest;
            PCArates(l,t) = mse ;
            PCArates_lab(l,t) = norm(PCRYlab - Ylab, 'fro')^2 / nlab;
            PCAvar(l,t) = norm(Xtest*Lpca, 'fro') / norm(Xtest, 'fro');
            PCAvar_lab(l,t) = norm(Zpca, 'fro') / norm([Xlab;Xunlab], 'fro');
        end
        
        %% PLS
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            [nlab, ~] = size(Ylab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            %solve for basis
            try
                [Xloadings,Yloadings,Xscores,Yscores,betaPLS,pctVar,~,stats] = plsregress(Xlab,Ylab,min(k, nlab));
            catch
                [Xloadings,Yloadings,Xscores,Yscores,betaPLS,pctVar,~,stats] = plsregress(Xlab,Ylab,1);
            end
            Lpls = orth(stats.W);
            % predict
            PLSYtest = [ones(ntest,1) Xtest]*betaPLS;
            PLSYlab = [ones(nlab,1) Xlab]*betaPLS;
            % compute error
            mse = norm(PLSYtest - Ytest, 'fro')^2;
            PLSrates(l,t) = mse / ntest;
            PLSrates_lab(l,t) = norm(PLSYlab - Ylab, 'fro')^2 / nlab;
            PLSvar(l,t) = norm(Xtest*Lpls, 'fro') / norm(Xtest, 'fro');
            PLSvar_lab(l,t) = sum(pctVar(1,:));
        end
        
        
        %% LSPCA
        
        Lambdas = fliplr(logspace(-3, 0, 51));
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            Xunlab = Xunlabs{l};
            [nlab, ~] = size(Ylab);
            [nunlab,~] = size(Xunlab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            for ii = 1:length(Lambdas)
                Lambda = Lambdas(ii);
                
                if ii == 1
                    [Zunlab, ~, Zlspca, Llspca, B] = lspca_sub_ss(Xunlab, Xlab, Ylab, Lambda, k, 0);
                else
                    [Zunlab, ~, Zlspca, Llspca, B] = lspca_sub_ss(Xunlab, Xlab, Ylab, Lambda, k, Llspca);
                end
                Ls{l,t,ii} = Llspca;
                %predict
                LSPCAXtest = Xtest*Llspca;
                LSPCAXlab = Xlab*Llspca;
                LSPCAYtest = LSPCAXtest*B;
                LSPCAYlab = LSPCAXlab*B;
                lspca_mbd_test{l,t,ii} =LSPCAXtest;
                lspca_mbd_lab{l,t,ii} = Zlspca;
                mse = norm(LSPCAYtest - Ytest, 'fro')^2 / ntest;
                % compute error
                lab_err = norm(Ylab - LSPCAYlab, 'fro')^2 / nlab;
                LSPCArates(l,t,ii) = mse ;
                LSPCArates_lab(l,t, ii) = lab_err;
                LSPCAvar(l,t, ii) = norm(LSPCAXtest, 'fro') / norm(Xtest, 'fro');
                LSPCAvar_lab(l,t, ii) = norm(Zlspca, 'fro') / norm(Xlab, 'fro');
            end
        end
        
        %% LSPCA (MLE)
        
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            Xunlab = Xunlabs{l};
            [nlab, ~] = size(Ylab);
            [nunlab,~] = size(Xunlab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            %solve
            %Linit = orth(randn(p,k));
            Linit = 0;
            %         V = pca(Xlab);
            %        Linit = V(:,1:2);
            %         c = 0.5;
            %         Linit = orth((1-c)*V(:,1:2) + c*randn(p,k));
            [Zunlab, ~, Zlspca, Llspca, B] = lspca_MLE_sub_ss(Xunlab, Xlab, Ylab, k, Linit, 1e-8);
            mle_Ls{l,t} = Llspca;
            %predict
            LSPCAXtest = Xtest*Llspca;
            LSPCAYtest = LSPCAXtest*B;
            LSPCAYlab = Zlspca*B;
            mle_lspca_mbd_test{l,t} =LSPCAXtest;
            mle_lspca_mbd_lab{l,t} = Zlspca;
            mse = norm(LSPCAYtest - Ytest, 'fro')^2 / ntest;
            % compute error
            lab_err = norm(Ylab - LSPCAYlab, 'fro')^2 / nlab;
            mle_LSPCArates(l,t) = mse ;
            mle_LSPCArates_lab(l,t) = lab_err;
            mle_LSPCAvar(l,t) = norm(LSPCAXtest, 'fro') / norm(Xtest, 'fro');
            mle_LSPCAvar_lab(l,t) = norm(Zlspca, 'fro') / norm(Xlab, 'fro');
        end
        
        %% kLSPCA
        Lambdas = fliplr(logspace(-3, 0, 51));
        
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            Xunlab = Xunlabs{l};
            [nlab, ~] = size(Ylab);
            [nunlab,~] = size(Xunlab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            for kk = 1:length(sigmas)
                sigma = sigmas(kk)
                for ii = 1:length(Lambdas)
                    Lambda = Lambdas(ii);
                    
                    if ii == 1
                        [Zunlab, ~, Zklspca, Lorth, B, Klspca] = klspca_sub_ss(Xunlab, Xlab, Ylab, Lambda, sigma, k, 0, 0);
                    else
                        [Zunlab, ~, Zklspca, Lorth, B, Klspca] = klspca_sub_ss(Xunlab, Xlab, Ylab, Lambda, sigma, k, Lorth, Klspca);
                    end
                    embedFunc = @(data) klspca_embed(data, [Xlab; Xunlab], Lorth, sigma);
                    kLs{l,t,ii,kk} = Lorth;
                    kLSPCAXtest = embedFunc(Xtest);
                    klspca_mbd_test{l,t,ii,kk} = kLSPCAXtest;
                    kLSPCAXlab = Zklspca;
                    klspca_mbd_lab{l,t,ii,kk} = Zklspca;
                    kLSPCAYtest = kLSPCAXtest*B;
                    kLSPCAYlab = kLSPCAXlab*B;
                    mse = norm(kLSPCAYtest - Ytest, 'fro')^2 / ntest;
                    lab_err = norm(Ylab - kLSPCAYlab, 'fro')^2 / nlab;
                    kLSPCArates(l,t,ii,kk) = mse ;
                    kLSPCArates_lab(l,t,ii,kk) = lab_err;
                    kLSPCAvar(l,t,ii,kk) = norm(kLSPCAXtest, 'fro') / norm(gaussian_kernel(Xtest,[Xlab; Xunlab],sigma), 'fro');
                    kLSPCAvar_lab(l,t,ii,kk) = norm(Zklspca, 'fro') / norm(Klspca, 'fro');
                end
            end
            
        end
        
        %% kLSPCA (MLE)
        
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            Xunlab = Xunlabs{l};
            [nlab, ~] = size(Ylab);
            [nunlab,~] = size(Xunlab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            Linit = orth(randn(nlab+nunlab,k));
            %Linit = 0;
            for ss = 1:length(sigmas)
                sigma = sigmas(ss)
                [Zunlab, ~, Zklspca, Lorth, B, Klspca] = klspca_MLE_sub_ss(Xunlab, Xlab, Ylab, sigma, k, Linit, 0, 1e-10);
                embedFunc = @(data) klspca_embed(data, [Xlab; Xunlab], Lorth, sigma);
                mle_kLs{l,t,ss} = Lorth;
                kLSPCAXtest = embedFunc(Xtest);
                mle_klspca_mbd_test{l,t,ss} = kLSPCAXtest;
                mle_klspca_mbd_lab{l,t,ss} = Zklspca;
                kLSPCAYtest = kLSPCAXtest*B;
                kLSPCAYlab = Zklspca*B;
                mse = norm(kLSPCAYtest - Ytest, 'fro')^2 / ntest;
                lab_err = norm(Ylab - kLSPCAYlab, 'fro')^2 / nlab;
                mle_kLSPCArates(l,t,ss) = mse ;
                mle_kLSPCArates_lab(l,t,ss) = lab_err;
                mle_kLSPCAvar(l,t,ss) = norm(kLSPCAXtest, 'fro') / norm(gaussian_kernel(Xtest,[Xlab; Xunlab],sigma), 'fro');
                mle_kLSPCAvar_lab(l,t,ss) = norm(Zklspca, 'fro') / norm(Klspca, 'fro');
            end
        end
        
        
        %% KPCA
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            Xunlab = Xunlabs{l};
            [nlab, ~] = size(Ylab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            for jj = 1:length(sigmas)
                sigma = sigmas(jj);
                K = gaussian_kernel([Xlab; Xunlab], [Xlab; Xunlab], sigma);
                Klab = K(1:nlab,:);
                Ktest = gaussian_kernel(Xtest, [Xlab; Xunlab], sigma);
                tic
                [Lkpca, Zkpca] = pca(K, 'NumComponents', k);
                kPCAtimes(l,t,t,jj) = toc;
                Zkpcas{l,t,jj} = Zkpca;
                % compute embedding for test data
                kpcaXtest = Ktest*Lkpca;
                kpcaXtests{l,t,jj} = kpcaXtest;
                % compute error
                B = (Klab*Lkpca) \ Ylab;
                kpcaYtest = kpcaXtest*B;
                kpcaYlab = (Klab*Lkpca)*B;
                mse = norm(kpcaYtest - Ytest, 'fro')^2 / ntest;
                lab_err = norm(Ylab - kpcaYlab, 'fro')^2 / nlab;
                kPCArates(l,t,jj) = mse;
                kPCArates_lab(l,t,jj) = lab_err;
                kPCAvar(l,t,jj) = norm(kpcaXtest, 'fro') / norm(Ktest, 'fro');
                kPCAvar_lab(l,t,jj) = norm(Zkpca, 'fro') / norm(K, 'fro');
            end
        end
        
        %% R4
        
        if ~strcmp(dataset, 'DLBCL')
            params = logspace(-1, 6, 101);
            params(1) = 0; %edge case is just RRR
            ridge_solns = [];
            for l = 1:kfold+1
                test_num = l
                % get lth fold
                Xlab = Xlabs{l};
                Ylab = Ylabs{l};
                [nlab, ~] = size(Ylab);
                Xtest = Xtests{l};
                Ytest = Ytests{l};
                [ntest, ~] = size(Ytest);
                
                for pp = 1:length(params)
                    param = params(pp);
                    [Mrrr, Lrrr] = rrr_ridge(Xlab, Ylab, k, param);
                    ridge_solns(:,:,pp) = Lrrr;
                end
                ridge_rrr_solns{l,t} = ridge_solns;
                Ls_rrr = ridge_rrr_solns{l,t};
                [~, ~, numsols] = size(Ls_rrr);
                for tt = 1:numsols
                    b = (Xlab*Ls_rrr(:,:,tt)') \ Ylab;
                    ridge_rrr_err = norm(Ytest - Xtest*Ls_rrr(:,:,tt)'*b, 'fro')^2 / ntest;
                    ridge_rrr_err_lab = norm(Ylab - Xlab*Ls_rrr(:,:,tt)'*b, 'fro')^2 / nlab;
                    ridge_rrr_var = norm(Xtest*Ls_rrr(:,:,tt)', 'fro') / norm(Xtest, 'fro');
                    ridge_rrr_var_lab = norm(Xlab*Ls_rrr(:,:,tt)', 'fro') / norm(Xlab, 'fro');
                    ridge_rrr_rates(l,t, tt) = ridge_rrr_err;
                    ridge_rrr_rates_lab(l,t,tt) = ridge_rrr_err_lab;
                    ridge_rrrvars(l,t,tt) = ridge_rrr_var;
                    ridge_rrrvars_lab(l,t, tt) = ridge_rrr_var_lab;
                    Lridge_rrrs{l, t, tt} = Ls_rrr(:,:,tt);
                end
            end
        else
            ridge_rrr_rates(l,t, 1) = nan;
            ridge_rrr_rates_lab(l,t,1) = nan;
            ridge_rrrvars(l,t,1) = nan;
            ridge_rrrvars_lab(l,t,1) = nan;
        end
        
        %% ISPCA
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            [nlab, ~] = size(Ylab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            %find basis
            tic
            [Zispca, Lispca, B] = ISPCA(Xlab,Ylab,min(k,nlab));
            ISPCAtimes(l,t) = toc;
            % predict
            ISPCAXtest = Xtest*Lispca';
            ISPCAYtest = ISPCAXtest*B;
            ISPCAYlab = Zispca*B;
            % compute error
            mse = norm(ISPCAYtest - Ytest, 'fro')^2 / ntest;
            ISPCArates(l,t) = mse ;
            ISPCArates_lab(l,t) = norm(ISPCAYlab - Ylab, 'fro')^2 / nlab;
            ISPCAvar(l,t) = norm(Xtest*Lispca', 'fro') / norm(Xtest, 'fro');
            ISPCAvar_lab(l,t) = norm(Xlab*Lispca', 'fro') / norm(Xlab, 'fro');
        end
        
        %% SPPCA
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            Xunlab = Xunlabs{l};
            [nlab, ~] = size(Ylab);
            [nunlab,~] = size(Xunlab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            tic
            SPPCAXtest = {};
            SPPCAYtest = {};
            SPPCAYlab = {};
            sppca_err = [];
            % solve
            for count = 1%:10 %do 10 initializations and take the best b/c ends up in bad local minima a lot
                [Zunlab, ~, Zsppca, Lsppca, ~] = SPPCA_SS(Xunlab,Xlab,Ylab,k,1e-6);
                Lsppca = Lsppca';
                Zsppcas{count} = Zsppca;
                Lsppcas{count} = Lsppca;
                SPPCAXtest{count} = Xtest*Lsppca;
                SPPCAYtest{count} = SPPCAXtest{count}*B;
                SPPCAYlab{count} = Zsppca*B;
                sppca_err(count) =  norm(SPPCAYlab{count} - Ylab, 'fro')^2;
            end
            [~, loc] = min(sppca_err);
            Zsppca = Zsppcas{loc};
            Lsppca = orth(Lsppcas{loc});
            % Predict
            SPPCAXtest = SPPCAXtest{loc};
            SPPCAYtest = SPPCAYtest{loc};
            SPPCAYlab = SPPCAYlab{loc};
            SPPCAtimes(l,t) = toc;
            % compute error
            mse = norm(SPPCAYtest - Ytest, 'fro')^2/ ntest;
            SPPCArates(l,t) = mse ;
            SPPCArates_lab(l,t)= norm(SPPCAYlab - Ylab, 'fro')^2 / nlab;
            SPPCAvar(l,t) = norm(Xtest*Lsppca, 'fro') / norm(Xtest, 'fro');
            SPPCAvar_lab(l,t) = norm(Xlab*Lsppca, 'fro') / norm(Xlab, 'fro');
        end
        
        
        %% Barshan
        
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            [nlab, ~] = size(Ylab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            %learn basis
            [Zspca Lspca] = SPCA(Xlab', Ylab', min(k,nlab));
            spcaXtest = Xtest*Lspca';
            % predict
            betaSPCA = Zspca \ Ylab;
            SPCAYtest = spcaXtest * betaSPCA;
            SPCAYlab = Zspca*betaSPCA;
            %compute error
            mse = norm(SPCAYtest - Ytest, 'fro')^2/ ntest;
            SPCArates(l,t) = mse ;
            SPCArates_lab(l,t) = norm(SPCAYlab - Ylab, 'fro')^2 / nlab;
            SPCAvar(l,t) = norm(Xtest*Lspca', 'fro') / norm(Xtest, 'fro');
            SPCAvar_lab(l,t) = norm(Xlab*Lspca', 'fro') / norm(Xlab, 'fro');
        end
        
        %% Perform Barshan's KSPCA based 2D embedding
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            [nlab, ~] = size(Ylab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            for jj = 1:length(sigmas)
                sigma = sigmas(jj);
                % calc with best param on full labing set
                barshparam.ktype_y = 'linear';
                barshparam.kparam_y = 1;
                barshparam.ktype_x = 'rbf';
                barshparam.kparam_x = sigma;
                [Zkspca, Lkspca] = KSPCA(Xlab', Ylab', min(k,nlab), barshparam);
                Zkspca = Zkspca';
                %do regression in learned basis
                betakSPCA = Zkspca \ Ylab;
                Ktest = gaussian_kernel(Xtest, Xlab, sigma);
                K = gaussian_kernel(Xlab, Xlab, sigma);
                kspcaXtest = Ktest*Lkspca;
                kSPCAYtest = kspcaXtest*betakSPCA;
                kSPCAYlab = Zkspca*betakSPCA;
                kspca_mbd_test{l,t,jj} = kspcaXtest;
                kspca_mbd_lab{l,t,jj} = Zkspca;
                % compute error
                mse = norm(kSPCAYtest - Ytest, 'fro')^2 / ntest;
                kSPCArates(l,t,jj) = mse;
                kSPCArates_lab(l,t,jj) = norm(kSPCAYlab - Ylab, 'fro')^2  / nlab;
                kSPCAvar(l,t,jj) = norm(kspcaXtest, 'fro') / norm(Ktest, 'fro');
                kSPCAvar_lab(l,t,jj) = norm(Zkspca, 'fro') / norm(K, 'fro');
            end
        end
        
        
%         %% Sup SVD
%         for l = 1:kfold+1
%             test_num = l;
%             % get lth fold
%             Xlab = Xlabs{l};
%             Ylab = Ylabs{l};
%             [nlab, ~] = size(Ylab);
%             Xtest = Xtests{l};
%             Ytest = Ytests{l};
%             [ntest, ~] = size(Ytest);
%             % solve
%             [~,V,Zssvd,~,~]=SupPCA(Ylab,Xlab,k);
%             Lssvd = V';
%             % Predict
%             Bssvd = Zssvd \ Ylab;
%             SSVDXtest = Xtest*Lssvd';
%             SSVDYtest = SSVDXtest*Bssvd;
%             SSVDYlab = Zssvd*Bssvd;
%             %compute error
%             mse = norm(SSVDYtest - Ytest, 'fro')^2 / ntest;
%             SSVDrates(l,t) = mse ;
%             SSVDrates_lab(l,t) = norm(SSVDYlab - Ylab, 'fro')^2 / nlab;
%             SSVDvar(l,t) = norm(SSVDXtest, 'fro') / norm(Xtest, 'fro');
%             SSVDvar_lab(l,t) = norm(Zssvd, 'fro') / norm(Xlab, 'fro');
%         end
        
        
    end
    
    %% compute avg performance for each k
    
    %means
    avgPCA = mean(PCArates(1:end-1,:));
    avgPCA_lab = mean(PCArates_lab(1:end-1,:));
    avgkPCA = mean(kPCArates(1:end-1,:,:));
    avgkPCA_lab = mean(kPCArates_lab(1:end-1,:,:));
    avgPLS = mean(PLSrates(1:end-1,:));
    avgPLS_lab = mean(PLSrates_lab(1:end-1,:));
    avgLSPCA = mean(LSPCArates(1:end-1,:,:), 1);
    avgLSPCA_lab = mean(LSPCArates_lab(1:end-1,:,:), 1);
    avgkLSPCA = mean(kLSPCArates(1:end-1,:,:,:), 1);
    avgkLSPCA_lab = mean(kLSPCArates_lab(1:end-1,:,:,:), 1);
    avgLSPCAmle = mean(mle_LSPCArates(1:end-1,:,:), 1);
    avgLSPCAmle_lab = mean(mle_LSPCArates_lab(1:end-1,:,:), 1);
    avgkLSPCAmle = mean(mle_kLSPCArates(1:end-1,:,:,:), 1);
    avgkLSPCAmle_lab = mean(mle_kLSPCArates_lab(1:end-1,:,:,:), 1);
    avgSPCA = mean(SPCArates(1:end-1,:));
    avgSPCA_lab = mean(SPCArates_lab(1:end-1,:));
    avgkSPCA = mean(kSPCArates(1:end-1,:,:));
    avgkSPCA_lab = mean(kSPCArates_lab(1:end-1,:,:));
    avgISPCA = mean(ISPCArates(1:end-1,:));
    avgISPCA_lab = mean(ISPCArates_lab(1:end-1,:));
    avgSPPCA = mean(SPPCArates(1:end-1,:));
    avgSPPCA_lab = mean(SPPCArates_lab(1:end-1,:));
    avgR4 = mean(ridge_rrr_rates(1:end-1,:,:), 1);
    avgR4_lab = mean(ridge_rrr_rates_lab(1:end-1,:,:), 1);
%     avgSSVD = mean(SSVDrates(1:end-1,:));
%     avgSSVD_lab = mean(SSVDrates_lab(1:end-1,:));
    
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
%     avgSSVDvar = mean(SSVDvar(1:end-1,:));
    
    avgPCAvar_lab = mean(PCAvar_lab(1:end-1,:));
    avgkPCAvar_lab = mean(kPCAvar_lab(1:end-1,:,:));
    avgPLSvar_lab = mean(PLSvar_lab(1:end-1,:));
    avgLSPCAvar_lab = mean(LSPCAvar_lab(1:end-1,:,:), 1);
    avgkLSPCAvar_lab = mean(kLSPCAvar_lab(1:end-1,:,:,:), 1);
    avgLSPCAmlevar_lab = mean(mle_LSPCAvar_lab(1:end-1,:,:), 1);
    avgkLSPCAmlevar_lab = mean(mle_kLSPCAvar_lab(1:end-1,:,:,:), 1);
    avgSPCAvar_lab = mean(SPCAvar_lab(1:end-1,:));
    avgkSPCAvar_lab = mean(kSPCAvar_lab(1:end-1,:,:));
    avgISPCAvar_lab = mean(ISPCAvar_lab(1:end-1,:));
    avgSPPCAvar_lab = mean(SPPCAvar_lab(1:end-1,:));
    avgR4var_lab = mean(ridge_rrrvars_lab(1:end-1,:,:), 1);
%     avgSSVDvar_lab = mean(SSVDvar_lab(1:end-1,:));
    
    %% Calc performance for best model and store
    
    % cv over subspace dim
    loc = find(avgPCA==min(avgPCA,[],'all'),1,'last');
    [~,kloc] = ind2sub(size(avgPCA), loc);
    kpca = ks(kloc);
    PCAval(dd) = PCArates(end,kloc);
    PCAvalVar(dd) = PCAvar(end,kloc);
    
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
    
%     loc = find(avgSSVD==min(avgSSVD,[],'all'),1,'last');
%     [~,kloc] = ind2sub(size(avgSSVD), loc);
%     kssvd = ks(kloc);
%     SSVDval(dd) = SSVDrates(end,kloc);
%     SSVDvalVar(dd) = SSVDvar(end,kloc);
    
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
    
%     kssvd = ks(kloc);
%     SSVDval_fixed(dd) = SSVDrates(end,kloc);
%     SSVDvalVar_fixed(dd) = SSVDvar(end,kloc);
    
    krrr = ks(kloc);
    loc = 1; % RRR with parameter value 0
    RRRval_fixed(dd) = ridge_rrr_rates(end,kloc,loc);
    RRRvalVar_fixed(dd) = ridge_rrrvars(end,kloc,loc);
    
    loc = find(avgR4(:,kloc,:)==min(avgR4(:,kloc,:),[],'all'),1,'last');
    [~,~,locr4] = ind2sub(size(avgR4(:,kloc,:)), loc);
    kr4 = ks(kloc);
    R4val_fixed(dd) = ridge_rrr_rates(end,kloc,locr4);
    R4valVar_fixed(dd) = ridge_rrrvars(end,kloc,locr4);
    
end

%% save all data
save(strcat(dataset, '_results_dim_ss'))

%% Print results over all subspace dimensions
m = mean(PCAval);
v = mean(PCAvalVar);
sm = std(PCAval);
sv = std(PCAvalVar);
sprintf('PCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)


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

% m = mean(SSVDval);
% v = mean(SSVDvalVar);
% sm = std(SSVDval);
% sv = std(SSVDvalVar);
% sprintf('SSVD: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

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

% m = mean(SSVDval_fixed);
% v = mean(SSVDvalVar_fixed);
% sm = std(SSVDval_fixed);
% sv = std(SSVDvalVar_fixed);
% sprintf('SSVD: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

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
% %
% %
% kloc = find(ks==klspca);
% t = kloc;
% temp = avgLSPCA(:,t,:,:);
% temp = reshape(temp, [length(Gammas), length(Lambdas)]);
% tempv = avgLSPCAvar(:,t,:,:);
% tempv = reshape(tempv, [length(Gammas), length(Lambdas)]);
% [mLSPCA, I] = min(temp);
% I = sub2ind(size(temp),I,1:length(I));
% vLSPCA = tempv(I);
% 
% 
% kloc = find(ks==kklspca);
% t = kloc;
% temp = avgkLSPCA(:,t,:,:);
% temp = reshape(temp, [length(Gammas), length(Lambdas), length(sigmas)]);
% tempv = avgkLSPCAvar(:,t,:,:);
% tempv = reshape(tempv, [length(Gammas), length(Lambdas), length(sigmas)]);
% [res, I1] = min(temp, [], 1);
% [~,a,b] = size(I1);
% for i=1:a
%     for j=1:b
%         %tempvv(i,j) = tempv(I1(1,i,j),i,j);
%         tempvv(i,j) = tempv(I1(1,i,j),i,j);
%     end
% end
% [mkLSPCA, I2] = min(res, [], 3);
% [~,c] = size(I2);
% for i=1:c
%     vkLSPCA(i) = tempvv(i,10);
% end
% 
% 
% 
% 
% % for t = 1:length(ks)
% figure()
% hold on
% plot(avgPCAvar(1,kpca-1), avgPCA(1,kpca-1), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% %plot(avgkPCAvar_lab(1,t), avgkPCA(1,t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% plot(vLSPCA(:), mLSPCA(:), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% % plot(lambda_avgLSPCAvar_lab(1,t,:), lambda_avgLSPCA(1,t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(vkLSPCA(:), mkLSPCA(:), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(avgISPCAvar(1,kispca-1), avgISPCA(1,kispca-1), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgSPPCAvar(1,ksppca-1), avgSPPCA(1,ksppca-1), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgSPCAvar(1,kspca-1), avgSPCA(1,kspca-1), '+', 'MarkerSize', 20, 'LineWidth', 2)
% loc = find(avgkSPCA(:,kkspca-1,:)==min(avgkSPCA(:,kkspca-1,:),[],'all'),1,'last');
% [~,~,sigloc] = ind2sub(size(avgkSPCA(:,kkspca-1,:)), loc);
% plot(avgkSPCAvar(1,kkspca-1,sigloc), avgkSPCA(1,kkspca-1,sigloc), '>', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgR4var(kr4-1,2:end), avgR4(kr4-1, 2:end), ':', 'LineWidth', 2, 'MarkerSize', 20)
% plot(avgR4var(krrr-1,1), avgR4(krrr-1, 1), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgPLSvar(1,kpls), avgPLS(1,kpls), 'h', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgSSVDvar(1,kssvd), avgSSVD(1,kssvd), 'd', 'MarkerSize', 20, 'LineWidth', 2)
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
% %% add MLE results to plot
% avgLSPCA = mean(LSPCArates, 1);
% avgLSPCA_lab = mean(LSPCArates_lab, 1);
% avgkLSPCA = mean(kLSPCArates, 1);
% avgkLSPCA_lab = mean(kLSPCArates_lab, 1);
% avgLSPCAvar = mean(LSPCAvar, 1);
% avgkLSPCAvar = mean(kLSPCAvar, 1);
% avgLSPCAvar_lab = mean(LSPCAvar_lab, 1);
% avgkLSPCAvar_lab = mean(kLSPCAvar_lab, 1);
% hold on
% plot(avgLSPCAvar(klspca), avgLSPCA(klspca), 'b*', 'LineWidth', 2, 'MarkerSize', 20)
% [m, loc] = min(avgkLSPCA(1,kklspca,:));
% plot(avgkLSPCAvar(1,kklspca,loc), m, 'r*', 'LineWidth', 2, 'MarkerSize', 20)
% lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'LSPCA (CV)', 'kLSPCA (CV)', 'Location', 'best'); lgd.FontSize = 15;
% 
% 
% %% labing error and var
% t = find(ks==klspca);
% temp = avgLSPCA(:,t,:,:);
% temp = reshape(temp, [length(Gammas), length(Lambdas)]);
% temp_lab = reshape(avgLSPCA_lab(:,t,:,:), [length(Gammas), length(Lambdas)]);
% tempv = avgLSPCAvar(:,t,:,:);
% tempv = reshape(tempv, [length(Gammas), length(Lambdas)]);
% tempv_lab = reshape(avgLSPCAvar_lab(:,t,:,:), [length(Gammas), length(Lambdas)]);
% [mLSPCA, I] = min(temp);
% I = sub2ind(size(temp),I,1:length(I));
% vLSPCA = tempv(I);
% mLSPCA_lab = temp_lab(I);
% vLSPCA_lab = tempv_lab(I);
% 
% t = find(ks==kklspca);
% temp = avgkLSPCA(:,t,:,:);
% temp = reshape(temp, [length(Gammas), length(Lambdas), length(sigmas)]);
% temp_lab = reshape(avgkLSPCA_lab(:,t,:,:), [length(Gammas), length(Lambdas), length(sigmas)]);
% tempv = avgkLSPCAvar(:,t,:,:);
% tempv = reshape(tempv, [length(Gammas), length(Lambdas), length(sigmas)]);
% tempv_lab = reshape(avgkLSPCAvar_lab(:,t,:,:), [length(Gammas), length(Lambdas), length(sigmas)]);
% [res, I1] = min(temp, [], 1);
% [~,a,b] = size(I1);
% for i=1:a
%     for j=1:b
%         tempvv(i,j) = tempv(I1(1,i,j),i,j);
%         tempmm_lab(i,j) = temp_lab(I1(1,i,j),i,j);
%         tempvv_lab(i,j) = tempv_lab(I1(1,i,j),i,j);
%     end
% end
% [mkLSPCA, I2] = min(res, [], 3);
% [~,c] = size(I2);
% for i=1:c
%     vkLSPCA(i) = tempvv(i,I2(1,i));
%     mkLSPCA_lab(i) = tempmm_lab(i,10);
%     vkLSPCA_lab(i) = tempvv_lab(i,10);
% end
% 
% 
% % for t = 1:length(ks)
% figure()
% hold on
% plot(avgPCAvar_lab(1,kpca-1), avgPCA_lab(1,kpca-1), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% %plot(avgkPCAvar_lab(1,t), avgkPCA(1,t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% plot(vLSPCA_lab(:), mLSPCA_lab(:), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% % plot(lambda_avgLSPCAvar_lab(1,t,:), lambda_avgLSPCA(1,t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(vkLSPCA_lab(:), mkLSPCA_lab(:), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(avgISPCAvar_lab(1,kispca-1), avgISPCA_lab(1,kispca-1), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgSPPCAvar_lab(1,ksppca-1), avgSPPCA_lab(1,ksppca-1), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgSPCAvar_lab(1,kspca-1), avgSPCA_lab(1,kspca-1), '+', 'MarkerSize', 20, 'LineWidth', 2)
% loc = find(avgkSPCA(:,kkspca-1,:)==min(avgkSPCA(:,kkspca-1,:),[],'all'),1,'last');
% [~,~,sigloc] = ind2sub(size(avgkSPCA(:,kkspca-1,:)), loc);
% plot(avgkSPCAvar_lab(1,kkspca-1,sigloc), avgkSPCA_lab(1,kkspca-1,sigloc), '>', 'MarkerSize', 20, 'LineWidth', 2)
% x=avgR4var_lab(1,kr4-1,2:end); y = avgR4_lab(1,kr4-1, 2:end);
% plot(x(:), y(:), ':', 'LineWidth', 2, 'MarkerSize', 20)
% plot(avgR4var_lab(1,krrr-1,1), avgR4_lab(1,krrr-1, 1), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgPLSvar_lab(1,kpls-1), avgPLS_lab(1,kpls-1), 'h', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgSSVDvar_lab(1,kssvd-1), avgSSVD_lab(1,kssvd-1), 'd', 'MarkerSize', 20, 'LineWidth', 2)
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
% avgLSPCA_lab = mean(LSPCArates_lab, 1);
% avgkLSPCA = mean(kLSPCArates, 1);
% avgkLSPCA_lab = mean(kLSPCArates_lab, 1);
% avgLSPCAvar = mean(LSPCAvar, 1);
% avgkLSPCAvar = mean(kLSPCAvar, 1);
% avgLSPCAvar_lab = mean(LSPCAvar_lab, 1);
% avgkLSPCAvar_lab = mean(kLSPCAvar_lab, 1);
% hold on
% plot(avgLSPCAvar_lab(klspca-1), avgLSPCA_lab(klspca-1), 'b*', 'LineWidth', 2, 'MarkerSize', 20)
% [m, loc] = min(avgkLSPCA_lab(1,kklspca-1,:));
% plot(avgkLSPCAvar_lab(1,kklspca-1,loc), m, 'r*', 'LineWidth', 2, 'MarkerSize', 20)
% lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'LSPCA (CV)', 'kLSPCA (CV)', 'Location', 'best'); lgd.FontSize = 15;
% 
% % end
% % saveas(gcf, strcat(dataset, 'multi_obj_gamma.jpg'))
% %
% %
% % %% plot labing error - var tradeoff curves
% % for t = 1:length(ks)
% %     figure()
% %     hold on
% %     plot(avgPCAvar_lab(t), avgPCA_lab(t), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% %     %    plot(avgkPCAvar_lab(t), avgkPCA_lab(t), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgLSPCAvar_lab(t,:,lamloc), avgLSPCA_lab(t, :,lamloc), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% %     plot(avgkLSPCAvar_lab(t,:,lamlock,siglock), avgkLSPCA_lab(t, :,lamlock,siglock), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% %     plot(avgISPCAvar_lab(t), avgISPCA_lab(t), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgSPPCAvar_lab(t), avgSPPCA_lab(t), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgSPCAvar_lab(t), avgSPCA_lab(t), '+', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgkSPCAvar_lab(t), avgkSPCA_lab(t), '>', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgR4var_lab(t,2:end), avgR4_lab(t, 2:end), ':', 'LineWidth', 2, 'MarkerSize', 20)
% %     plot(avgR4var_lab(t,1), avgR4_lab(t, 1), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgPLSvar_lab(t), avgPLS_lab(t), 'h', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgSSVDvar_lab(t), avgSSVD_lab(t), 'd', 'MarkerSize', 20, 'LineWidth', 2)
% %
% %     xlabel('Variation Explained', 'fontsize', 25)
% %     %title('lab', 'fontsize', 25)
% %     ylabel('MSE', 'fontsize', 25)
% %     %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
% %     set(gca, 'fontsize', 25)
% %     lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'Location', 'best'); lgd.FontSize = 15;
% %     %ylim([0, 0.12])
% %     %set(gca, 'YScale', 'log')
% %     xlim([0,1])
% % end
% % saveas(gcf, strcat(dataset, 'multi_obj_lab_gamma.jpg'))
% %
% %
% %
% 
% 


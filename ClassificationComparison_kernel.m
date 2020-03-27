numExps = 10;
for dd = 1:numExps
    %% setup and load data
    dd
    load(strcat(dataset, '.mat'));
    [n, p] = size(X);
    [~, q] = size(Y);
    ks = 2:min(10, p-1);
    
    %holdout an independent test set
    proportion = 0.2;
    nhold = floor(n*proportion); % 20%
    idxhold = ~crossvalind('HoldOut',Y,proportion,'classes', unique(Y), 'min', 2);
    Xhold = X(idxhold, :);
    X = X(~idxhold, :);
    Yhold = Y(idxhold, :);
    Y = Y(~idxhold, :);
    n = n - nhold; %this is our new n
    
    % create same splits to use every time and center
    kfold = 10;
    cvidx = crossvalind('kfold',Y,kfold,'classes', unique(Y), 'min', 2);
    for l = 1:kfold
        Xtrain = X(cvidx~=l, :); %lth training set
        Ytrain = Y(cvidx~=l, :);
        Xtest = X(cvidx==l, :); %lth testing set
        Ytest = Y(cvidx==l, :);
        [Xtrain,Xtest,Ytrain,Ytest] = center_data(Xtrain,Xtest,Ytrain,Ytest,'classification');
        Xtrains{l} = Xtrain; %lth centered training set
        Ytrains{l} = Ytrain;
        Xtests{l} = Xtest; %lth centered testing set
        Ytests{l} = Ytest;
    end
    
    % store the independent test set, and corresponding training set (non holdout data)
    % at the end of the train  and test cell arrays for convenience of
    % implementation
    [Xtrain,Xtest,Ytrain,Ytest] = center_data(X,Xhold,Y,Yhold,'classification');
    Xtrains{kfold+1} = Xtrain; %lth centered training set
    Ytrains{kfold+1} = Ytrain;
    Xtests{kfold+1} = Xtest; %lth centered testing set
    Ytests{kfold+1} = Ytest;
    % this way we will end up evaluating the independent set for every possible
    % model, which will take longer, but this is just easier. At the end, we
    % will choose the best model by looking only at the tests corresponding to
    % l = 1:kfold, and l=kfold+1 will give us the independent test error
    % corresponding to that model
    
    % p = gcp('nocreate'); % If no pool, do not create new one.
    % if isempty(p)
    %     parpool(6)
    % end
    %delete(gcp('nocreate'))
    for t = 1:length(ks) %dimensionality of reduced data
        k = ks(t)
        
        
        
        %% kLRPCA
        Lambdas = [linspace(1, 0.5, 11), linspace(0.4, 0, 5)];
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
                sigma = sigmas(kk);
                for ii = 1:length(Lambdas)
                    Lambda = Lambdas(ii);
                    if ii == 1
                        [ Zklspca, Lorth, B, Klspca] = klrpca(Xtrain, Ytrain, Lambda, sigma, k, 0, 0);
                    else
                        [ Zklspca, Lorth, B, Klspca] = klrpca(Xtrain, Ytrain, Lambda, sigma, k, Lorth, Klspca);
                    end
                    %Lorth = Lorth';
                    embedFunc = @(data) klspca_embed(data, Xtrain, Lorth, sigma);
                    kLs{l,t,ii,kk} = Lorth;
                    kLSPCAXtest = embedFunc(Xtest);
                    klspca_mbd_test{l,t,ii,kk} = kLSPCAXtest;
                    kLSPCAXtrain = Zklspca;
                    klspca_mbd_train{l,t,ii,kk} = Zklspca;
                    [~, kLSPCAYtest] = max(kLSPCAXtest*B(2:end,:) + B(1,:), [], 2);
                    [~, kLSPCAYtrain] = max(Zklspca*B(2:end,:) + B(1,:), [], 2);
                    err = 1 - sum(Ytest == kLSPCAYtest)/ntest;
                    train_err = 1 - sum(Ytrain == kLSPCAYtrain)/ntrain;
                    kLSPCArates(l,t,ii,kk) = err ;
                    kLSPCArates_train(l,t,ii,kk) = train_err;
                    kLSPCAvar(l,t,ii,kk) = norm(kLSPCAXtest, 'fro') / norm(gaussian_kernel(Xtest,Xtrain,sigma), 'fro');
                    kLSPCAvar_train(l,t,ii,kk) = norm(Zklspca, 'fro') / norm(Klspca, 'fro');
                end
                
            end
        end
        
        %% kLRPCA (MLE)
        for l = 1:kfold+1
            test_num = l;
            % get lth fold
            Xtrain = Xtrains{l};
            Ytrain = Ytrains{l};
            [ntrain, ~] = size(Ytrain);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            for kk = 1:length(sigmas)
                sigma = sigmas(kk);
                [ Zklspca, Lorth, B, Klspca] = klrpca_MLE(Xtrain, Ytrain, sigma, k, 0, 0);
                embedFunc = @(data) klspca_embed(data, Xtrain, Lorth, sigma);
                mle_kLs{l,t,kk} = Lorth;
                kLSPCAXtest = embedFunc(Xtest);
                mle_klspca_mbd_test{l,t,kk} = kLSPCAXtest;
                kLSPCAXtrain = Zklspca;
                mle_klspca_mbd_train{l,t,kk} = Zklspca;
                [~, kLSPCAYtest] = max(kLSPCAXtest*B(2:end,:) + B(1,:), [], 2);
                [~, kLSPCAYtrain] = max(Zklspca*B(2:end,:) + B(1,:), [], 2);
                err = 1 - sum(Ytest == kLSPCAYtest)/ntest;
                train_err = 1 - sum(Ytrain == kLSPCAYtrain)/ntrain;
                mle_kLSPCArates(l,t,kk) = err ;
                mle_kLSPCArates_train(l,t,kk) = train_err;
                mle_kLSPCAvar(l,t,kk) = norm(kLSPCAXtest, 'fro') / norm(gaussian_kernel(Xtest,Xtrain,sigma), 'fro');
                mle_kLSPCAvar_train(l,t,kk) = norm(Zklspca, 'fro') / norm(Klspca, 'fro');
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
                kPCAtimes(l,t,jj) = toc;
                Zkpcas{l,t,jj} = Zkpca;
                Lkpca = Lkpca';
                % compute embedding for test data
                kpcaXtest = Ktest*Lkpca';
                kpcaXtests{l,t,jj} = kpcaXtest;
                % compute error
                B = mnrfit(Zkpca,Ytrain, 'Model', 'nominal', 'Interactions', 'on');
                [~, Yhat] = max(mnrval(B,kpcaXtest),[], 2);
                [~,kpcaYtrain] = max(mnrval(B,Zkpca),[], 2);
                kPCArates(l,t,jj) = 1 - sum(Yhat == Ytest) / ntest;
                kPCArates_train(l,t,jj) = 1 - sum(kpcaYtrain == Ytrain) / ntrain;
                kPCAvar(l,t,jj) = norm(kpcaXtest, 'fro') / norm(Ktest, 'fro');
                kPCAvar_train(l,t,jj) = norm(Zkpca, 'fro') / norm(K, 'fro');
            end
        end
            
            %% Perform Barshan's KSPCA 
            %learn basis
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
                    
                    tic
                    %calc with best param on full training set
                    barshparam.ktype_y = 'delta';
                    barshparam.kparam_y = 1;
                    barshparam.ktype_x = 'rbf';
                    barshparam.kparam_x = sigma;
                    [Zkspca Lkspca] = KSPCA(Xtrain', Ytrain', k, barshparam);
                    Zkspca = Zkspca';
                    %do prediction in learned basis
                    betakSPCA = Zkspca \ Ytrain;
                    Ktrain = gaussian_kernel(Xtrain, Xtrain, sigma);
                    Ktest = gaussian_kernel(Xtest, Xtrain, sigma);
                    kspcaXtest = Ktest*Lkspca;
                    kspca_mbd_test{l,t,jj} = kspcaXtest;
                    kspca_mbd_train{l,t,jj} = Zkspca;
                    B = mnrfit(Zkspca,Ytrain, 'Model', 'nominal', 'Interactions', 'on');
                    [~,Yhat] = max(mnrval(B,kspcaXtest),[], 2);
                    [~,kSPCAYtrain] = max(mnrval(B,Zkspca),[], 2);
                    %compute error
                    kSPCArates(l,t,jj) = 1 - sum(Yhat == Ytest) / ntest;
                    kSPCArates_train(l,t,jj) = 1 - sum(kSPCAYtrain == Ytrain) / ntrain;
                    kSPCAvar(l,t,jj) = norm(kspcaXtest, 'fro') / norm(Ktest, 'fro');
                    kSPCAvar_train(l,t,jj) = norm(Zkspca, 'fro') / norm(Ktrain, 'fro');
                    kBarshantimes(l,t,jj) = toc;
                    kspcaXtests{l,t,jj} = kspcaXtest;
                    Zkspcas{l,t,jj} = Zkspca;
                end
            end
            
            %% LDA
            % solve
            for l = 1:kfold+1
                test_num = l;
                % get lth fold
                Xtrain = Xtrains{l};
                Ytrain = Ytrains{l};
                [ntrain, ~] = size(Ytrain);
                Xtest = Xtests{l};
                Ytest = Ytests{l};
                [ntest, ~] = size(Ytest);
                Mdl = fitcdiscr(Xtrain,Ytrain, 'DiscrimType', 'pseudolinear');
                LDAYtest = predict(Mdl,Xtest);
                % Predict
                LDAYtrain = predict(Mdl,Xtrain);
                %compute error
                LDArates(l,t) = 1 - sum(LDAYtest == Ytest) / ntest;
                LDArates_train(l,t) = 1 - sum(LDAYtrain == Ytrain) / ntrain;
                lin = Mdl.Coeffs(1,2).Linear / norm([Mdl.Coeffs(1,2).Const; Mdl.Coeffs(1,2).Linear]);
                const = Mdl.Coeffs(1,2).Const / norm([Mdl.Coeffs(1,2).Const; Mdl.Coeffs(1,2).Linear]);
                Zlda = Xtrain*lin + const;
                LDAXtest = Xtest*lin + const;
                LDAvar(l,t) = norm(LDAXtest, 'fro') / norm(Xtest, 'fro');
                LDAvar_train(l,t) = norm(Zlda, 'fro') / norm(Xtrain, 'fro');
            end
            
            %% Local Fisher Discriminant Analysis (LFDA)
            for l = 1:kfold+1
                test_num = l;
                % get lth fold
                Xtrain = Xtrains{l};
                Ytrain = Ytrains{l};
                [ntrain, ~] = size(Ytrain);
                Xtest = Xtests{l};
                Ytest = Ytests{l};
                [ntest, ~] = size(Ytest);
                K = Xtrain*Xtrain';
                Ktest = Xtest*Xtrain';
                [Llfda,~] = KLFDA(K,Ytrain,k, 'plain',9);
                %predict
                Llfda = orth(Llfda);
                Zlfda = K*Llfda;
                B = mnrfit(Zlfda,Ytrain, 'Model', 'nominal', 'Interactions', 'on');
                LFDAXtest = Ktest*Llfda;
                [~,Yhat] = max(mnrval(B,LFDAXtest),[], 2);
                [~,LFDAYtrain] = max(mnrval(B,Zlfda),[], 2);
                %compute error
                LFDArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
                LFDArates_train(l,t) = 1 - sum(LFDAYtrain == Ytrain) / ntrain;
                
                LFDAvar(l,t) = norm(LFDAXtest, 'fro') / norm(Ktest, 'fro');
                LFDAvar_train(l,t) = norm(Zlfda, 'fro') / norm(K, 'fro');
            end
            
            %% Kernel Local Fisher Discriminant Analysis (KLFDA)
            %choose kernel param
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
                    %train
                    K = gaussian_kernel(Xtrain, Xtrain, sigma);
                    [Llfda,~] = KLFDA(K,Ytrain,k, 'plain');
                    Llfda = orth(Llfda);
                    Zlfda = K*Llfda;
                    %predict
                    Ktest = gaussian_kernel(Xtest, Xtrain, sigma);
                    LFDAXtest = Ktest*Llfda;
                    B = mnrfit(Zlfda,Ytrain, 'Model', 'nominal', 'Interactions', 'on');
                    [~,Yhat] = max(mnrval(B,LFDAXtest),[], 2);
                    [~,LFDAYtrain] = max(mnrval(B,Zlfda),[], 2);
                    %compute error
                    kLFDArates(l,t,jj) = 1 - sum(Yhat == Ytest) / ntest;
                    kLFDArates_train(l,t,jj) = 1 - sum(LFDAYtrain == Ytrain) / ntrain;
                    kLFDAvar(l,t,jj) = norm(LFDAXtest, 'fro') / norm(Ktest, 'fro');
                    kLFDAvar_train(l,t,jj) = norm(Zlfda, 'fro') / norm(K, 'fro');
                end
            end
            
            
        end
        

    
    %% compute avg performance accross folds
    
    
    avgkPCA = mean(kPCArates(1:end-1,:,:));
    avgkPCA_train = mean(kPCArates_train(1:end-1,:,:));
    avgkLSPCA = mean(kLSPCArates(1:end-1,:,:,:));
    avgkLSPCA_train = mean(kLSPCArates_train(1:end-1,:,:,:));
    avgkLSPCAmle = mean(mle_kLSPCArates(1:end-1,:,:), 1);
    avgkLSPCAmle_train = mean(mle_kLSPCArates_train(1:end-1,:,:), 1);
    avgkSPCA = mean(kSPCArates(1:end-1,:,:));
    avgkSPCA_train = mean(kSPCArates_train(1:end-1,:,:));
    avgLFDA = mean(LFDArates(1:end-1,:));
    avgLFDA_train = mean(LFDArates_train(1:end-1,:));
    avgkLFDA = mean(kLFDArates(1:end-1,:,:));
    avgkLFDA_train = mean(kLFDArates_train(1:end-1,:,:));
    %
    avgkPCAvar = mean(kPCAvar(1:end-1,:,:));
    avgkLSPCAvar = mean(kLSPCAvar(1:end-1,:,:,:));
    avgkLSPCAmlevar = mean(mle_kLSPCAvar(1:end-1,:,:), 1);
    avgkSPCAvar = mean(kSPCAvar(1:end-1,:,:));
    avgLFDAvar = mean(LFDAvar(1:end-1,:));
    avgkLFDAvar = mean(kLFDAvar(1:end-1,:,:));
    %
    avgkPCAvar_train = mean(kPCAvar_train(1:end-1,:,:));
    avgkLSPCAvar_train = mean(kLSPCAvar_train(1:end-1,:,:,:));
    avgkLSPCAmlevar_train = mean(mle_kLSPCAvar_train(1:end-1,:,:), 1);
    avgkSPCAvar_train = mean(kSPCAvar_train(1:end-1,:,:));
    avgLFDAvar_train = mean(LFDAvar_train(1:end-1,:));
    avgkLFDAvar_train = mean(kLFDAvar_train(1:end-1,:,:));
    
     %% Calc performance for best model and store
    
    % cv over subspace dim
    
    loc = find(avgkPCA==min(avgkPCA,[],'all'),1,'last');
    [~,kloc,sigloc] = ind2sub(size(avgkPCA), loc);
    kpca = ks(kloc);
    kPCAval(dd) = kPCArates(end,kloc,sigloc);
    kPCAvalVar(dd) = kPCAvar(end,kloc,sigloc);
    kPCAval_train(dd) = kPCArates_train(end,kloc,sigloc);
    kPCAvalVar_train(dd) = kPCAvar_train(end,kloc,sigloc);
    
    
    loc = find(avgkLSPCA==min(avgkLSPCA,[],'all'),1,'last');
    [~,klock,lamlock,siglock] = ind2sub(size(avgkLSPCA), loc);
    kklspca = ks(klock);
    kLSPCAval(dd) = kLSPCArates(end,klock,lamlock,siglock);
    kLSPCAvalVar(dd) = kLSPCAvar(end,klock,lamlock,siglock);
    kLSPCAval_train(dd) = kLSPCArates_train(end,klock,lamlock,siglock);
    kLSPCAvalVar_train(dd) = kLSPCAvar_train(end,klock,lamlock,siglock);
    

    loc = find(avgkLSPCAmle==min(avgkLSPCAmle,[],'all'),1,'last');
    [~,klock,siglock] = ind2sub(size(avgkLSPCAmle), loc);
    kklspca = ks(klock);
    mle_kLSPCAval(dd) = mle_kLSPCArates(end,klock,siglock);
    mle_kLSPCAvalVar(dd) = mle_kLSPCAvar(end,klock,siglock);
    mle_kLSPCAval_train(dd) = mle_kLSPCArates_train(end,klock,siglock);
    mle_kLSPCAvalVar_train(dd) = mle_kLSPCAvar_train(end,klock,siglock);
    
    
    loc = find(avgkSPCA==min(avgkSPCA,[],'all'),1,'last');
    [~,kloc,sigloc] = ind2sub(size(avgkSPCA), loc);
    kkspca = ks(kloc);
    kSPCAval(dd) = kSPCArates(end,kloc,sigloc);
    kSPCAvalVar(dd) = kSPCAvar(end,kloc,sigloc);
    kSPCAval_train(dd) = kSPCArates_train(end,kloc,sigloc);
    kSPCAvalVar_train(dd) = kSPCAvar_train(end,kloc,sigloc);
    
    
    loc = find(avgLFDA==min(avgLFDA,[],'all'),1,'last');
    [~,kloc] = ind2sub(size(avgLFDA), loc);
    klfda = ks(kloc);
    LFDAval(dd) = LFDArates(end,kloc);
    LFDAvalVar(dd) = LFDAvar(end,kloc);
    LFDAval_train(dd) = LFDArates_train(end,kloc);
    LFDAvalVar_train(dd) = LFDAvar_train(end,kloc);
    
    loc = find(avgkLFDA==min(avgkLFDA,[],'all'),1,'last');
    [~,kloc,sigloc] = ind2sub(size(avgkLFDA), loc);
    kklfda = ks(kloc);
    kLFDAval(dd) = kLFDArates(end,kloc,sigloc);
    kLFDAvalVar(dd) = kLFDAvar(end,kloc,sigloc);
    kLFDAval_train(dd) = kLFDArates_train(end,kloc,sigloc);
    kLFDAvalVar_train(dd) = kLFDAvar_train(end,kloc,sigloc);

    %fixed subspace dimension k=2
    
    kloc=1; %k=2
    

    loc = find(avgkPCA(:,kloc,:)==min(avgkPCA(:,kloc,:),[],'all'),1,'last');
    [~,~,sigloc] = ind2sub(size(avgkPCA(:,kloc,:)), loc);
    kpca = ks(kloc);
    kPCAval_fixed(dd) = kPCArates(end,kloc,sigloc);
    kPCAvalVar_fixed(dd) = kPCAvar(end,kloc,sigloc);
    kPCAval_fixed_train(dd) = kPCArates_train(end,kloc,sigloc);
    kPCAvalVar_fixed_train(dd) = kPCAvar_train(end,kloc,sigloc);
    
    
    loc = find(avgkLSPCA(:,kloc,:,:)==min(avgkLSPCA(:,kloc,:,:),[],'all'),1,'last');
    [~,~,lamlock,siglock] = ind2sub(size(avgkLSPCA(:,kloc,:,:)), loc);
    kklspca = ks(kloc);
    kLSPCAval(dd) = kLSPCArates(end,kloc,lamlock,siglock);
    kLSPCAvalVar(dd) = kLSPCAvar(end,kloc,lamlock,siglock);
    kLSPCAval_train(dd) = kLSPCArates_train(end,kloc,lamlock,siglock);
    kLSPCAvalVar_train(dd) = kLSPCAvar_train(end,kloc,lamlock,siglock);
    
    loc = find(avgkLSPCAmle(:,kloc,:)==min(avgkLSPCAmle(:,kloc,:),[],'all'),1,'last');
    [~,~,sigloc] = ind2sub(size(avgkLSPCAmle(:,kloc,:)), loc);
    kklspca = ks(kloc);
    mle_kLSPCAval_fixed(dd) = mle_kLSPCArates(end,kloc,sigloc);
    mle_kLSPCAvalVar_fixed(dd) = mle_kLSPCAvar(end,kloc,sigloc);
    mle_kLSPCAval_fixed_train(dd) = mle_kLSPCArates_train(end,kloc,sigloc);
    mle_kLSPCAvalVar_fixed_train(dd) = mle_kLSPCAvar_train(end,kloc,sigloc);
    
    
    loc = find(avgkSPCA(:,kloc,:)==min(avgkSPCA(:,kloc,:),[],'all'),1,'last');
    [~,~,sigloc] = ind2sub(size(avgkSPCA(:,kloc,:)), loc);
    kkspca = ks(kloc);
    kSPCAval_fixed(dd) = kSPCArates(end,kloc,sigloc);
    kSPCAvalVar_fixed(dd) = kSPCAvar(end,kloc,sigloc);
    kSPCAval_fixed_train(dd) = kSPCArates_train(end,kloc,sigloc);
    kSPCAvalVar_fixed_train(dd) = kSPCAvar_train(end,kloc,sigloc);
    
    
    klfda = ks(kloc);
    LFDAval_fixed(dd) = LFDArates(end,kloc);
    LFDAvalVar_fixed(dd) = LFDAvar(end,kloc);
    LFDAval_fixed_train(dd) = LFDArates_train(end,kloc);
    LFDAvalVar_fixed_train(dd) = LFDAvar_train(end,kloc);
    
    loc = find(avgkLFDA==min(avgkLFDA,[],'all'),1,'last');
    [~,~,sigloc] = ind2sub(size(avgkLFDA), loc);
    kklfda = ks(kloc);
    kLFDAval_fixed(dd) = kLFDArates(end,kloc,sigloc);
    kLFDAvalVar_fixed(dd) = kLFDAvar(end,kloc,sigloc);
    kLFDAval_fixed_train(dd) = kLFDArates_train(end,kloc,sigloc);
    kLFDAvalVar_fixed_train(dd) = kLFDAvar_train(end,kloc,sigloc);
    
    %track vals from all exps
    kLSPCAval_track(dd,:,:,:,:) = kLSPCArates;
    kLSPCAvalVar_track(dd,:,:,:,:) = kLSPCAvar;
    kLSPCAval_track_train(dd,:,:,:,:) = kLSPCArates_train;
    kLSPCAvalVar_track_train(dd,:,:,:,:) = kLSPCAvar_train;
    
end
%% save all data
save(strcat(dataset, '_results_dim'))

%% print mean performance with std errors


m = mean(LFDAval);
v = mean(LFDAvalVar);
sm = std(LFDAval);
sv = std(LFDAvalVar);
sprintf('LFDAerr: $%0.3f \\pm %0.3f $ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(kPCAval);
v = mean(kPCAvalVar);
sm = std(kPCAval);
sv = std(kPCAvalVar);
sprintf('kPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(kSPCAval);
v = mean(kSPCAvalVar);
sm = std(kSPCAval);
sv = std(kSPCAvalVar);
sprintf('kBarshanerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(kLFDAval);
v = mean(kLFDAvalVar);
sm = std(kLFDAval);
sv = std(kLFDAvalVar);
sprintf('kLFDAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

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

m = mean(LFDAval_fixed);
v = mean(LFDAvalVar_fixed);
sm = std(LFDAval_fixed);
sv = std(LFDAvalVar_fixed);
sprintf('LFDAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(kPCAval_fixed);
v = mean(kPCAvalVar_fixed);
sm = std(kPCAval_fixed);
sv = std(kPCAvalVar_fixed);
sprintf('kPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(kSPCAval_fixed);
v = mean(kSPCAvalVar_fixed);
sm = std(kSPCAval_fixed);
sv = std(kSPCAvalVar_fixed);
sprintf('kBarshanerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(kLFDAval_fixed);
v = mean(kLFDAvalVar_fixed);
sm = std(kLFDAval_fixed);
sv = std(kLFDAvalVar_fixed);
sprintf('kLFDAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(kLSPCAval_fixed);
v = mean(kLSPCAvalVar_fixed);
sm = std(kLSPCAval_fixed);
sv = std(kLSPCAvalVar_fixed);
sprintf('kLSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(mle_kLSPCAval_fixed);
v = mean(mle_kLSPCAvalVar_fixed);
sm = std(mle_kLSPCAval_fixed);
sv = std(mle_kLSPCAvalVar_fixed);
sprintf('mle_kLSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)



% %% plot error - var tradeoff curves
% 
% % for t = 1:length(ks)
% figure()
% hold on
% plot(mean(PCAvalVar_fixed), mean(PCAval_fixed), 'sr', 'MarkerSize', 30, 'LineWidth', 2)
% %plot(mean(kPCAvalVar_fixed), mean(kPCAval_fixed), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% plot(squeeze(mean(LSPCAvalVar_track(:,end,1,:))), squeeze(mean(LSPCAval_track(:,end,1,:))), 'r.:', 'LineWidth', 2, 'MarkerSize', 20)
% plot(squeeze(mean(kLSPCAvalVar_track(:,end,1,:,end))), squeeze(mean(kLSPCAval_track(:,end,1,:,end))), 'b.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(mle_LSPCAvalVar_fixed), mean(mle_LSPCAval_fixed), 'r>', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(mle_kLSPCAvalVar_fixed), mean(mle_kLSPCAval_fixed), 'b^', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(ISPCAvalVar_fixed), mean(ISPCAval_fixed), 'm+', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(SPPCAvalVar_fixed), mean(SPPCAval_fixed), 'xc', 'MarkerSize', 20, 'LineWidth', 3)
% plot(mean(SPCAvalVar_fixed), mean(SPCAval_fixed), 'pk', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(kSPCAvalVar_fixed), mean(kSPCAval_fixed), '<', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(LDAvalVar_fixed), mean(LDAval_fixed), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(LFDAvalVar_fixed), mean(LFDAval_fixed), 'h', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(kLFDAvalVar_fixed), mean(kLFDAval_fixed), 'ok', 'MarkerSize', 20, 'LineWidth', 3)
% 
% xlabel('Variation Explained', 'fontsize', 25)
% %title('Test', 'fontsize', 25)
% ylabel('MSE', 'fontsize', 25)
% %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
% set(gca, 'fontsize', 25)
% lgd = legend('PCC', 'LRPCA (CV)', 'kLRPCA (CV)', 'LRPCA (MLE)', 'kLRPCA (MLE)', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'LDA', 'LFDA', 'kLFDA', 'Location', 'best'); lgd.FontSize = 15;
% %lgd = legend('LSPCA', 'R4', 'PLS', 'SPPCA', 'Barshan', 'SSVD', 'PCA', 'Location', 'southeast'); lgd.FontSize = 15;
% %ylim([0, 0.12])
% %set(gca, 'YScale', 'log')
% xlim([0,1])
% 
% 
% %% plot error - var tradeoff curves
% 
% % for t = 1:length(ks)
% figure()
% hold on
% plot(mean(PCAvalVar_fixed_train), mean(PCAval_fixed_train), 'sr', 'MarkerSize', 30, 'LineWidth', 2)
% % plot(mean(kPCAvalVar_fixed), mean(kPCAval_fixed), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% plot(squeeze(mean(LSPCAvalVar_track_train(:,end,1,:))), squeeze(mean(LSPCAval_track_train(:,end,1,:))), 'r.:', 'LineWidth', 2, 'MarkerSize', 20)
% plot(squeeze(mean(kLSPCAvalVar_track_train(:,end,1,:,end))), squeeze(mean(kLSPCAval_track_train(:,end,1,:,end))), 'b.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(mle_LSPCAvalVar_fixed_train), mean(mle_LSPCAval_fixed_train), 'r>', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(mle_kLSPCAvalVar_fixed_train), mean(mle_kLSPCAval_fixed_train), 'b^', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(ISPCAvalVar_fixed_train), mean(ISPCAval_fixed_train), 'm+', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(SPPCAvalVar_fixed_train), mean(SPPCAval_fixed_train), 'xc', 'MarkerSize', 20, 'LineWidth', 3)
% plot(mean(SPCAvalVar_fixed_train), mean(SPCAval_fixed_train), 'pk', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(kSPCAvalVar_fixed_train), mean(kSPCAval_fixed_train), '<', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(LDAvalVar_fixed_train), mean(LDAval_fixed_train), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(LFDAvalVar_fixed_train), mean(LFDAval_fixed_train), 'h', 'MarkerSize', 20, 'LineWidth', 2)
% plot(mean(kLFDAvalVar_fixed_train), mean(kLFDAval_fixed_train), 'ok', 'MarkerSize', 20, 'LineWidth', 3)
% 
% xlabel('Variation Explained', 'fontsize', 25)
% %title('Test', 'fontsize', 25)
% ylabel('MSE', 'fontsize', 25)
% %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
% set(gca, 'fontsize', 25)
% lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'LSPCA (MLE)', 'kLSPCA (MLE)', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'LDA', 'LFDA', 'kLFDA', 'Location', 'best'); lgd.FontSize = 15;
% %lgd = legend('LSPCA', 'R4', 'PLS', 'SPPCA', 'Barshan', 'SSVD', 'PCA', 'Location', 'southeast'); lgd.FontSize = 15;
% %ylim([0, 0.12])
% %set(gca, 'YScale', 'log')
% xlim([0,1])
% 
% % %% plot error - var tradeoff curves
% % %avgLSPCA_train = avgLSPCA_train(:,:,1:5,:);
% % %avgLSPCAvar_train = avgLSPCAvar_train(:,:,1:5,:);
% % 
% % kloc = find(ks==klspca);
% % t = kloc;
% % temp = avgLSPCA(:,t,:,:);
% % temp = reshape(temp, [length(Gammas), length(Lambdas)]);
% % %temp_train = reshape(avgLSPCA_train(:,t,:,:), [length(Gammas), length(Lambdas)]);
% % tempv = avgLSPCAvar(:,t,:,:);
% % tempv = reshape(tempv, [length(Gammas), length(Lambdas)]);
% % %tempv_train = reshape(avgLSPCAvar_train(:,t,:,:), [length(Gammas), length(Lambdas)]);
% % [mLSPCA, I] = min(temp);
% % I = sub2ind(size(temp),I,1:length(I));
% % vLSPCA = tempv(I);
% % %mLSPCA_train = temp_train(I);
% % %vLSPCA_train = tempv_train(I);
% % 
% % kloc = find(ks==kklspca);
% % t = kloc;
% % temp = avgkLSPCA(:,t,:,:,:);
% % temp = reshape(temp, [length(Gammas), length(Lambdas), length(sigmas)]);
% % %temp_train = reshape(avgkLSPCA_train(:,t,:,:), [length(Gammas), length(Lambdas), length(sigmas)]);
% % tempv = avgkLSPCAvar(:,t,:,:,:);
% % tempv = reshape(tempv, [length(Gammas), length(Lambdas), length(sigmas)]);
% % %tempv_train = reshape(avgkLSPCAvar_train(:,t,:,:), [length(Gammas), length(Lambdas), length(sigmas)]);
% % [res, I1] = min(temp, [], 1);
% % [~,a,b] = size(I1);
% % for i=1:a
% %     for j=1:b
% %         tempvv(i,j) = tempv(I1(1,i,j),i,j);
% %         %    tempmm_train(i,j) = temp_train(I1(1,i,j),i,j);
% %         %    tempvv_train(i,j) = tempv_train(I1(1,i,j),i,j);
% %     end
% % end
% % [mkLSPCA, I2] = min(res, [], 3);
% % [~,c] = size(I2);
% % for i=1:c
% %     vkLSPCA(i) = tempvv(i,10);
% %     %     mkLSPCA_train(i) = tempmm_train(i,I2(1,i));
% %     %     vkLSPCA_train(i) = tempvv_train(i,I2(1,i));
% % end
% % %     gamloc = 3;
% % %     gamlock = 3;
% % 
% % kloc = 1;
% % 
% % figure()
% % hold on
% % plot(avgPCAvar(1,kpca-1), avgPCA(1,kpca-1), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% % %plot(avgkPCAvar_train(1,t), avgkPCA(1,t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% % m = LSPCArates(end,kloc,:); m = m(:); v = LSPCAvar(end,kloc,:); v = v(:);
% % plot(v,m, '.-', 'LineWidth', 2, 'MarkerSize', 20)
% % % plot(lambda_avgLSPCAvar_train(1,t,:), lambda_avgLSPCA(1,t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% % m = kLSPCArates(end,kloc,:); m = m(:); v = kLSPCAvar(end,kloc,:); v = v(:);
% % plot(v, m, '.-', 'LineWidth', 2, 'MarkerSize', 20)
% % plot(avgISPCAvar(1,kispca-1), avgISPCA(1,kispca-1), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(avgSPPCAvar(1,ksppca-1), avgSPPCA(1,ksppca-1), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(avgSPCAvar(1,kspca-1), avgSPCA(1,kspca-1), '+', 'MarkerSize', 20, 'LineWidth', 2)
% % loc = find(avgkSPCA(:,kkspca-1,:)==min(avgkSPCA(:,kkspca-1,:),[],'all'),1,'last');
% % [~,~,sigloc] = ind2sub(size(avgkSPCA(:,kkspca-1,:)), loc);
% % plot(avgkSPCAvar(1,kkspca-1,sigloc), avgkSPCA(1,kkspca-1,sigloc), '>', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(avgLDAvar(klda-1), avgLDA(klda-1), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% % % plot([0,1], [1,1]*avgQDA(t), ':', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(avgLFDAvar(klfda-1), avgLFDA(klfda-1), '^', 'MarkerSize', 20, 'LineWidth', 2)
% % loc = find(avgkLFDA(:,kklfda-1,:)==min(avgkLFDA(:,kklfda-1,:),[],'all'),1,'last');
% % [~,~,sigloc] = ind2sub(size(avgkLFDA(:,kklfda-1,:)), loc);
% % plot(avgkLFDAvar(1,kklfda-1,sigloc), avgkLFDA(1,kklfda-1,sigloc), '<', 'MarkerSize', 20, 'LineWidth', 2)
% % 
% % %plot([0,1], avgLR*[1,1], 'k', 'LineWidth', 4)
% % 
% % %xlabel('Variation Explained', 'fontsize', 25)
% % %title('Test', 'fontsize', 25)
% % %ylabel('Classification Error', 'fontsize', 25)
% % %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
% % set(gca, 'fontsize', 25)
% % lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', ...
% %     'LDA', 'LFDA', 'kLFDA', 'Location', 'best'); lgd.FontSize = 25;
% % xlim([0,1.01])
% % %ylim([0,0.5])
% % %saveas(gcf, strcat(dataset, 'multi_obj.jpg'))
% % 
% % %% add MLE results to plot
% % avgLSPCA = mean(LSPCArates, 1);
% % avgLSPCA_train = mean(LSPCArates_train, 1);
% % avgkLSPCA = mean(kLSPCArates, 1);
% % avgkLSPCA_train = mean(kLSPCArates_train, 1);
% % avgLSPCAvar = mean(LSPCAvar, 1);
% % avgkLSPCAvar = mean(kLSPCAvar, 1);
% % avgLSPCAvar_train = mean(LSPCAvar_train, 1);
% % avgkLSPCAvar_train = mean(kLSPCAvar_train, 1);
% % 
% % 
% % %%
% % hold on
% % loc = find(avgLSPCA == min(avgLSPCA, [], 'all'));
% % plot(avgLSPCAvar(loc), avgLSPCA(loc), 'b*', 'LineWidth', 2, 'MarkerSize', 20)
% % %%
% % hold on
% % loc = find(avgkLSPCA == min(avgkLSPCA, [], 'all'), 1, 'last');
% % plot(avgkLSPCAvar(loc), avgkLSPCA(loc), 'r*', 'LineWidth', 2, 'MarkerSize', 20)
% % lgd = legend('PCA', 'LSPCA (CV)', 'kLSPCA (CV)', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', ...
% %     'LDA', 'LFDA', 'kLFDA', 'LSPCA (MLE)', 'kLSPCA (MLE)', 'Location', 'best'); lgd.FontSize = 25;
% % 
% % 
% % %
% % %% plot error - var tradeoff curves
% % kloc = 1;
% % t = kloc;
% % temp = avgLSPCA(:,t,:,:);
% % temp = reshape(temp, [length(Gammas), length(Lambdas)]);
% % temp_train = reshape(avgLSPCA_train(:,t,:,:), [length(Gammas), length(Lambdas)]);
% % tempv = avgLSPCAvar(:,t,:,:);
% % tempv = reshape(tempv, [length(Gammas), length(Lambdas)]);
% % tempv_train = reshape(avgLSPCAvar_train(:,t,:,:), [length(Gammas), length(Lambdas)]);
% % [mLSPCA, I] = min(temp);
% % I = sub2ind(size(temp),I,1:length(I));
% % vLSPCA = tempv(I);
% % mLSPCA_train = temp_train(I);
% % vLSPCA_train = tempv_train(I);
% % 
% % 
% % temp = avgkLSPCA(:,t,:,:);
% % temp = reshape(temp, [length(Gammas), length(Lambdas), length(sigmas)]);
% % temp_train = reshape(avgkLSPCA_train(:,t,:,:), [length(Gammas), length(Lambdas), length(sigmas)]);
% % tempv = avgkLSPCAvar(:,t,:,:);
% % tempv = reshape(tempv, [length(Gammas), length(Lambdas), length(sigmas)]);
% % tempv_train = reshape(avgkLSPCAvar_train(:,t,:,:), [length(Gammas), length(Lambdas), length(sigmas)]);
% % [res, I1] = min(temp, [], 1);
% % [~,a,b] = size(I1);
% % for i=1:a
% %     for j=1:b
% %         tempvv(i,j) = tempv(I1(1,i,j),i,j);
% %         tempmm_train(i,j) = temp_train(I1(1,i,j),i,j);
% %         tempvv_train(i,j) = tempv_train(I1(1,i,j),i,j);
% %     end
% % end
% % [mkLSPCA, I2] = min(res, [], 3);
% % [~,c] = size(I2);
% % for i=1:c
% %     vkLSPCA(i) = tempvv(i,I2(1,i));
% %     mkLSPCA_train(i) = tempmm_train(i,I2(1,i));
% %     vkLSPCA_train(i) = tempvv_train(i,I2(1,i));
% % end
% % %
% % % for t = 1:length(ks)
% % figure()
% % hold on
% % plot(avgPCAvar_train(1,kpca-1), avgPCA_train(1,kpca-1), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% % %plot(avgkPCAvar_train(1,t), avgkPCA(1,t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(vLSPCA_train(:), mLSPCA_train(:), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% % % plot(lambda_avgLSPCAvar_train(1,t,:), lambda_avgLSPCA(1,t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% % plot(vkLSPCA_train(:), mkLSPCA_train(:), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% % plot(avgISPCAvar_train(1,kispca-1), avgISPCA_train(1,kispca-1), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(avgSPPCAva_trainr(1,ksppca-1), avgSPPCA_train(1,ksppca-1), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(avgSPCAvar_train(1,kspca-1), avgSPCA_train(1,kspca-1), '+', 'MarkerSize', 20, 'LineWidth', 2)
% % loc = find(avgkSPCA(:,kkspca-1,:)==min(avgkSPCA(:,kkspca-1,:),[],'all'),1,'last');
% % [~,~,sigloc] = ind2sub(size(avgkSPCA(:,kkspca-1,:)), loc);
% % plot(avgkSPCAvar_train(1,kkspca-1,sigloc), avgkSPCA_train(1,kkspca-1,sigloc), '>', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(avgLDAvar_train(klda), avgLDA_train(klda), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% % % plot([0,1], [1,1]*avgQDA(t), ':', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(avgLFDAvar_train(klfda), avgLFDA_train(klfda), '^', 'MarkerSize', 20, 'LineWidth', 2)
% % loc = find(avgkLFDA(:,klfda,:)==min(avgkLFDA(:,klfda,:),[],'all'),1,'last');
% % [~,~,sigloc] = ind2sub(size(avgkLFDA(:,klfda,:)), loc);
% % plot(avgkLFDAvar_train(1,klfda,sigloc), avgkLFDA_train(1,klfda,sigloc), '<', 'MarkerSize', 20, 'LineWidth', 2)
% % 
% % %plot([0,1], avgLR*[1,1], 'k', 'LineWidth', 4)
% % 
% % xlabel('Variation Explained', 'fontsize', 25)
% % %title('Train', 'fontsize', 25)
% % ylabel('Classification Error', 'fontsize', 25)
% % %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
% % set(gca, 'fontsize', 25)
% % lgd = legend('PCA', 'LSPCA', 'KLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', ...
% %     'LDA', 'LFDA', 'kLFDA', 'Location', 'best'); lgd.FontSize = 15;
% % 
% % %% add MLE results to plot
% % %% add MLE results to plot
% % avgLSPCA = mean(LSPCArates, 1);
% % avgLSPCA_train = mean(LSPCArates_train, 1);
% % avgkLSPCA = mean(kLSPCArates, 1);
% % avgkLSPCA_train = mean(kLSPCArates_train, 1);
% % avgLSPCAvar = mean(LSPCAvar, 1);
% % avgkLSPCAvar = mean(kLSPCAvar, 1);
% % avgLSPCAvar_train = mean(LSPCAvar_train, 1);
% % avgkLSPCAvar_train = mean(kLSPCAvar_train, 1);
% % hold on
% % plot(avgLSPCAvar_train(klspca-1), avgLSPCA_train(klspca-1), 'b*', 'LineWidth', 2, 'MarkerSize', 20)
% % [m, loc] = min(avgkLSPCA_train(1,kklspca-1,:));
% % plot(avgkLSPCAvar_train(1,kklspca-1,loc), m, 'r*', 'LineWidth', 2, 'MarkerSize', 20)
% % lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'LSPCA (CV)', 'kLSPCA (CV)', 'Location', 'best'); lgd.FontSize = 15;
% % %     xlim([0,1.01])
% % %     saveas(gcf, strcat(dataset, 'multi_obj_train.jpg'))
% % % end
% % %
% % %
% % %
% % % %% Visualize for select methods (only really good for classification)
% % % load('colorblind_colormap.mat')
% % % colormap(colorblind)
% % %
% % % %KLSPCA
% % % figure()
% % % loc = find(kLSPCArates == min(kLSPCArates, [], 'all'), 1, 'last');
% % % [idx, ~, ~, ~] = ind2sub(size(kLSPCArates),loc);
% % % data = klspca_mbd_train{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytrains{idx}, 'filled', 'linewidth', 3)
% % % %scatter3(data(:,1), data(:,2), data(:,3), 100, Ytrains(:,:,row), 'filled', 'linewidth', 3)
% % % hold on
% % % data = klspca_mbd_test{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytests{idx})
% % % %scatter3(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % % colormap(colorblind)
% % % set(gca, 'yticklabel', '')
% % % set(gca, 'xticklabel', '')
% % % title(strcat('KLSPCA err:  ', num2str(min(kLSPCArates, [], 'all'))))
% % % grid on; set(gca, 'fontsize', 25)
% % % saveas(gcf, strcat(dataset, 'KLSPCA.jpg'))
% % %
% % % %LSPCA
% % % figure()
% % % loc = find(LSPCArates == min(LSPCArates, [], 'all'), 1, 'last');
% % % [idx, ~, ~, ~] = ind2sub(size(kLSPCArates),loc);
% % % data = lspca_mbd_train{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytrains{idx}, 'filled', 'linewidth', 3)
% % % %scatter3(data(:,1), data(:,2), data(:,3), 100, Ytrains(:,:,row), 'filled', 'linewidth', 3)
% % % hold on
% % % data = lspca_mbd_test{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytests{idx})
% % % %scatter3(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % % colormap(colorblind)
% % % set(gca, 'yticklabel', '')
% % % set(gca, 'xticklabel', '')
% % % title(strcat('LSPCA err:  ', num2str(min(LSPCArates, [], 'all'))))
% % % grid on; set(gca, 'fontsize', 25)
% % % saveas(gcf, strcat(dataset, 'LSPCA.jpg'))
% % %
% % % % %LSPCA
% % % % figure()
% % % % [r,row] = min(min(ILSPCArates, [], 2));
% % % % [r,col] = min(min(ILSPCArates, [], 1));
% % % % data = Ilspca_mbd_train{row,col};
% % % % %scatter(data(:,1), data(:,2), 100, Ytrains(:,:,row), 'filled', 'linewidth', 3)
% % % % scatter3(data(:,1), data(:,2), data(:,3), 100, Ytrains(:,:,row), 'filled', 'linewidth', 3)
% % % % hold on
% % % % data = Ilspca_mbd_test{row,col};
% % % % %scatter(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % % % scatter3(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % % % colormap(colorblind)
% % % % set(gca, 'yticklabel', '')
% % % % set(gca, 'xticklabel', '')
% % % % title(strcat('LSPCA err:  ', num2str(min(LSPCArates, [], 'all'))))
% % % % grid on; set(gca, 'fontsize', 25)
% % % % saveas(gcf, strcat(dataset, 'LSPCA_gamma.jpg'))
% % % %
% % % % %lambda LSPCA
% % % % figure()
% % % % [r,row] = min(min(lambda_LSPCArates, [], 2));
% % % % [r,col] = min(min(lambda_LSPCArates, [], 1));
% % % % data = lambda_lspca_mbd_train{row,col};
% % % % %scatter(data(:,1), data(:,2), 100, Ytrains(:,:,row), 'filled', 'linewidth', 3)
% % % % scatter3(data(:,1), data(:,2), data(:,3), 100, Ytrains(:,:,row), 'filled', 'linewidth', 3)
% % % % hold on
% % % % data = lambda_lspca_mbd_test{row,col};
% % % % %scatter(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % % % scatter3(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % % % colormap(colorblind)
% % % % set(gca, 'yticklabel', '')
% % % % set(gca, 'xticklabel', '')
% % % % title(strcat('lambda LSPCA err:  ', num2str(min(LSPCArates, [], 'all'))))
% % % % grid on; set(gca, 'fontsize', 25)
% % % % saveas(gcf, strcat(dataset, 'LSPCA_gamma.jpg'))
% % %
% % % %PCA
% % % figure()
% % % [r,loc] = min(PCArates);
% % % data = Zpcas{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytrains{loc}, 'filled', 'linewidth', 3)
% % % hold on
% % % data = pcaXtests{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytests{loc})
% % % colormap(colorblind)
% % % set(gca, 'yticklabel', '')
% % % set(gca, 'xticklabel', '')
% % % title(strcat('PCA err:  ', num2str(r)))
% % % grid on; set(gca, 'fontsize', 25)
% % % saveas(gcf, strcat(dataset, 'PCA.jpg'))
% % %
% % % %Barshan
% % % figure()
% % % [r,loc] = min(SPCArates);
% % % data = Zspcas{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytrains{loc}, 'filled', 'linewidth', 3)
% % % hold on
% % % data = spcaXtests{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytests{loc})
% % % colormap(colorblind)
% % % set(gca, 'yticklabel', '')
% % % set(gca, 'xticklabel', '')
% % % title(strcat('Barshan err:  ', num2str(r)))
% % % grid on; set(gca, 'fontsize', 25)
% % % saveas(gcf, strcat(dataset, 'Barshan.jpg'))
% % %
% % % %kBarshan
% % % figure()
% % % loc = find(kSPCArates == min(kSPCArates, [], 'all'), 1, 'last');
% % % [idx, ~, ~] = ind2sub(size(kSPCArates),loc);
% % % data = Zkspcas{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytrains{idx}, 'filled', 'linewidth', 3)
% % % hold on
% % % data = kspcaXtests{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytests{idx})
% % % colormap(colorblind)
% % % set(gca, 'yticklabel', '')
% % % set(gca, 'xticklabel', '')
% % % title(strcat('kBarshan err:  ', num2str(r)))
% % % grid on; set(gca, 'fontsize', 25)
% % % saveas(gcf, strcat(dataset, 'kBarshan.jpg'))
% % %
% % % %ISPCA
% % % figure()
% % % [r,loc] = min(ISPCArates);
% % % data = Zispcas{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytrains{loc}, 'filled', 'linewidth', 3)
% % % hold on
% % % data = ISPCAXtests{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytests{loc})
% % % colormap(colorblind)
% % % set(gca, 'yticklabel', '')
% % % set(gca, 'xticklabel', '')
% % % title(strcat('ISPCA err:  ', num2str(r)))
% % % grid on; set(gca, 'fontsize', 25)
% % % saveas(gcf, strcat(dataset, 'ISPCA.jpg'))
% % %
% % % %SPPCA
% % % figure()
% % % [r,loc] = min(SPPCArates);
% % % data = Zsppcas{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytrains{loc}, 'filled', 'linewidth', 3)
% % % hold on
% % % data = SPPCAXtests{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytests{loc})
% % % colormap(colorblind)
% % % set(gca, 'yticklabel', '')
% % % set(gca, 'xticklabel', '')
% % % title(strcat('SPPCA err:  ', num2str(r)))
% % % grid on; set(gca, 'fontsize', 25)
% % % saveas(gcf, strcat(dataset, 'SPPCA.jpg'))
% % %
% % % %LFDA
% % % figure()
% % % [r,loc] = min(SPPCArates);
% % % data = Zsppcas{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytrains{loc}, 'filled', 'linewidth', 3)
% % % hold on
% % % data = SPPCAXtests{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytests{loc})
% % % colormap(colorblind)
% % % set(gca, 'yticklabel', '')
% % % set(gca, 'xticklabel', '')
% % % title(strcat('SPPCA err:  ', num2str(r)))
% % % grid on; set(gca, 'fontsize', 25)
% % % saveas(gcf, strcat(dataset, 'SPPCA.jpg'))
% % %
% % % %kLFDA
% % % figure()
% % % [r,loc] = min(SPPCArates);
% % % data = Zsppcas{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytrains{loc}, 'filled', 'linewidth', 3)
% % % hold on
% % % data = SPPCAXtests{loc};
% % % scatter(data(:,1), data(:,2), 100, Ytests{loc})
% % % colormap(colorblind)
% % % set(gca, 'yticklabel', '')
% % % set(gca, 'xticklabel', '')
% % % title(strcat('SPPCA err:  ', num2str(r)))
% % % grid on; set(gca, 'fontsize', 25)
% % % saveas(gcf, strcat(dataset, 'SPPCA.jpg'))
% 
% 
% 

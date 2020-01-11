%% setup and load data
load(strcat(dataset, '.mat'));
[n, p] = size(X);
[~, q] = size(Y);
ks = 2:min(10, p-1);

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

% p = gcp('nocreate'); % If no pool, do not create new one.
% if isempty(p)
%     parpool(6)
% end
%delete(gcp('nocreate'))
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
        PCAtimes(l,t) = toc;
        % compute error
        B = mnrfit(Xtrain*Lpca',Ytrain, 'Model', 'nominal', 'Interactions', 'on');
        [~, Yhat] = max(mnrval(B,pcaXtest),[], 2);
        [~,pcaYtrain] = max(mnrval(B,Zpca),[], 2);
        PCArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
        PCArates_train(l,t) = 1 - sum(pcaYtrain == Ytrain) / ntrain;
        PCAvar(l,t) = norm(Xtest*Lpca'*Lpca, 'fro') / norm(Xtest, 'fro');
        PCAvar_train(l,t) = norm(Xtrain*Lpca'*Lpca, 'fro') / norm(Xtrain, 'fro');
    end
    
    %         %% kPCA
    %     for l = 1:kfold
    %         test_num = l
    %         % get lth fold
    %         Xtrain = Xtrains{l};
    %         Ytrain = Ytrains{l};
    %         Xtest = Xtests{l};
    %         Ytest = Ytests{l};
    %
    %         if l == 1
    %             for jj = 1:length(sigmas)
    %                 sigma = sigmas(jj)
    %                 K = gaussian_kernel(Xtrainh, Xtrainh, sigma);
    %                 [Lkpca, Zkpca] = pca(K, 'NumComponents', k);
    %                 Lkpca = Lkpca';
    %                 B = mnrfit(Zkpca,Ytrainh, 'Model', 'nominal', 'Interactions', 'on');
    %                 [~, Yhat] = max(mnrval(B,gaussian_kernel(Xtesth, Xtrainh, sigma)*Lkpca'),[], 2);
    %                 PCAratesh(jj) = 1 - sum(Yhat == Ytesth) / nHoldTest;
    %             end
    %             sigloc = find(PCAratesh==min(PCAratesh,[],'all'),1,'last');
    %             bestSigma = sigmas(sigloc);
    %         end
    %
    %         K = gaussian_kernel(Xtrain, Xtrain, bestSigma);
    %         Ktest = gaussian_kernel(Xtest, Xtrain, sigma);
    %         tic
    %         [Lkpca, Zkpca] = pca(K, 'NumComponents', k);
    %         kPCAtimes(l,t) = toc;
    %         Zkpcas{l,t} = Zkpca;
    %         Lkpca = Lkpca';
    %         % compute embedding for test data
    %         kpcaXtest = Ktest*Lkpca';
    %         kpcaXtests{l,t} = kpcaXtest;
    %         % compute error
    %         B = mnrfit(Zkpca,Ytrain, 'Model', 'nominal', 'Interactions', 'on');
    %         [~, Yhat] = max(mnrval(B,kpcaXtest),[], 2);
    %         [~,kpcaYtrain] = max(mnrval(B,Zkpca),[], 2);
    %         kPCArates(l,t) = 1 - sum(Yhat == Ytest) / ntest
    %         kPCArates_train(l,t) = 1 - sum(kpcaYtrain == Ytrain) / ntrain;
    %         kPCAvar(l,t) = norm(kpcaXtest, 'fro') / norm(Ktest, 'fro');
    %         kPCAvar_train(l,t) = norm(Zkpca, 'fro') / norm(K, 'fro');
    %     end
    
    %% LRPCA
    
    Lambdas = fliplr(logspace(-2, 0, 40));
    %Lambdas = linspace(1, 0, 40)
    %Lambdas = 0.01;
    Gammas = [1, 1.5, 2, 5, 10];
    %Gammas = 10;
    %Gammas = 10;
    
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
        for ii = 1:length(Gammas)
            Gamma = Gammas(ii);
            for jj = 1:length(Lambdas)
                Lambda = Lambdas(jj)
                
                if jj == 1
                    [Zlspca, Llspca, B] = lrpca_gamma_lambda(Xtrain, Ytrain, Lambda, Gamma, k, 0);
                    %[Zlspca, Llspca, B] = ilspca_gamma(Xtrain, Ytrain, Lambda, Gamma, k);

                else
                    [Zlspca, Llspca, B] = lrpca_gamma_lambda(Xtrain, Ytrain, Lambda, Gamma, k, Llspca');
                    %[Zlspca, Llspca, B] = ilspca_gamma(Xtrain, Ytrain, Lambda, Gamma, k);
                end
                Llspca = Llspca';
                Ls{l,t,ii, jj} = Llspca;
                %predict
                LSPCAXtest = Xtest*Llspca';
                LSPCAXtrain = Xtrain*Llspca';
                [~, LSPCAYtest] = max(LSPCAXtest*B(2:end,:) + B(1,:), [], 2);
                [~, LSPCAYtrain] = max(Xtrain*Llspca'*B(2:end,:) + B(1,:), [], 2);
                lspca_mbd_test{l,t,ii,jj} =LSPCAXtest;
                lspca_mbd_train{l,t,ii,jj} = Zlspca;
                % compute error
                err = 1 - sum(Ytest == LSPCAYtest) / ntest;
                train_err = 1 - sum(Ytrain == LSPCAYtrain) / ntrain;                
                LSPCArates(l,t, ii, jj) = err ;
                LSPCArates_train(l,t, ii, jj) = train_err;
                LSPCAvar(l,t, ii, jj) = norm(LSPCAXtest, 'fro') / norm(Xtest, 'fro');
                LSPCAvar_train(l,t, ii, jj) = norm(Zlspca, 'fro') / norm(Xtrain, 'fro');
            end
        end
    end
    
    
    %% kLRPCA
    Lambdas = fliplr(logspace(-2, 0, 10));
    Gammas = [1, 1.5, 2];
    
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
            for ii = 1:length(Gammas)
                Gamma = Gammas(ii);
                for jj = 1:length(Lambdas)
                    Lambda = Lambdas(jj)
                    
                    if jj == 1
                        %[ Zklspca, Lorth, B, Klspca] = klspca_gamma(Xtrain, Ytrain, Lambda, Gamma, sigma, k, 0);
                        [ Zklspca, Lorth, B, Klspca] = klrpca_gamma_lambda(Xtrain, Ytrain, Lambda, Gamma, sigma, k, 0, 0);
                    else
                        %[ Zklspca, Lorth, B, Klspca] = klspca_gamma(Xtrain, Ytrain, Lambda, Gamma, sigma, k, Klspca);
                        [ Zklspca, Lorth, B, Klspca] = klrpca_gamma_lambda(Xtrain, Ytrain, Lambda, Gamma, sigma, k, Lorth, Klspca);
                    end
                    %Lorth = Lorth';
                    embedFunc = @(data) klspca_embed(data, Xtrain, Lorth, sigma);
                    kLs{l,t,ii,jj,kk} = Lorth;
                    kLSPCAXtest = embedFunc(Xtest);
                    klspca_mbd_test{l,t,ii,jj,kk} = kLSPCAXtest;
                    kLSPCAXtrain = Zklspca;
                    klspca_mbd_train{l,t,ii,jj,kk} = Zklspca;
                    [~, kLSPCAYtest] = max(kLSPCAXtest*B(2:end,:) + B(1,:), [], 2);
                    [~, kLSPCAYtrain] = max(Zklspca*B(2:end,:) + B(1,:), [], 2);
                    err = 1 - sum(Ytest == kLSPCAYtest)/ntest;
                    train_err = 1 - sum(Ytrain == kLSPCAYtrain)/ntrain;
                    kLSPCArates(l,t,ii,jj,kk) = err ;
                    kLSPCArates_train(l,t,ii,jj,kk) = train_err;
                    kLSPCAvar(l,t,ii,jj,kk) = norm(kLSPCAXtest, 'fro') / norm(gaussian_kernel(Xtest,Xtrain,sigma), 'fro');
                    kLSPCAvar_train(l,t,ii,jj,kk) = norm(Zklspca, 'fro') / norm(Klspca, 'fro');
                end
            end
        end
    end

    
    %% KPCA
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xtrain = Xtrains{l};
        Ytrain = Ytrains{l};
        [ntrain, ~] = size(Ytrain);
        Xtest = Xtests{l};
        Ytest = Ytests{l};
        [ntest, ~] = size(Ytest);
        for jj = 1:length(sigmas)
            sigma = sigmas(jj)
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
    
    %% ISPCA
    %find basis
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
        [Zispca, Lispca, B] = ISPCA(Xtrain,Ytrain,k);
        Zispcas{l,t} = Zispca;
        ISPCAtimes(l,t) = toc;
        % predict
        ISPCAXtest = Xtest*Lispca';
        ISPCAXtests{l,t} = ISPCAXtest;
        B = mnrfit(Xtrain*Lispca',Ytrain, 'Model', 'nominal', 'Interactions', 'on');
        [~,Yhat] = max(mnrval(B,ISPCAXtest),[], 2);
        [~, ISPCAYtrain] = max(mnrval(B,Zispca),[], 2);
        % compute error
        ISPCArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
        ISPCArates_train(l,t) = 1 - sum(ISPCAYtrain == Ytrain) / ntrain;
        ISPCAvar(l,t) = norm(Xtest*Lispca', 'fro') / norm(Xtest, 'fro');
        ISPCAvar_train(l,t) = norm(Xtrain*Lispca', 'fro') / norm(Xtrain, 'fro');
        ISPCAtimes(l,t) = toc;
    end
    
    %% SPPCA
    % solve
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
        Zsppcasin = {};
        Lsppcasin = {};
        SPPCAXtestin = {};
        SPPCAYtestin = {};
        SPPCAYtrainin = {};
        sppca_err = [];
        for count = 1%:10 %do 10 initializations and take the best b/c ends up in bad local minima a lot
            [Zsppca, Lsppca, ~, W_x, W_y, var_x, var_y] = SPPCA(Xtrain,Ytrain,k,exp(-10), randn(p,k), randn(q,k));
            Zsppcasin{count} = Zsppca;
            Lsppcasin{count} = Lsppca;
            SPPCAXtestin{count} = Xtest*Lsppca';
            B = mnrfit(Zsppca,Ytrain, 'Model', 'nominal', 'Interactions', 'on');
            SPPCAYtestin{count} = max(mnrval(B,SPPCAXtestin{count}),[], 2);
            SPPCAYtrainin{count} = max(mnrval(B,Zsppca),[], 2);
            sppca_err(count) =  norm(SPPCAYtrainin{count} - Ytrain, 'fro');
        end
        [~, loc] = min(sppca_err);
        Zsppca = Zsppcasin{loc};
        Zsppcas{l,t} = Zsppca;
        Lsppca = orth(Lsppcasin{loc}')';
        % Predict
        SPPCAXtest = SPPCAXtestin{loc};
        SPPCAXtests{l,t} = SPPCAXtest;
        SPPCAYtest = SPPCAYtestin{loc};
        SPPCAYtrain = SPPCAYtrainin{loc};
        % compute error
        B = mnrfit(Zsppca,Ytrain, 'Model', 'nominal', 'Interactions', 'on');
        [~,SPPCAYtrain] = max(mnrval(B,Zsppca),[], 2);
        [~,Yhat] = max(mnrval(B,SPPCAXtest),[], 2);
        SPPCArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
        SPPCArates_train(l,t) = 1 - sum(SPPCAYtrain == Ytrain) / ntrain;
        Lsppca_orth = orth(Lsppca'); %normalize latent directions for variation explained comparison
        Lsppca_orth = Lsppca_orth';
        SPPCAvar(l,t) = norm(Xtest*Lsppca_orth', 'fro') / norm(Xtest, 'fro');
        SPPCAvar_train(l,t) = norm(Xtrain*Lsppca_orth', 'fro') / norm(Xtrain, 'fro');
        
        SPPCAtimes(l,t) = toc;
    end
    
    %% Barshan
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
        barshparam = struct;
        if n>p
            %learn basis
            [Zspca Lspca] = SPCA(Xtrain', Ytrain', k);
            spcaXtest = Xtest*Lspca';
            % predict
            B = mnrfit(Zspca,Ytrain, 'Model', 'nominal', 'Interactions', 'on');
            [~,Yhat] = max(mnrval(B,spcaXtest),[], 2);
            %compute error
            SPCArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
            [~,SPCAYtrain] = max(mnrval(B,Zspca),[], 2);
            SPCArates_train(l,t) = 1 - sum(SPCAYtrain == Ytrain) / ntrain;
            SPCAvar(l,t) = norm(Xtest*Lspca', 'fro') / norm(Xtest, 'fro');
            SPCAvar_train(l,t) = norm(Xtrain*Lspca', 'fro') / norm(Xtrain, 'fro');
        else
            % kernel version faster in this regime
            barshparam.ktype_y = 'delta';
            barshparam.kparam_y = 1;
            barshparam.ktype_x = 'linear';
            barshparam.kparam_x = 1;
            [Zspca Lspca] = KSPCA(Xtrain', Ytrain', k, barshparam);
            Zspca = Zspca';
            %do prediction in learned basis
            betaSPCA = Zspca \ Ytrain;
            Ktrain = Xtrain*Xtrain';
            Ktest = Xtest*Xtrain';
            spcaXtest = Ktest*Lspca;
            spca_mbd_test{l,t} = spcaXtest;
            spca_mbd_train{l,t} = Zspca;
            B = mnrfit(Zspca,Ytrain, 'Model', 'nominal', 'Interactions', 'on');
            [~,Yhat] = max(mnrval(B,spcaXtest),[], 2);
            [~,SPCAYtrain] = max(mnrval(B,Zspca),[], 2);
            % compute error
            SPCArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
            SPCArates_train(l,t) = 1 - sum(SPCAYtrain == Ytrain) / ntrain;
            SPCAvar(l,t) = norm(spcaXtest, 'fro') / norm(Ktest, 'fro');
            SPCAvar_train(l,t) = norm(Zspca, 'fro') / norm(Ktrain, 'fro');
        end
        spcaXtests{l,t} = spcaXtest;
        Zspcas{l,t} = Zspca;
        Barshantimes(l,t) = toc;
    end
    
    %% Perform Barshan's KSPCA based 2D embedding
    %learn basis
    for l = 1:kfold
        test_num = l
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
    for l = 1:kfold
        test_num = l
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
    
    
    %         %% QDA
    %         % solve
    %         QMdl = fitcdiscr(Xtrain,Ytrain, 'DiscrimType', 'pseudoquadratic');
    %         QDAYtest = predict(QMdl,Xtest);
    %         % Predict
    %         QDAYtrain = predict(QMdl,Xtrain);
    %         %compute error
    %         QDArates(l,t) = 1 - sum(QDAYtest == Ytest) / ntest;
    %         QDArates(l,t)
    %         QDArates_train(l,t) = 1 - sum(QDAYtrain == Ytrain) / ntrain;
    
    %% Local Fisher Discriminant Analysis (LFDA)
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xtrain = Xtrains{l};
        Ytrain = Ytrains{l};
        [ntrain, ~] = size(Ytrain);
        Xtest = Xtests{l};
        Ytest = Ytests{l};
        [ntest, ~] = size(Ytest);
        K = Xtrain*Xtrain';
        Ktest = Xtest*Xtrain';
        %if ~strcmp(dataset, 'colon') && ~strcmp(dataset, 'Arcene')
            %train
            %[Llfda,Zlfda] = LFDA(Xtrain',Ytrain,k, 'plain');
            [Llfda,~] = KLFDA(K,Ytrain,k, 'plain',9);
            %predict
            Llfda = orth(Llfda);
            Zlfda = K*Llfda;
            B = mnrfit(Zlfda,Ytrain, 'Model', 'nominal', 'Interactions', 'on');
            %LFDAXtest = Xtest*Llfda;
            LFDAXtest = Ktest*Llfda;
            [~,Yhat] = max(mnrval(B,LFDAXtest),[], 2);
            [~,LFDAYtrain] = max(mnrval(B,Zlfda),[], 2);
            %compute error
            LFDArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
            LFDArates_train(l,t) = 1 - sum(LFDAYtrain == Ytrain) / ntrain;
%             LFDAvar(l,t) = norm(LFDAXtest, 'fro') / norm(Xtest, 'fro');
%             LFDAvar_train(l,t) = norm(Zlfda, 'fro') / norm(Xtrain, 'fro');
            LFDAvar(l,t) = norm(LFDAXtest, 'fro') / norm(Ktest, 'fro');
            LFDAvar_train(l,t) = norm(Zlfda, 'fro') / norm(K, 'fro');
%         else
%             LFDArates(l,t) = nan;
%             LFDArates_train(l,t) = nan;
%             LFDAvar(l,t) = nan;
%             LFDAvar_train(l,t) = nan;
%         %end
    end
    
    %% Kernel Local Fisher Discriminant Analysis (KLFDA)
    %choose kernel param
    for l = 1:kfold
        test_num = l
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
%% save all data
save(strcat(dataset, '_results_dim_aao'))

%% compute avg performance accross folds

avgPCA = mean(PCArates);
avgPCA_train = mean(PCArates_train);
avgkPCA = mean(kPCArates);
avgkPCA_train = mean(kPCArates_train);
avgLSPCA = mean(LSPCArates);
avgLSPCA_train = mean(LSPCArates_train);
%     lambda_avgLSPCA = mean(lambda_LSPCArates, 1);
%     lambda_avgLSPCA_train = mean(lambda_LSPCArates_train, 1);
avgkLSPCA = mean(kLSPCArates);
avgkLSPCA_train = mean(kLSPCArates_train);
%     avgILSPCA = mean(ILSPCArates);
%     avgILSPCA_train = mean(ILSPCArates_train);
avgSPCA = mean(SPCArates);
avgSPCA_train = mean(SPCArates_train);
avgkSPCA = mean(kSPCArates);
avgkSPCA_train = mean(kSPCArates_train);
avgISPCA = mean(ISPCArates);
avgISPCA_train = mean(ISPCArates_train);
avgSPPCA = mean(SPPCArates);
avgSPPCA_train = mean(SPPCArates_train);
avgLDA = mean(LDArates);
avgLDA_train = mean(LDArates_train);
%     avgQDA = mean(QDArates);
%     avgQDA_train = mean(QDArates_train);
avgLFDA = mean(LFDArates);
avgLFDA_train = mean(LFDArates_train);
avgkLFDA = mean(kLFDArates);
avgkLFDA_train = mean(kLFDArates_train);
%
avgPCAvar = mean(PCAvar);
avgkPCAvar = mean(kPCAvar);
avgLSPCAvar = mean(LSPCAvar);
%     lambda_avgLSPCAvar = mean(lambda_LSPCAvar);
avgkLSPCAvar = mean(kLSPCAvar);
%     avgILSPCAvar = mean(ILSPCAvar);
avgSPCAvar = mean(SPCAvar);
avgkSPCAvar = mean(kSPCAvar);
avgISPCAvar = mean(ISPCAvar);
avgSPPCAvar = mean(SPPCAvar);
avgLDAvar = mean(LDAvar);
avgLFDAvar = mean(LFDAvar);
avgkLFDAvar = mean(kLFDAvar);

avgPCAvar_train = mean(PCAvar_train);
avgkPCAvar_train = mean(kPCAvar_train);
avgLSPCAvar_train = mean(LSPCAvar_train);
%     lambda_avgLSPCAvar_train = mean(lambda_LSPCAvar_train);
avgkLSPCAvar_train = mean(kLSPCAvar_train);
%     avgILSPCAvar_train = mean(ILSPCAvar_train);
avgSPCAvar_train = mean(SPCAvar_train);
avgkSPCAvar_train = mean(kSPCAvar_train);
avgISPCAvar_train = mean(ISPCAvar_train);
avgSPPCAvar_train = mean(SPPCAvar_train);
avgLDAvar_train = mean(LDAvar_train);
avgLFDAvar_train = mean(LFDAvar_train);
avgkLFDAvar_train = mean(kLFDAvar_train);

%% print mean performance with std errors


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

loc = find(avgLDA==min(avgLDA,[],'all'),1,'last');
[~,kloc] = ind2sub(size(avgLDA), loc);
k = ks(kloc);
m = mean(LDArates(:,kloc));
v = mean(LDAvar(:,kloc));
sm = std(LDArates(:,kloc));
sv = std(LDAvar(:,kloc));
sprintf('LDAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

loc = find(avgLFDA==min(avgLFDA,[],'all'),1,'last');
[~,kloc] = ind2sub(size(avgLFDA), loc);
k = ks(kloc);
m = mean(LFDArates(:,kloc));
v = mean(LFDAvar(:,kloc));
sm = std(LFDArates(:,kloc));
sv = std(LFDAvar(:,kloc));
sprintf('LFDAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

loc = find(avgkLFDA==min(avgkLFDA,[],'all'),1,'last');
[~,kloc,sigloc] = ind2sub(size(avgkLFDA), loc);
k = ks(kloc);
m = mean(kLFDArates(:,kloc,sigloc));
v = mean(kLFDAvar(:,kloc,sigloc));
sm = std(kLFDArates(:,kloc,sigloc));
sv = std(kLFDAvar(:,kloc,sigloc));
sprintf('kLFDAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)
% 
% %% print mean performance with std errors for k=2
% k = 3;
% kloc = 2;
% klock=2;
% 
% loc = find(avgPCA==min(avgPCA,[],'all'),1,'last');
% m = mean(PCArates(:,kloc));
% v = mean(PCAvar(:,kloc));
% sm = std(PCArates(:,kloc));
% sv = std(PCAvar(:,kloc));
% sprintf('PCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)
% 
% loc = find(avgLSPCA(:,kloc,:)==min(avgLSPCA(:,kloc,:),[],'all'),1,'last');
% [~,~,gamloc] = ind2sub(size(avgLSPCA(:,1,:)), loc);
% m = mean(LSPCArates(:,kloc,gamloc), 1);
% v = mean(LSPCAvar(:,kloc,gamloc), 1);
% sm = std(LSPCArates(:,kloc,gamloc), 1);
% sv = std(LSPCAvar(:,kloc,gamloc), 1);
% sprintf('LSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)
% 
% 
% loc = find(avgkLSPCA(:,kloc,:,:)==min(avgkLSPCA(:,kloc,:,:),[],'all'),1,'last');
% [~,~,gamlock,siglock] = ind2sub(size(avgkLSPCA(:,kloc,:,:)), loc);
% m = mean(kLSPCArates(:,klock,gamlock,siglock), 1);
% v = mean(kLSPCAvar(:,klock,gamlock,siglock), 1);
% sm = std(kLSPCArates(:,klock,gamlock,siglock), 1);
% sv = std(kLSPCAvar(:,klock,gamlock,siglock), 1);
% sprintf('kLSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)
% 
% 
% m = mean(ISPCArates(:,kloc));
% v = mean(ISPCAvar(:,kloc));
% sm = std(ISPCArates(:,kloc));
% sv = std(ISPCAvar(:,kloc));
% sprintf('ISPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)
% 
% m = mean(SPPCArates(:,kloc));
% v = mean(SPPCAvar(:,kloc));
% sm = std(SPPCArates(:,kloc));
% sv = std(SPPCAvar(:,kloc));
% sprintf('SPPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)
% 
% m = mean(SPCArates(:,kloc));
% v = mean(SPCAvar(:,kloc));
% sm = std(SPCArates(:,kloc));
% sv = std(SPCAvar(:,kloc));
% sprintf('Barshanerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)
% 
% loc = find(avgkSPCA(:,kloc,:)==min(avgkSPCA(:,kloc,:),[],'all'),1,'last');
% [~,~,sigloc] = ind2sub(size(avgkSPCA(:,kloc,:)), loc);
% m = mean(kSPCArates(:,kloc,sigloc));
% v = mean(kSPCAvar(:,kloc,sigloc));
% sm = std(kSPCArates(:,kloc,sigloc));
% sv = std(kSPCAvar(:,kloc,sigloc));
% sprintf('kBarshanerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)
% 
% 
% m = mean(LDArates(:,kloc));
% v = mean(LDAvar(:,kloc));
% sm = std(LDArates(:,kloc));
% sv = std(LDAvar(:,kloc));
% sprintf('LDAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)
% 
% 
% m = mean(LFDArates(:,kloc));
% v = mean(LFDAvar(:,kloc));
% sm = std(LFDArates(:,kloc));
% sv = std(LFDAvar(:,kloc));
% sprintf('LFDAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)
% 
% loc = find(avgkLFDA(:,kloc,:)==min(avgkLFDA(:,kloc,:),[],'all'),1,'last');
% [~,~,sigloc] = ind2sub(size(avgkLFDA(:,kloc,:)), loc);
% m = mean(kLFDArates(:,kloc,sigloc));
% v = mean(kLFDAvar(:,kloc,sigloc));
% sm = std(kLFDArates(:,kloc,sigloc));
% sv = std(kLFDAvar(:,kloc,sigloc));
% sprintf('kLFDAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)


%% plot error - var tradeoff curves
% 
% for t = 1:length(ks)
%     figure()
%     hold on
%     plot(avgPCAvar(t), avgPCA(t), 'sb', 'MarkerSize', 20, 'LineWidth', 2)
%     %plot(avgkPCAvar_train(t), avgkPCA(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
%     %plot(avgkLSPCAvar_train(t,1), avgkLSPCA(t, 1), 'sr', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgLSPCAvar(t,:), avgLSPCA(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     % plot(avgILSPCAvar_train(t,:), avgILSPCA(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     % plot(lambda_avgLSPCAvar_train(t,:), lambda_avgLSPCA(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgkLSPCAvar(t,1:end-1,siglock), avgkLSPCA(t, 1:end-1,siglock), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgISPCAvar(t), avgISPCA(t), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPPCAvar(t), avgSPPCA(t), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPCAvar(t), avgSPCA(t), '+', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgkSPCAvar(t,locb), avgkSPCA(t,locb), '>', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgLDAvar(t), avgLDA(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
%     % plot([0,1], [1,1]*avgQDA(t), ':', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgLFDAvar(t), avgLFDA(t), '^', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgkLFDAvar(t,locl), avgkLFDA(t,locl), '<', 'MarkerSize', 20, 'LineWidth', 2)
%     
%     %plot([0,1], avgLR*[1,1], 'k', 'LineWidth', 4)
%     
%     %xlabel('Variation Explained', 'fontsize', 25)
%     %title('Test', 'fontsize', 25)
%     %ylabel('Classification Error', 'fontsize', 25)
%     %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
%     set(gca, 'fontsize', 25)
%     lgd = legend('PCA', 'LSPCA', 'KLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', ...
%         'LDA', 'LFDA', 'kLFDA', 'Location', 'best'); lgd.FontSize = 15;
%     xlim([0,1.01])
%     %ylim([0,0.5])
%     saveas(gcf, strcat(dataset, 'multi_obj.jpg'))
% end
% 
% %% plot error - var tradeoff curves
% 
% for t = 1:length(ks)
%     figure()
%     hold on
%     plot(avgPCAvar_train(t), avgPCA_train(t), 'sb', 'MarkerSize', 20, 'LineWidth', 2)
%     % plot(avgkPCAvar_train(t), avgkPCA_train(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
%     %plot(avgkLSPCAvar_train(t,1), avgkLSPCA_train(t, 1), 'sr', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgLSPCAvar_train(t,:), avgLSPCA_train(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     % plot(avgILSPCAvar_train(t,:), avgILSPCA_train(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     % plot(lambda_avgLSPCAvar_train(t,:), lambda_avgLSPCA_train(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgkLSPCAvar_train(t,1:end-2,siglock), avgkLSPCA_train(t, 1:end-2,siglock), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgISPCAvar_train(t), avgISPCA_train(t), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPPCAvar_train(t), avgSPPCA_train(t), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPCAvar_train(t), avgSPCA_train(t), '+', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgkSPCAvar_train(t,locb), avgkSPCA_train(t,locb), '>', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgLDAvar_train(t), avgLDA_train(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
%     %plot([0,1], [1,1]*avgQDA_train(t), ':', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgLFDAvar_train(t), avgLFDA_train(t), '^', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgkLFDAvar_train(t,locl), avgkLFDA_train(t,locl), '<', 'MarkerSize', 20, 'LineWidth', 2)
%     
%     %plot([0,1], avgLR*[1,1], 'k', 'LineWidth', 4)
%     
%     xlabel('Variation Explained', 'fontsize', 25)
%     %title('Train', 'fontsize', 25)
%     ylabel('Classification Error', 'fontsize', 25)
%     %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
%     set(gca, 'fontsize', 25)
%     lgd = legend('PCA', 'LSPCA', 'KLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', ...
%         'LDA', 'LFDA', 'kLFDA', 'Location', 'best'); lgd.FontSize = 15;
%     xlim([0,1.01])
%     saveas(gcf, strcat(dataset, 'multi_obj_train.jpg'))
% end
% 
% 
% 
% %% Visualize for select methods (only really good for classification)
% load('colorblind_colormap.mat')
% colormap(colorblind)
% 
% %KLSPCA
% figure()
% loc = find(kLSPCArates == min(kLSPCArates, [], 'all'), 1, 'last');
% [idx, ~, ~, ~] = ind2sub(size(kLSPCArates),loc);
% data = klspca_mbd_train{loc};
% scatter(data(:,1), data(:,2), 100, Ytrains{idx}, 'filled', 'linewidth', 3)
% %scatter3(data(:,1), data(:,2), data(:,3), 100, Ytrains(:,:,row), 'filled', 'linewidth', 3)
% hold on
% data = klspca_mbd_test{loc};
% scatter(data(:,1), data(:,2), 100, Ytests{idx})
% %scatter3(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% colormap(colorblind)
% set(gca, 'yticklabel', '')
% set(gca, 'xticklabel', '')
% title(strcat('KLSPCA err:  ', num2str(min(kLSPCArates, [], 'all'))))
% grid on; set(gca, 'fontsize', 25)
% saveas(gcf, strcat(dataset, 'KLSPCA.jpg'))
% 
% %LSPCA
% figure()
% loc = find(LSPCArates == min(LSPCArates, [], 'all'), 1, 'last');
% [idx, ~, ~, ~] = ind2sub(size(kLSPCArates),loc);
% data = lspca_mbd_train{loc};
% scatter(data(:,1), data(:,2), 100, Ytrains{idx}, 'filled', 'linewidth', 3)
% %scatter3(data(:,1), data(:,2), data(:,3), 100, Ytrains(:,:,row), 'filled', 'linewidth', 3)
% hold on
% data = lspca_mbd_test{loc};
% scatter(data(:,1), data(:,2), 100, Ytests{idx})
% %scatter3(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% colormap(colorblind)
% set(gca, 'yticklabel', '')
% set(gca, 'xticklabel', '')
% title(strcat('LSPCA err:  ', num2str(min(LSPCArates, [], 'all'))))
% grid on; set(gca, 'fontsize', 25)
% saveas(gcf, strcat(dataset, 'LSPCA.jpg'))
% 
% % %LSPCA
% % figure()
% % [r,row] = min(min(ILSPCArates, [], 2));
% % [r,col] = min(min(ILSPCArates, [], 1));
% % data = Ilspca_mbd_train{row,col};
% % %scatter(data(:,1), data(:,2), 100, Ytrains(:,:,row), 'filled', 'linewidth', 3)
% % scatter3(data(:,1), data(:,2), data(:,3), 100, Ytrains(:,:,row), 'filled', 'linewidth', 3)
% % hold on
% % data = Ilspca_mbd_test{row,col};
% % %scatter(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % scatter3(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % colormap(colorblind)
% % set(gca, 'yticklabel', '')
% % set(gca, 'xticklabel', '')
% % title(strcat('LSPCA err:  ', num2str(min(LSPCArates, [], 'all'))))
% % grid on; set(gca, 'fontsize', 25)
% % saveas(gcf, strcat(dataset, 'LSPCA_gamma.jpg'))
% %
% % %lambda LSPCA
% % figure()
% % [r,row] = min(min(lambda_LSPCArates, [], 2));
% % [r,col] = min(min(lambda_LSPCArates, [], 1));
% % data = lambda_lspca_mbd_train{row,col};
% % %scatter(data(:,1), data(:,2), 100, Ytrains(:,:,row), 'filled', 'linewidth', 3)
% % scatter3(data(:,1), data(:,2), data(:,3), 100, Ytrains(:,:,row), 'filled', 'linewidth', 3)
% % hold on
% % data = lambda_lspca_mbd_test{row,col};
% % %scatter(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % scatter3(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % colormap(colorblind)
% % set(gca, 'yticklabel', '')
% % set(gca, 'xticklabel', '')
% % title(strcat('lambda LSPCA err:  ', num2str(min(LSPCArates, [], 'all'))))
% % grid on; set(gca, 'fontsize', 25)
% % saveas(gcf, strcat(dataset, 'LSPCA_gamma.jpg'))
% 
% %PCA
% figure()
% [r,loc] = min(PCArates);
% data = Zpcas{loc};
% scatter(data(:,1), data(:,2), 100, Ytrains{loc}, 'filled', 'linewidth', 3)
% hold on
% data = pcaXtests{loc};
% scatter(data(:,1), data(:,2), 100, Ytests{loc})
% colormap(colorblind)
% set(gca, 'yticklabel', '')
% set(gca, 'xticklabel', '')
% title(strcat('PCA err:  ', num2str(r)))
% grid on; set(gca, 'fontsize', 25)
% saveas(gcf, strcat(dataset, 'PCA.jpg'))
% 
% %Barshan
% figure()
% [r,loc] = min(SPCArates);
% data = Zspcas{loc};
% scatter(data(:,1), data(:,2), 100, Ytrains{loc}, 'filled', 'linewidth', 3)
% hold on
% data = spcaXtests{loc};
% scatter(data(:,1), data(:,2), 100, Ytests{loc})
% colormap(colorblind)
% set(gca, 'yticklabel', '')
% set(gca, 'xticklabel', '')
% title(strcat('Barshan err:  ', num2str(r)))
% grid on; set(gca, 'fontsize', 25)
% saveas(gcf, strcat(dataset, 'Barshan.jpg'))
% 
% %kBarshan
% figure()
% loc = find(kSPCArates == min(kSPCArates, [], 'all'), 1, 'last');
% [idx, ~, ~] = ind2sub(size(kSPCArates),loc);
% data = Zkspcas{loc};
% scatter(data(:,1), data(:,2), 100, Ytrains{idx}, 'filled', 'linewidth', 3)
% hold on
% data = kspcaXtests{loc};
% scatter(data(:,1), data(:,2), 100, Ytests{idx})
% colormap(colorblind)
% set(gca, 'yticklabel', '')
% set(gca, 'xticklabel', '')
% title(strcat('kBarshan err:  ', num2str(r)))
% grid on; set(gca, 'fontsize', 25)
% saveas(gcf, strcat(dataset, 'kBarshan.jpg'))
% 
% %ISPCA
% figure()
% [r,loc] = min(ISPCArates);
% data = Zispcas{loc};
% scatter(data(:,1), data(:,2), 100, Ytrains{loc}, 'filled', 'linewidth', 3)
% hold on
% data = ISPCAXtests{loc};
% scatter(data(:,1), data(:,2), 100, Ytests{loc})
% colormap(colorblind)
% set(gca, 'yticklabel', '')
% set(gca, 'xticklabel', '')
% title(strcat('ISPCA err:  ', num2str(r)))
% grid on; set(gca, 'fontsize', 25)
% saveas(gcf, strcat(dataset, 'ISPCA.jpg'))
% 
% %SPPCA
% figure()
% [r,loc] = min(SPPCArates);
% data = Zsppcas{loc};
% scatter(data(:,1), data(:,2), 100, Ytrains{loc}, 'filled', 'linewidth', 3)
% hold on
% data = SPPCAXtests{loc};
% scatter(data(:,1), data(:,2), 100, Ytests{loc})
% colormap(colorblind)
% set(gca, 'yticklabel', '')
% set(gca, 'xticklabel', '')
% title(strcat('SPPCA err:  ', num2str(r)))
% grid on; set(gca, 'fontsize', 25)
% saveas(gcf, strcat(dataset, 'SPPCA.jpg'))
% 
% %LFDA
% figure()
% [r,loc] = min(SPPCArates);
% data = Zsppcas{loc};
% scatter(data(:,1), data(:,2), 100, Ytrains{loc}, 'filled', 'linewidth', 3)
% hold on
% data = SPPCAXtests{loc};
% scatter(data(:,1), data(:,2), 100, Ytests{loc})
% colormap(colorblind)
% set(gca, 'yticklabel', '')
% set(gca, 'xticklabel', '')
% title(strcat('SPPCA err:  ', num2str(r)))
% grid on; set(gca, 'fontsize', 25)
% saveas(gcf, strcat(dataset, 'SPPCA.jpg'))
% 
% %kLFDA
% figure()
% [r,loc] = min(SPPCArates);
% data = Zsppcas{loc};
% scatter(data(:,1), data(:,2), 100, Ytrains{loc}, 'filled', 'linewidth', 3)
% hold on
% data = SPPCAXtests{loc};
% scatter(data(:,1), data(:,2), 100, Ytests{loc})
% colormap(colorblind)
% set(gca, 'yticklabel', '')
% set(gca, 'xticklabel', '')
% title(strcat('SPPCA err:  ', num2str(r)))
% grid on; set(gca, 'fontsize', 25)
% saveas(gcf, strcat(dataset, 'SPPCA.jpg'))




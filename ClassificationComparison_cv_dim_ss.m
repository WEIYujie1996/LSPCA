numExps = 10;
for dd = 1:numExps
    %% setup and load data
    load(strcat(dataset, '.mat'));
    [n, p] = size(X);
    [~, q] = size(Y);
    ks = 2:min(10, p-1);
    
    %holdout an independent test set
    proportion = 0.2;
    nhold = floor(n*proportion); % 10%
    idxhold = ~crossvalind('HoldOut',Y,proportion,'classes', unique(Y), 'min', 2);
    Xhold = X(idxhold, :);
    X = X(~idxhold, :);
    Yhold = Y(idxhold, :);
    Y = Y(~idxhold, :);
    n = n - nhold; %this is our new n
    
    % create same splits to use every time and center
    kfold = 10;
    cvidx = crossvalind('kfold',Y,kfold,'classes',unique(Y),'min',1);
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
        [Xlab,Xunlab,Xtest,Ylab,Yunlab,Ytest] = ss_center_data(Xlab,Xunlab,Xtest,Ylab,Yunlab,Ytest,'classification');
        Xlabs{l} = Xlab; %lth centered labled set
        Ylabs{l} = Ylab;
        Xunlabs{l} = Xunlab; %lth centered unlabled set
        Yunlabs{l} = Yunlab;
        Xtests{l} = Xtest; %lth centered test set
        Ytests{l} = Ytest;
    end
    
    % store the independent test set, and corresponding training set (non holdout data)
    % at the end of the train  and test cell arrays for convenience of
    % implementation
    
    %pick 5% of data for labeled data and use the rest as unlabeled
    labIdx = crossvalind('kfold',Y,kfold,'classes',unique(Y),'min',1);
    Xlab = X(labIdx==1,:); Ylab = Y(labIdx==1,:);
    Xunlab = X(labIdx~=1,:); Yunlab = Y(labIdx~=1,:);
    %center the labeled, unlabeled, and holdout data
    [Xlab,Xunlab,Xhold,Ylab,Yunlab,Yhold] = ss_center_data(Xlab,Xunlab,Xhold,Ylab,Yunlab,Yhold,'classification');
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
    
    % p = gcp('nocreate'); % If no pool, do not create new one.
    % if isempty(p)
    %     parpool(6)
    % end
    %delete(gcp('nocreate'))
    for t = 1:length(ks) %dimensionality of reduced data
        k = ks(t)
        
        %% PCA
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            [nlab, ~] = size(Ylab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            
            tic
            [Lpca, Zpca] = pca(Xlab, 'NumComponents', k);
            Zpcas{l,t} = Zpca;
            Lpca = Lpca';
            % compute embedding for test data
            pcaXtest = Xtest*Lpca';
            pcaXtests{l,t} = pcaXtest;
            PCAtimes(l,t) = toc;
            % compute error
            B = mnrfit(Xlab*Lpca',Ylab, 'Model', 'nominal', 'Interactions', 'on');
            [~, Yhat] = max(mnrval(B,pcaXtest),[], 2);
            [~,pcaYlab] = max(mnrval(B,Zpca),[], 2);
            PCArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
            PCArates_lab(l,t) = 1 - sum(pcaYlab == Ylab) / nlab;
            PCAvar(l,t) = norm(Xtest*Lpca'*Lpca, 'fro') / norm(Xtest, 'fro');
            PCAvar_lab(l,t) = norm(Xlab*Lpca'*Lpca, 'fro') / norm(Xlab, 'fro');
        end
        
%         %% LRPCA
%         
%         Lambdas = fliplr(logspace(-2, 0, 31));
%         for l = 1:kfold+1
%             test_num = l
%             % get lth fold
%             Xlab = Xlabs{l};
%             Ylab = Ylabs{l};
%             Xunlab = Xunlabs{l};
%             Yunlab = Yunlabs{l};
%             [nlab, ~] = size(Ylab);
%             Xtest = Xtests{l};
%             Ytest = Ytests{l};
%             [ntest, ~] = size(Ytest);
%             
%             %solve
%             for ii = 1:length(Lambdas)
%                 Lambda = Lambdas(ii);
%                 
%                 if ii == 1
%                     [Zunlab, Yunlab, Zlspca, Llspca, B] = lrpca_ss(Xunlab, Xlab, Ylab, Lambda, k, 0);
%                     
%                 else
%                     [Zunlab, Yunlab, Zlspca, Llspca, B] = lrpca_ss(Xunlab, Xlab, Ylab, Lambda, k, Llspca');
%                 end
%                 Llspca = Llspca';
%                 Ls{l,t,ii} = Llspca;
%                 %predict
%                 LSPCAXtest = Xtest*Llspca';
%                 LSPCAXlab = Xlab*Llspca';
%                 [~, LSPCAYtest] = max(LSPCAXtest*B(2:end,:) + B(1,:), [], 2);
%                 [~, LSPCAYlab] = max(Xlab*Llspca'*B(2:end,:) + B(1,:), [], 2);
%                 lspca_mbd_test{l,t,ii} =LSPCAXtest;
%                 lspca_mbd_lab{l,t,ii} = Zlspca;
%                 % compute error
%                 err = 1 - sum(Ytest == LSPCAYtest) / ntest;
%                 lab_err = 1 - sum(Ylab == LSPCAYlab) / nlab;
%                 LSPCArates(l,t, ii) = err ;
%                 LSPCArates_lab(l,t, ii) = lab_err;
%                 LSPCAvar(l,t, ii) = norm(LSPCAXtest, 'fro') / norm(Xtest, 'fro');
%                 LSPCAvar_lab(l,t, ii) = norm(Zlspca, 'fro') / norm(Xlab, 'fro');
%             end
%         end
        
        %% LRPCA (MLE)
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
            
            [Zunlab, Yunlab, Zlspca, Llspca, B] = lrpca_MLE_ss(Xunlab, Xlab, Ylab, k, 0);
            Llspca = Llspca';
            mle_Ls{l,t} = Llspca;
            %predict
            LSPCAXtest = Xtest*Llspca';
            LSPCAXlab = Xlab*Llspca';
            [~, LSPCAYtest] = max(LSPCAXtest*B(2:end,:) + B(1,:), [], 2);
            [~, LSPCAYlab] = max(Xlab*Llspca'*B(2:end,:) + B(1,:), [], 2);
            mle_lspca_mbd_test{l,t} =LSPCAXtest;
            mle_lspca_mbd_lab{l,t} = Zlspca;
            % compute error
            err = 1 - sum(Ytest == LSPCAYtest) / ntest;
            lab_err = 1 - sum(Ylab == LSPCAYlab) / nlab;
            mle_LSPCArates(l,t) = err ;
            mle_LSPCArates_lab(l,t) = lab_err;
            mle_LSPCAvar(l,t) = norm(LSPCAXtest, 'fro') / norm(Xtest, 'fro');
            mle_LSPCAvar_lab(l,t) = norm(Zlspca, 'fro') / norm(Xlab, 'fro');
            
        end
        
        
%         %% kLRPCA
%         Lambdas = fliplr(logspace(-2, 0, 31));
%         for l = 1:kfold+1
%             test_num = l
%             % get lth fold
%             Xlab = Xlabs{l};
%             Ylab = Ylabs{l};
%             Xunlab = Xunlabs{l};
%             Yunlab = Yunlabs{l};
%             [nlab, ~] = size(Ylab);
%             Xtest = Xtests{l};
%             Ytest = Ytests{l};
%             [ntest, ~] = size(Ytest);
%             
%             for kk = 1:length(sigmas)
%                 sigma = sigmas(kk)
%                 for ii = 1:length(Lambdas)
%                     Lambda = Lambdas(ii);
%                     if ii == 1
%                         [Zunlab, Yunlab, Zklspca, Lorth, B, Klspca] = klrpca_ss(Xunlab, Xlab, Ylab, Lambda, sigma, k, 0, 0);
%                     else
%                         [Zunlab, Yunlab, Zklspca, Lorth, B, Klspca] = klrpca_ss(Xunlab, Xlab, Ylab, Lambda, sigma, k, Lorth, Klspca);
%                     end
%                     %Lorth = Lorth';
%                     embedFunc = @(data) klspca_embed(data, [Xlab; Xunlab], Lorth, sigma);
%                     kLs{l,t,ii,kk} = Lorth;
%                     kLSPCAXtest = embedFunc(Xtest);
%                     klspca_mbd_test{l,t,ii,kk} = kLSPCAXtest;
%                     kLSPCAXlab = Zklspca;
%                     klspca_mbd_lab{l,t,ii,kk} = Zklspca;
%                     [~, kLSPCAYtest] = max(kLSPCAXtest*B(2:end,:) + B(1,:), [], 2);
%                     [~, kLSPCAYlab] = max(Zklspca*B(2:end,:) + B(1,:), [], 2);
%                     err = 1 - sum(Ytest == kLSPCAYtest)/ntest;
%                     lab_err = 1 - sum(Ylab == kLSPCAYlab)/nlab;
%                     kLSPCArates(l,t,ii,kk) = err ;
%                     kLSPCArates_lab(l,t,ii,kk) = lab_err;
%                     kLSPCAvar(l,t,ii,kk) = norm(kLSPCAXtest, 'fro') / norm(gaussian_kernel(Xtest,[Xlab; Xunlab],sigma), 'fro');
%                     kLSPCAvar_lab(l,t,ii,kk) = norm(Zklspca, 'fro') / norm(Klspca, 'fro');
%                 end
%                 
%             end
%         end
%         
        %% kLRPCA (MLE)
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
            
            for kk = 1:length(sigmas)
                sigma = sigmas(kk)
                [Zunlab, Yunlab, Zklspca, Lorth, B, Klspca] = klrpca_MLE_ss(Xunlab, Xlab, Ylab, sigma, k, 0, 0);
                embedFunc = @(data) klspca_embed(data, [Xlab; Xunlab], Lorth, sigma);
                mle_kLs{l,t,kk} = Lorth;
                kLSPCAXtest = embedFunc(Xtest);
                mle_klspca_mbd_test{l,t,kk} = kLSPCAXtest;
                kLSPCAXlab = Zklspca;
                mle_klspca_mbd_lab{l,t,kk} = Zklspca;
                [~, kLSPCAYtest] = max(kLSPCAXtest*B(2:end,:) + B(1,:), [], 2);
                [~, kLSPCAYlab] = max(Zklspca*B(2:end,:) + B(1,:), [], 2);
                err = 1 - sum(Ytest == kLSPCAYtest)/ntest;
                lab_err = 1 - sum(Ylab == kLSPCAYlab)/nlab;
                mle_kLSPCArates(l,t,kk) = err ;
                mle_kLSPCArates_lab(l,t,kk) = lab_err;
                mle_kLSPCAvar(l,t,kk) = norm(kLSPCAXtest, 'fro') / norm(gaussian_kernel(Xtest,[Xlab; Xunlab],sigma), 'fro');
                mle_kLSPCAvar_lab(l,t,kk) = norm(Zklspca, 'fro') / norm(Klspca, 'fro');
            end
        end
        
        
        %% KPCA
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
                sigma = sigmas(jj)
                K = gaussian_kernel(Xlab, [Xlab; Xunlab], sigma);
                Ktest = gaussian_kernel(Xtest, [Xlab; Xunlab], sigma);
                tic
                [Lkpca, Zkpca] = pca(K, 'NumComponents', k);
                kPCAtimes(l,t,jj) = toc;
                Zkpcas{l,t,jj} = Zkpca;
                Lkpca = Lkpca';
                % compute embedding for test data
                kpcaXtest = Ktest*Lkpca';
                kpcaXtests{l,t,jj} = kpcaXtest;
                % compute error
                B = mnrfit(Zkpca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
                [~, Yhat] = max(mnrval(B,kpcaXtest),[], 2);
                [~,kpcaYlab] = max(mnrval(B,Zkpca),[], 2);
                kPCArates(l,t,jj) = 1 - sum(Yhat == Ytest) / ntest;
                kPCArates_lab(l,t,jj) = 1 - sum(kpcaYlab == Ylab) / nlab;
                kPCAvar(l,t,jj) = norm(kpcaXtest, 'fro') / norm(Ktest, 'fro');
                kPCAvar_lab(l,t,jj) = norm(Zkpca, 'fro') / norm(K, 'fro');
            end
        end
        
        %% ISPCA
        %find basis
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            [nlab, ~] = size(Ylab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            tic
            [Zispca, Lispca, B] = ISPCA(Xlab,Ylab,k);
            Zispcas{l,t} = Zispca;
            ISPCAtimes(l,t) = toc;
            % predict
            ISPCAXtest = Xtest*Lispca';
            ISPCAXtests{l,t} = ISPCAXtest;
            B = mnrfit(Xlab*Lispca',Ylab, 'Model', 'nominal', 'Interactions', 'on');
            [~,Yhat] = max(mnrval(B,ISPCAXtest),[], 2);
            [~, ISPCAYlab] = max(mnrval(B,Zispca),[], 2);
            % compute error
            ISPCArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
            ISPCArates_lab(l,t) = 1 - sum(ISPCAYlab == Ylab) / nlab;
            ISPCAvar(l,t) = norm(Xtest*Lispca', 'fro') / norm(Xtest, 'fro');
            ISPCAvar_lab(l,t) = norm(Xlab*Lispca', 'fro') / norm(Xlab, 'fro');
            ISPCAtimes(l,t) = toc;
        end
        
        %% SPPCA
        % solve
        for l = 1:kfold+1
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            [nlab, ~] = size(Ylab);
            Xtest = Xtests{l};
            Ytest = Ytests{l};
            [ntest, ~] = size(Ytest);
            tic
            Zsppcasin = {};
            Lsppcasin = {};
            SPPCAXtestin = {};
            SPPCAYtestin = {};
            SPPCAYlabin = {};
            sppca_err = [];
            for count = 1%:10 %do 10 initializations and take the best b/c ends up in bad local minima a lot
                Ypm1 = Ylab; Ypm1(Ypm1 == 2) = -1;
                [Zunlab, Yunlab, Zsppca, Lsppca, B] = SPPCA_SS(Xlab,Ypm1,k,1e-6);
                Lsppca = Lsppca';
                Zsppcasin{count} = Zsppca;
                Lsppcasin{count} = Lsppca;
                SPPCAXtestin{count} = Xtest*Lsppca;
                B = mnrfit(Zsppca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
                [~,SPPCAYtestin{count}] = max(mnrval(B,SPPCAXtestin{count}),[], 2);
                [~,SPPCAYlabin{count}] = max(mnrval(B,Zsppca),[], 2);
                sppca_err(count) =  (1 - sum(SPPCAYlabin{count}==Ylab)) / ntest;
            end
            [~, loc] = min(sppca_err);
            Zsppca = Zsppcasin{loc};
            Zsppcas{l,t} = Zsppca;
            % Predict
            SPPCAXtest = SPPCAXtestin{loc};
            SPPCAXtests{l,t} = SPPCAXtest;
            SPPCAYtest = SPPCAYtestin{loc};
            SPPCAYlab = SPPCAYlabin{loc};
            % compute error
            B = mnrfit(Zsppca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
            [~,SPPCAYlab] = max(mnrval(B,Zsppca),[], 2);
            [~,Yhat] = max(mnrval(B,SPPCAXtest),[], 2);
            SPPCArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
            SPPCArates_lab(l,t) = 1 - sum(SPPCAYlab == Ylab) / nlab;
            Lsppca_orth = orth(Lsppca); %normalize latent directions for variation explained comparison
            SPPCAvar(l,t) = norm(Xtest*Lsppca_orth, 'fro') / norm(Xtest, 'fro');
            SPPCAvar_lab(l,t) = norm(Xlab*Lsppca_orth, 'fro') / norm(Xlab, 'fro');
            
            SPPCAtimes(l,t) = toc;
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
            tic
            barshparam = struct;
            %         if n>p
            %learn basis
            [Zspca Lspca] = SPCA(Xlab', Ylab', k);
            spcaXtest = Xtest*Lspca';
            % predict
            B = mnrfit(Zspca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
            [~,Yhat] = max(mnrval(B,spcaXtest),[], 2);
            %compute error
            SPCArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
            [~,SPCAYlab] = max(mnrval(B,Zspca),[], 2);
            SPCArates_lab(l,t) = 1 - sum(SPCAYlab == Ylab) / nlab;
            SPCAvar(l,t) = norm(Xtest*Lspca', 'fro') / norm(Xtest, 'fro');
            SPCAvar_lab(l,t) = norm(Xlab*Lspca', 'fro') / norm(Xlab, 'fro');
            %         else
            %             % kernel version faster in this regime
            %             barshparam.ktype_y = 'delta';
            %             barshparam.kparam_y = 1;
            %             barshparam.ktype_x = 'linear';
            %             barshparam.kparam_x = 1;
            %             [Zspca Lspca] = KSPCA(Xlab', Ylab', k, barshparam);
            %             Zspca = Zspca';
            %             %do prediction in learned basis
            %             betaSPCA = Zspca \ Ylab;
            %             Klab = Xlab*Xlab';
            %             Ktest = Xtest*Xlab';
            %             spcaXtest = Ktest*Lspca;
            %             spca_mbd_test{l,t} = spcaXtest;
            %             spca_mbd_lab{l,t} = Zspca;
            %             B = mnrfit(Zspca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
            %             [~,Yhat] = max(mnrval(B,spcaXtest),[], 2);
            %             [~,SPCAYlab] = max(mnrval(B,Zspca),[], 2);
            %             % compute error
            %             SPCArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
            %             SPCArates_lab(l,t) = 1 - sum(SPCAYlab == Ylab) / nlab;
            %             SPCAvar(l,t) = norm(spcaXtest, 'fro') / norm(Ktest, 'fro');
            %             SPCAvar_lab(l,t) = norm(Zspca, 'fro') / norm(Klab, 'fro');
            %         end
            spcaXtests{l,t} = spcaXtest;
            Zspcas{l,t} = Zspca;
            Barshantimes(l,t) = toc;
        end
            
            %% Perform Barshan's KSPCA based 2D embedding
            %learn basis
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
                    
                    tic
                    %calc with best param on full labing set
                    barshparam.ktype_y = 'delta';
                    barshparam.kparam_y = 1;
                    barshparam.ktype_x = 'rbf';
                    barshparam.kparam_x = sigma;
                    [Zkspca Lkspca] = KSPCA(Xlab', Ylab', k, barshparam);
                    Zkspca = Zkspca';
                    %do prediction in learned basis
                    betakSPCA = Zkspca \ Ylab;
                    Klab = gaussian_kernel(Xlab, [Xlab; Xunlab], sigma);
                    Ktest = gaussian_kernel(Xtest, [Xlab; Xunlab], sigma);
                    kspcaXtest = Ktest*Lkspca;
                    kspca_mbd_test{l,t,jj} = kspcaXtest;
                    kspca_mbd_lab{l,t,jj} = Zkspca;
                    B = mnrfit(Zkspca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
                    [~,Yhat] = max(mnrval(B,kspcaXtest),[], 2);
                    [~,kSPCAYlab] = max(mnrval(B,Zkspca),[], 2);
                    %compute error
                    kSPCArates(l,t,jj) = 1 - sum(Yhat == Ytest) / ntest;
                    kSPCArates_lab(l,t,jj) = 1 - sum(kSPCAYlab == Ylab) / nlab;
                    kSPCAvar(l,t,jj) = norm(kspcaXtest, 'fro') / norm(Ktest, 'fro');
                    kSPCAvar_lab(l,t,jj) = norm(Zkspca, 'fro') / norm(Klab, 'fro');
                    kBarshantimes(l,t,jj) = toc;
                    kspcaXtests{l,t,jj} = kspcaXtest;
                    Zkspcas{l,t,jj} = Zkspca;
                end
            end
            
            %% LDA
            % solve
            for l = 1:kfold+1
                test_num = l
                % get lth fold
                Xlab = Xlabs{l};
                Ylab = Ylabs{l};
                [nlab, ~] = size(Ylab);
                Xtest = Xtests{l};
                Ytest = Ytests{l};
                [ntest, ~] = size(Ytest);
                Mdl = fitcdiscr(Xlab,Ylab, 'DiscrimType', 'pseudolinear');
                LDAYtest = predict(Mdl,Xtest);
                % Predict
                LDAYlab = predict(Mdl,Xlab);
                %compute error
                LDArates(l,t) = 1 - sum(LDAYtest == Ytest) / ntest;
                LDArates_lab(l,t) = 1 - sum(LDAYlab == Ylab) / nlab;
                lin = Mdl.Coeffs(1,2).Linear / norm([Mdl.Coeffs(1,2).Const; Mdl.Coeffs(1,2).Linear]);
                const = Mdl.Coeffs(1,2).Const / norm([Mdl.Coeffs(1,2).Const; Mdl.Coeffs(1,2).Linear]);
                Zlda = Xlab*lin + const;
                LDAXtest = Xtest*lin + const;
                LDAvar(l,t) = norm(LDAXtest, 'fro') / norm(Xtest, 'fro');
                LDAvar_lab(l,t) = norm(Zlda, 'fro') / norm(Xlab, 'fro');
            end
            
            %% Local Fisher Discriminant Analysis (LFDA)
            for l = 1:kfold+1
                test_num = l
                % get lth fold
                Xlab = Xlabs{l};
                Ylab = Ylabs{l};
                [nlab, ~] = size(Ylab);
                Xtest = Xtests{l};
                Ytest = Ytests{l};
                [ntest, ~] = size(Ytest);
                K = Xlab*Xlab';
                Ktest = Xtest*Xlab';
                [Llfda,~] = KLFDA(K,Ylab,k, 'plain',9);
                %predict
                Llfda = orth(Llfda);
                Zlfda = K*Llfda;
                B = mnrfit(Zlfda,Ylab, 'Model', 'nominal', 'Interactions', 'on');
                LFDAXtest = Ktest*Llfda;
                [~,Yhat] = max(mnrval(B,LFDAXtest),[], 2);
                [~,LFDAYlab] = max(mnrval(B,Zlfda),[], 2);
                %compute error
                LFDArates(l,t) = 1 - sum(Yhat == Ytest) / ntest;
                LFDArates_lab(l,t) = 1 - sum(LFDAYlab == Ylab) / nlab;
                
                LFDAvar(l,t) = norm(LFDAXtest, 'fro') / norm(Ktest, 'fro');
                LFDAvar_lab(l,t) = norm(Zlfda, 'fro') / norm(K, 'fro');
            end
            
            %% Kernel Local Fisher Discriminant Analysis (KLFDA)
            %choose kernel param
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
                    %lab
                    K = gaussian_kernel(Xlab, [Xlab; Xunlab], sigma);
                    [Llfda,~] = KLFDA(K,Ylab,k, 'plain');
                    Llfda = orth(Llfda);
                    Zlfda = K*Llfda;
                    %predict
                    Ktest = gaussian_kernel(Xtest, [Xlab; Xunlab], sigma);
                    LFDAXtest = Ktest*Llfda;
                    B = mnrfit(Zlfda,Ylab, 'Model', 'nominal', 'Interactions', 'on');
                    [~,Yhat] = max(mnrval(B,LFDAXtest),[], 2);
                    [~,LFDAYlab] = max(mnrval(B,Zlfda),[], 2);
                    %compute error
                    kLFDArates(l,t,jj) = 1 - sum(Yhat == Ytest) / ntest;
                    kLFDArates_lab(l,t,jj) = 1 - sum(LFDAYlab == Ylab) / nlab;
                    kLFDAvar(l,t,jj) = norm(LFDAXtest, 'fro') / norm(Ktest, 'fro');
                    kLFDAvar_lab(l,t,jj) = norm(Zlfda, 'fro') / norm(K, 'fro');
                end
            end
            
            
        end
        

    
    %% compute avg performance accross folds
    
    avgPCA = mean(PCArates(1:end-1,:));
    avgPCA_lab = mean(PCArates_lab(1:end-1,:));
    avgkPCA = mean(kPCArates(1:end-1,:,:));
    avgkPCA_lab = mean(kPCArates_lab(1:end-1,:,:));
%     avgLSPCA = mean(LSPCArates(1:end-1,:,:));
%     avgLSPCA_lab = mean(LSPCArates_lab(1:end-1,:,:));
%     avgkLSPCA = mean(kLSPCArates(1:end-1,:,:,:));
%     avgkLSPCA_lab = mean(kLSPCArates_lab(1:end-1,:,:,:));
    avgLSPCAmle = mean(mle_LSPCArates(1:end-1,:), 1);
    avgLSPCAmle_lab = mean(mle_LSPCArates_lab(1:end-1,:), 1);
    avgkLSPCAmle = mean(mle_kLSPCArates(1:end-1,:,:), 1);
    avgkLSPCAmle_lab = mean(mle_kLSPCArates_lab(1:end-1,:,:), 1);
    avgSPCA = mean(SPCArates(1:end-1,:));
    avgSPCA_lab = mean(SPCArates_lab(1:end-1,:));
    avgkSPCA = mean(kSPCArates(1:end-1,:,:));
    avgkSPCA_lab = mean(kSPCArates_lab(1:end-1,:,:));
    avgISPCA = mean(ISPCArates(1:end-1,:));
    avgISPCA_lab = mean(ISPCArates_lab(1:end-1,:));
    avgSPPCA = mean(SPPCArates(1:end-1,:));
    avgSPPCA_lab = mean(SPPCArates_lab(1:end-1,:));
    avgLDA = mean(LDArates(1:end-1,:));
    avgLDA_lab = mean(LDArates_lab(1:end-1,:));
    avgLFDA = mean(LFDArates(1:end-1,:));
    avgLFDA_lab = mean(LFDArates_lab(1:end-1,:));
    avgkLFDA = mean(kLFDArates(1:end-1,:,:));
    avgkLFDA_lab = mean(kLFDArates_lab(1:end-1,:,:));
    %
    avgPCAvar = mean(PCAvar(1:end-1,:));
    avgkPCAvar = mean(kPCAvar(1:end-1,:,:));
%     avgLSPCAvar = mean(LSPCAvar(1:end-1,:,:));
%     avgkLSPCAvar = mean(kLSPCAvar(1:end-1,:,:,:));
    avgLSPCAmlevar = mean(mle_LSPCAvar(1:end-1,:), 1);
    avgkLSPCAmlevar = mean(mle_kLSPCAvar(1:end-1,:,:), 1);
    avgSPCAvar = mean(SPCAvar(1:end-1,:));
    avgkSPCAvar = mean(kSPCAvar(1:end-1,:,:));
    avgISPCAvar = mean(ISPCAvar(1:end-1,:));
    avgSPPCAvar = mean(SPPCAvar(1:end-1,:));
    avgLDAvar = mean(LDAvar(1:end-1,:));
    avgLFDAvar = mean(LFDAvar(1:end-1,:));
    avgkLFDAvar = mean(kLFDAvar(1:end-1,:,:));
    %
    avgPCAvar_lab = mean(PCAvar_lab(1:end-1,:));
    avgkPCAvar_lab = mean(kPCAvar_lab(1:end-1,:,:));
%     avgLSPCAvar_lab = mean(LSPCAvar_lab(1:end-1,:,:));
%     avgkLSPCAvar_lab = mean(kLSPCAvar_lab(1:end-1,:,:,:));
    avgLSPCAmlevar_lab = mean(mle_LSPCAvar_lab(1:end-1,:), 1);
    avgkLSPCAmlevar_lab = mean(mle_kLSPCAvar_lab(1:end-1,:,:), 1);
    avgSPCAvar_lab = mean(SPCAvar_lab(1:end-1,:));
    avgkSPCAvar_lab = mean(kSPCAvar_lab(1:end-1,:,:));
    avgISPCAvar_lab = mean(ISPCAvar_lab(1:end-1,:));
    avgSPPCAvar_lab = mean(SPPCAvar_lab(1:end-1,:));
    avgLDAvar_lab = mean(LDAvar_lab(1:end-1,:));
    avgLFDAvar_lab = mean(LFDAvar_lab(1:end-1,:));
    avgkLFDAvar_lab = mean(kLFDAvar_lab(1:end-1,:,:));
    
     %% Calc performance for best model and store
    
    % cv over subspace dim
    loc = find(avgPCA==min(avgPCA,[],'all'),1,'last');
    [~,kloc] = ind2sub(size(avgPCA), loc);
    kpca = ks(kloc);
    PCAval(dd) = PCArates(end,kloc);
    PCAvalVar(dd) = PCAvar(end,kloc);
    
%     loc = find(avgLSPCA==min(avgLSPCA,[],'all'),1,'last');
%     [~,kloc,lamloc] = ind2sub(size(avgLSPCA), loc);
%     klspca = ks(kloc);
%     LSPCAval(dd) = LSPCArates(end,kloc,lamloc);
%     LSPCAvalVar(dd) = LSPCAvar(end,kloc,lamloc);
%     
%     
%     loc = find(avgkLSPCA==min(avgkLSPCA,[],'all'),1,'last');
%     [~,klock,lamlock,siglock] = ind2sub(size(avgkLSPCA), loc);
%     kklspca = ks(klock);
%     kLSPCAval(dd) = kLSPCArates(end,klock,lamlock,siglock);
%     kLSPCAvalVar(dd) = kLSPCAvar(end,klock,lamlock,siglock);
    
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
    
    loc = find(avgLDA==min(avgLDA,[],'all'),1,'last');
    [~,kloc] = ind2sub(size(avgLDA), loc);
    klda = ks(kloc);
    LDAval(dd) = LDArates(end,kloc);
    LDAvalVar(dd) = LDAvar(end,kloc);
    
    loc = find(avgLFDA==min(avgLFDA,[],'all'),1,'last');
    [~,kloc] = ind2sub(size(avgLFDA), loc);
    klfda = ks(kloc);
    LFDAval(dd) = LFDArates(end,kloc);
    LFDAvalVar(dd) = LFDAvar(end,kloc);
    
    loc = find(avgkLFDA==min(avgkLFDA,[],'all'),1,'last');
    [~,kloc,sigloc] = ind2sub(size(avgkLFDA), loc);
    kklfda = ks(kloc);
    kLFDAval(dd) = kLFDArates(end,kloc,sigloc);
    kLFDAvalVar(dd) = kLFDAvar(end,kloc,sigloc);

    %fixed subspace dimension k=2
    
    kloc=1; %k=2
    
    kpca = ks(kloc);
    PCAval_fixed(dd) = PCArates(end,kloc);
    PCAvalVar_fixed(dd) = PCAvar(end,kloc);
    
%     loc = find(avgLSPCA(:,kloc,:)==min(avgLSPCA(:,kloc,:),[],'all'),1,'last');
%     [~,~,lamloc] = ind2sub(size(avgLSPCA(:,kloc,:)), loc);
%     klspca = ks(kloc);
%     LSPCAval_fixed(dd) = LSPCArates(end,kloc,lamloc);
%     LSPCAvalVar_fixed(dd) = LSPCAvar(end,kloc,lamloc);
%     
%     loc = find(avgkLSPCA(:,kloc,:,:)==min(avgkLSPCA(:,kloc,:,:),[],'all'),1,'last');
%     [~,~,lamlock,siglock] = ind2sub(size(avgkLSPCA(:,kloc,:,:)), loc);
%     klock=kloc;
%     kklspca = ks(klock);
%     kLSPCAval_fixed(dd) = kLSPCArates(end,klock,lamlock,siglock);
%     kLSPCAvalVar_fixed(dd) = kLSPCAvar(end,klock,lamlock,siglock);
    
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
    
    klda = ks(kloc);
    LDAval_fixed(dd) = LDArates(end,kloc);
    LDAvalVar_fixed(dd) = LDAvar(end,kloc);
    
    klfda = ks(kloc);
    LFDAval_fixed(dd) = LFDArates(end,kloc);
    LFDAvalVar_fixed(dd) = LFDAvar(end,kloc);
    
    loc = find(avgkLFDA==min(avgkLFDA,[],'all'),1,'last');
    [~,~,sigloc] = ind2sub(size(avgkLFDA), loc);
    kklfda = ks(kloc);
    kLFDAval_fixed(dd) = kLFDArates(end,kloc,sigloc);
    kLFDAvalVar_fixed(dd) = kLFDAvar(end,kloc,sigloc);
    
end
%% save all data
save(strcat(dataset, '_results_dim'))

%% print mean performance with std errors
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

m = mean(LDAval);
v = mean(LDAvalVar);
sm = std(LDAval);
sv = std(LDAvalVar);
sprintf('LDAerr: $%0.3f \\pm %0.3f $ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(LFDAval);
v = mean(LFDAvalVar);
sm = std(LFDAval);
sv = std(LFDAvalVar);
sprintf('LFDAerr: $%0.3f \\pm %0.3f $ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
% 
% m = mean(LSPCAval);
% v = mean(LSPCAvalVar);
% sm = std(LSPCAval);
% sv = std(LSPCAvalVar);
% sprintf('LSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

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

m = mean(kLFDAval);
v = mean(kLFDAvalVar);
sm = std(kLFDAval);
sv = std(kLFDAvalVar);
sprintf('kLFDAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
% 
% m = mean(kLSPCAval);
% v = mean(kLSPCAvalVar);
% sm = std(kLSPCAval);
% sv = std(kLSPCAvalVar);
% sprintf('kLSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(mle_kLSPCAval);
v = mean(mle_kLSPCAvalVar);
sm = std(mle_kLSPCAval);
sv = std(mle_kLSPCAvalVar);
sprintf('mle_kLSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)



%% Print results with k=2

kloc=1; %k=2
k = ks(kloc);

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

m = mean(LDAval_fixed);
v = mean(LDAvalVar_fixed);
sm = std(LDAval_fixed);
sv = std(LDAvalVar_fixed);
sprintf('LDAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(LFDAval_fixed);
v = mean(LFDAvalVar_fixed);
sm = std(LFDAval_fixed);
sv = std(LFDAvalVar_fixed);
sprintf('LFDAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

% m = mean(LSPCAval_fixed);
% v = mean(LSPCAvalVar_fixed);
% sm = std(LSPCAval_fixed);
% sv = std(LSPCAvalVar_fixed);
% sprintf('LSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

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

m = mean(kLFDAval_fixed);
v = mean(kLFDAvalVar_fixed);
sm = std(kLFDAval_fixed);
sv = std(kLFDAvalVar_fixed);
sprintf('kLFDAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

% m = mean(kLSPCAval_fixed);
% v = mean(kLSPCAvalVar_fixed);
% sm = std(kLSPCAval_fixed);
% sv = std(kLSPCAvalVar_fixed);
% sprintf('kLSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)

m = mean(mle_kLSPCAval_fixed);
v = mean(mle_kLSPCAvalVar_fixed);
sm = std(mle_kLSPCAval_fixed);
sv = std(mle_kLSPCAvalVar_fixed);
sprintf('mle_kLSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)




% %% plot error - var tradeoff curves
% %avgLSPCA_lab = avgLSPCA_lab(:,:,1:5,:);
% %avgLSPCAvar_lab = avgLSPCAvar_lab(:,:,1:5,:);
% 
% kloc = find(ks==klspca);
% t = kloc;
% temp = avgLSPCA(:,t,:,:);
% temp = reshape(temp, [length(Gammas), length(Lambdas)]);
% %temp_lab = reshape(avgLSPCA_lab(:,t,:,:), [length(Gammas), length(Lambdas)]);
% tempv = avgLSPCAvar(:,t,:,:);
% tempv = reshape(tempv, [length(Gammas), length(Lambdas)]);
% %tempv_lab = reshape(avgLSPCAvar_lab(:,t,:,:), [length(Gammas), length(Lambdas)]);
% [mLSPCA, I] = min(temp);
% I = sub2ind(size(temp),I,1:length(I));
% vLSPCA = tempv(I);
% %mLSPCA_lab = temp_lab(I);
% %vLSPCA_lab = tempv_lab(I);
% 
% kloc = find(ks==kklspca);
% t = kloc;
% temp = avgkLSPCA(:,t,:,:,:);
% temp = reshape(temp, [length(Gammas), length(Lambdas), length(sigmas)]);
% %temp_lab = reshape(avgkLSPCA_lab(:,t,:,:), [length(Gammas), length(Lambdas), length(sigmas)]);
% tempv = avgkLSPCAvar(:,t,:,:,:);
% tempv = reshape(tempv, [length(Gammas), length(Lambdas), length(sigmas)]);
% %tempv_lab = reshape(avgkLSPCAvar_lab(:,t,:,:), [length(Gammas), length(Lambdas), length(sigmas)]);
% [res, I1] = min(temp, [], 1);
% [~,a,b] = size(I1);
% for i=1:a
%     for j=1:b
%         tempvv(i,j) = tempv(I1(1,i,j),i,j);
%         %    tempmm_lab(i,j) = temp_lab(I1(1,i,j),i,j);
%         %    tempvv_lab(i,j) = tempv_lab(I1(1,i,j),i,j);
%     end
% end
% [mkLSPCA, I2] = min(res, [], 3);
% [~,c] = size(I2);
% for i=1:c
%     vkLSPCA(i) = tempvv(i,10);
%     %     mkLSPCA_lab(i) = tempmm_lab(i,I2(1,i));
%     %     vkLSPCA_lab(i) = tempvv_lab(i,I2(1,i));
% end
% %     gamloc = 3;
% %     gamlock = 3;
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
% plot(avgLDAvar(klda-1), avgLDA(klda-1), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% % plot([0,1], [1,1]*avgQDA(t), ':', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgLFDAvar(klfda-1), avgLFDA(klfda-1), '^', 'MarkerSize', 20, 'LineWidth', 2)
% loc = find(avgkLFDA(:,kklfda-1,:)==min(avgkLFDA(:,kklfda-1,:),[],'all'),1,'last');
% [~,~,sigloc] = ind2sub(size(avgkLFDA(:,kklfda-1,:)), loc);
% plot(avgkLFDAvar(1,kklfda-1,sigloc), avgkLFDA(1,kklfda-1,sigloc), '<', 'MarkerSize', 20, 'LineWidth', 2)
% 
% %plot([0,1], avgLR*[1,1], 'k', 'LineWidth', 4)
% 
% %xlabel('Variation Explained', 'fontsize', 25)
% %title('Test', 'fontsize', 25)
% %ylabel('Classification Error', 'fontsize', 25)
% %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
% set(gca, 'fontsize', 25)
% lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', ...
%     'LDA', 'LFDA', 'kLFDA', 'Location', 'best'); lgd.FontSize = 25;
% xlim([0,1.01])
% %ylim([0,0.5])
% %saveas(gcf, strcat(dataset, 'multi_obj.jpg'))
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
% 
% 
% %%
% hold on
% loc = find(avgLSPCA == min(avgLSPCA, [], 'all'));
% plot(avgLSPCAvar(loc), avgLSPCA(loc), 'b*', 'LineWidth', 2, 'MarkerSize', 20)
% %%
% hold on
% loc = find(avgkLSPCA == min(avgkLSPCA, [], 'all'), 1, 'last');
% plot(avgkLSPCAvar(loc), avgkLSPCA(loc), 'r*', 'LineWidth', 2, 'MarkerSize', 20)
% lgd = legend('PCA', 'LSPCA (CV)', 'kLSPCA (CV)', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', ...
%     'LDA', 'LFDA', 'kLFDA', 'LSPCA (MLE)', 'kLSPCA (MLE)', 'Location', 'best'); lgd.FontSize = 25;
% 
% 
% %
% %% plot error - var tradeoff curves
% kloc = 1;
% t = kloc;
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
% 
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
%     mkLSPCA_lab(i) = tempmm_lab(i,I2(1,i));
%     vkLSPCA_lab(i) = tempvv_lab(i,I2(1,i));
% end
% %
% % for t = 1:length(ks)
% figure()
% hold on
% plot(avgPCAvar_lab(1,kpca-1), avgPCA_lab(1,kpca-1), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% %plot(avgkPCAvar_lab(1,t), avgkPCA(1,t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% plot(vLSPCA_lab(:), mLSPCA_lab(:), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% % plot(lambda_avgLSPCAvar_lab(1,t,:), lambda_avgLSPCA(1,t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(vkLSPCA_lab(:), mkLSPCA_lab(:), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(avgISPCAvar_lab(1,kispca-1), avgISPCA_lab(1,kispca-1), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgSPPCAva_labr(1,ksppca-1), avgSPPCA_lab(1,ksppca-1), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgSPCAvar_lab(1,kspca-1), avgSPCA_lab(1,kspca-1), '+', 'MarkerSize', 20, 'LineWidth', 2)
% loc = find(avgkSPCA(:,kkspca-1,:)==min(avgkSPCA(:,kkspca-1,:),[],'all'),1,'last');
% [~,~,sigloc] = ind2sub(size(avgkSPCA(:,kkspca-1,:)), loc);
% plot(avgkSPCAvar_lab(1,kkspca-1,sigloc), avgkSPCA_lab(1,kkspca-1,sigloc), '>', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgLDAvar_lab(klda), avgLDA_lab(klda), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% % plot([0,1], [1,1]*avgQDA(t), ':', 'MarkerSize', 20, 'LineWidth', 2)
% plot(avgLFDAvar_lab(klfda), avgLFDA_lab(klfda), '^', 'MarkerSize', 20, 'LineWidth', 2)
% loc = find(avgkLFDA(:,klfda,:)==min(avgkLFDA(:,klfda,:),[],'all'),1,'last');
% [~,~,sigloc] = ind2sub(size(avgkLFDA(:,klfda,:)), loc);
% plot(avgkLFDAvar_lab(1,klfda,sigloc), avgkLFDA_lab(1,klfda,sigloc), '<', 'MarkerSize', 20, 'LineWidth', 2)
% 
% %plot([0,1], avgLR*[1,1], 'k', 'LineWidth', 4)
% 
% xlabel('Variation Explained', 'fontsize', 25)
% %title('lab', 'fontsize', 25)
% ylabel('Classification Error', 'fontsize', 25)
% %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
% set(gca, 'fontsize', 25)
% lgd = legend('PCA', 'LSPCA', 'KLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', ...
%     'LDA', 'LFDA', 'kLFDA', 'Location', 'best'); lgd.FontSize = 15;
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
% %     xlim([0,1.01])
% %     saveas(gcf, strcat(dataset, 'multi_obj_lab.jpg'))
% % end
% %
% %
% %
% % %% Visualize for select methods (only really good for classification)
% % load('colorblind_colormap.mat')
% % colormap(colorblind)
% %
% % %KLSPCA
% % figure()
% % loc = find(kLSPCArates == min(kLSPCArates, [], 'all'), 1, 'last');
% % [idx, ~, ~, ~] = ind2sub(size(kLSPCArates),loc);
% % data = klspca_mbd_lab{loc};
% % scatter(data(:,1), data(:,2), 100, Ylabs{idx}, 'filled', 'linewidth', 3)
% % %scatter3(data(:,1), data(:,2), data(:,3), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% % hold on
% % data = klspca_mbd_test{loc};
% % scatter(data(:,1), data(:,2), 100, Ytests{idx})
% % %scatter3(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % colormap(colorblind)
% % set(gca, 'yticklabel', '')
% % set(gca, 'xticklabel', '')
% % title(strcat('KLSPCA err:  ', num2str(min(kLSPCArates, [], 'all'))))
% % grid on; set(gca, 'fontsize', 25)
% % saveas(gcf, strcat(dataset, 'KLSPCA.jpg'))
% %
% % %LSPCA
% % figure()
% % loc = find(LSPCArates == min(LSPCArates, [], 'all'), 1, 'last');
% % [idx, ~, ~, ~] = ind2sub(size(kLSPCArates),loc);
% % data = lspca_mbd_lab{loc};
% % scatter(data(:,1), data(:,2), 100, Ylabs{idx}, 'filled', 'linewidth', 3)
% % %scatter3(data(:,1), data(:,2), data(:,3), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% % hold on
% % data = lspca_mbd_test{loc};
% % scatter(data(:,1), data(:,2), 100, Ytests{idx})
% % %scatter3(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % colormap(colorblind)
% % set(gca, 'yticklabel', '')
% % set(gca, 'xticklabel', '')
% % title(strcat('LSPCA err:  ', num2str(min(LSPCArates, [], 'all'))))
% % grid on; set(gca, 'fontsize', 25)
% % saveas(gcf, strcat(dataset, 'LSPCA.jpg'))
% %
% % % %LSPCA
% % % figure()
% % % [r,row] = min(min(ILSPCArates, [], 2));
% % % [r,col] = min(min(ILSPCArates, [], 1));
% % % data = Ilspca_mbd_lab{row,col};
% % % %scatter(data(:,1), data(:,2), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% % % scatter3(data(:,1), data(:,2), data(:,3), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% % % hold on
% % % data = Ilspca_mbd_test{row,col};
% % % %scatter(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % % scatter3(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % % colormap(colorblind)
% % % set(gca, 'yticklabel', '')
% % % set(gca, 'xticklabel', '')
% % % title(strcat('LSPCA err:  ', num2str(min(LSPCArates, [], 'all'))))
% % % grid on; set(gca, 'fontsize', 25)
% % % saveas(gcf, strcat(dataset, 'LSPCA_gamma.jpg'))
% % %
% % % %lambda LSPCA
% % % figure()
% % % [r,row] = min(min(lambda_LSPCArates, [], 2));
% % % [r,col] = min(min(lambda_LSPCArates, [], 1));
% % % data = lambda_lspca_mbd_lab{row,col};
% % % %scatter(data(:,1), data(:,2), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% % % scatter3(data(:,1), data(:,2), data(:,3), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% % % hold on
% % % data = lambda_lspca_mbd_test{row,col};
% % % %scatter(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % % scatter3(data(:,1), data(:,2), data(:,3), 100, Ytests(:,:,row))
% % % colormap(colorblind)
% % % set(gca, 'yticklabel', '')
% % % set(gca, 'xticklabel', '')
% % % title(strcat('lambda LSPCA err:  ', num2str(min(LSPCArates, [], 'all'))))
% % % grid on; set(gca, 'fontsize', 25)
% % % saveas(gcf, strcat(dataset, 'LSPCA_gamma.jpg'))
% %
% % %PCA
% % figure()
% % [r,loc] = min(PCArates);
% % data = Zpcas{loc};
% % scatter(data(:,1), data(:,2), 100, Ylabs{loc}, 'filled', 'linewidth', 3)
% % hold on
% % data = pcaXtests{loc};
% % scatter(data(:,1), data(:,2), 100, Ytests{loc})
% % colormap(colorblind)
% % set(gca, 'yticklabel', '')
% % set(gca, 'xticklabel', '')
% % title(strcat('PCA err:  ', num2str(r)))
% % grid on; set(gca, 'fontsize', 25)
% % saveas(gcf, strcat(dataset, 'PCA.jpg'))
% %
% % %Barshan
% % figure()
% % [r,loc] = min(SPCArates);
% % data = Zspcas{loc};
% % scatter(data(:,1), data(:,2), 100, Ylabs{loc}, 'filled', 'linewidth', 3)
% % hold on
% % data = spcaXtests{loc};
% % scatter(data(:,1), data(:,2), 100, Ytests{loc})
% % colormap(colorblind)
% % set(gca, 'yticklabel', '')
% % set(gca, 'xticklabel', '')
% % title(strcat('Barshan err:  ', num2str(r)))
% % grid on; set(gca, 'fontsize', 25)
% % saveas(gcf, strcat(dataset, 'Barshan.jpg'))
% %
% % %kBarshan
% % figure()
% % loc = find(kSPCArates == min(kSPCArates, [], 'all'), 1, 'last');
% % [idx, ~, ~] = ind2sub(size(kSPCArates),loc);
% % data = Zkspcas{loc};
% % scatter(data(:,1), data(:,2), 100, Ylabs{idx}, 'filled', 'linewidth', 3)
% % hold on
% % data = kspcaXtests{loc};
% % scatter(data(:,1), data(:,2), 100, Ytests{idx})
% % colormap(colorblind)
% % set(gca, 'yticklabel', '')
% % set(gca, 'xticklabel', '')
% % title(strcat('kBarshan err:  ', num2str(r)))
% % grid on; set(gca, 'fontsize', 25)
% % saveas(gcf, strcat(dataset, 'kBarshan.jpg'))
% %
% % %ISPCA
% % figure()
% % [r,loc] = min(ISPCArates);
% % data = Zispcas{loc};
% % scatter(data(:,1), data(:,2), 100, Ylabs{loc}, 'filled', 'linewidth', 3)
% % hold on
% % data = ISPCAXtests{loc};
% % scatter(data(:,1), data(:,2), 100, Ytests{loc})
% % colormap(colorblind)
% % set(gca, 'yticklabel', '')
% % set(gca, 'xticklabel', '')
% % title(strcat('ISPCA err:  ', num2str(r)))
% % grid on; set(gca, 'fontsize', 25)
% % saveas(gcf, strcat(dataset, 'ISPCA.jpg'))
% %
% % %SPPCA
% % figure()
% % [r,loc] = min(SPPCArates);
% % data = Zsppcas{loc};
% % scatter(data(:,1), data(:,2), 100, Ylabs{loc}, 'filled', 'linewidth', 3)
% % hold on
% % data = SPPCAXtests{loc};
% % scatter(data(:,1), data(:,2), 100, Ytests{loc})
% % colormap(colorblind)
% % set(gca, 'yticklabel', '')
% % set(gca, 'xticklabel', '')
% % title(strcat('SPPCA err:  ', num2str(r)))
% % grid on; set(gca, 'fontsize', 25)
% % saveas(gcf, strcat(dataset, 'SPPCA.jpg'))
% %
% % %LFDA
% % figure()
% % [r,loc] = min(SPPCArates);
% % data = Zsppcas{loc};
% % scatter(data(:,1), data(:,2), 100, Ylabs{loc}, 'filled', 'linewidth', 3)
% % hold on
% % data = SPPCAXtests{loc};
% % scatter(data(:,1), data(:,2), 100, Ytests{loc})
% % colormap(colorblind)
% % set(gca, 'yticklabel', '')
% % set(gca, 'xticklabel', '')
% % title(strcat('SPPCA err:  ', num2str(r)))
% % grid on; set(gca, 'fontsize', 25)
% % saveas(gcf, strcat(dataset, 'SPPCA.jpg'))
% %
% % %kLFDA
% % figure()
% % [r,loc] = min(SPPCArates);
% % data = Zsppcas{loc};
% % scatter(data(:,1), data(:,2), 100, Ylabs{loc}, 'filled', 'linewidth', 3)
% % hold on
% % data = SPPCAXtests{loc};
% % scatter(data(:,1), data(:,2), 100, Ytests{loc})
% % colormap(colorblind)
% % set(gca, 'yticklabel', '')
% % set(gca, 'xticklabel', '')
% % title(strcat('SPPCA err:  ', num2str(r)))
% % grid on; set(gca, 'fontsize', 25)
% % saveas(gcf, strcat(dataset, 'SPPCA.jpg'))





%% setup and load data
%rng(0);
ks = 2:10;
load(strcat(dataset, '.mat'));
[n, p] = size(X);
[~, q] = size(Y);


% create same splits to use every time and center
kfold = 20;
cvidx = crossvalind('kfold',n,kfold);
for l = 1:kfold
    Xlab = X(cvidx==l, :); %lth labled set
    Ylab = Y(cvidx==l, :);
    Xunlab = X(cvidx~=l, :); %lth unlabled set
    Yunlab = Y(cvidx~=l, :);
    [Xlab,Xunlab,Ylab,Yunlab] = ss_center_data(Xlab,Xunlab,Ylab,Yunlab,'regression');
    Xlabs{l} = Xlab; %lth centered labled set
    Ylabs{l} = Ylab;
    Xunlabs{l} = Xunlab; %lth centered unlabled set
    Yunlabs{l} = Yunlab;
end



for t = 1:length(ks) %dimensionality of reduced data
    k = ks(t)
    
    %% PCA
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nlab, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nunlab, ~] = size(Yunlab);
        
        tic
        [Lpca, Zpca] = pca([Xlab;Xunlab], 'NumComponents', k);
        % compute embedding for unlab data
        pcaXunlab = Zpca(nlab+1:end,:);
        pcaXunlabs{l,t} = pcaXunlab;
        PCAtimes(l,t,t) = toc;
        Zpca = Zpca(1:nlab,:);
        Zpcas{l,t} = Zpca;
        Lpca = Lpca';
        % compute error
        B = Zpca \ Ylab;
        PCRYlab = Zpca*B;
        PCRYunlab = pcaXunlab*B;
        mse = norm(PCRYunlab - Yunlab, 'fro')^2 /nunlab;
        PCArates(l,t) = mse ;
        PCArates_lab(l,t) = norm(PCRYlab - Ylab, 'fro')^2 / nlab;
        PCAvar(l,t) = norm(Xunlab*Lpca', 'fro') / norm(Xunlab, 'fro');
        PCAvar_lab(l,t) = norm(Xlab*Lpca', 'fro') / norm(Xlab, 'fro');
    end
    
    %% PLS
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nlab, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nunlab, ~] = size(Yunlab);
        %solve for basis
        [Xloadings,Yloadings,Xscores,Yscores,betaPLS,pctVar,~,stats] = plsregress(Xlab,Ylab,k);
        Lpls = orth(stats.W)';
        Lpls = Lpls(1:k, :);
        % predict
        PLSYunlab = [ones(nunlab,1) Xunlab]*betaPLS;
        PLSYlab = [ones(nlab,1) Xlab]*betaPLS;
        % compute error
        mse = norm(PLSYunlab - Yunlab, 'fro')^2;
        PLSrates(l,t) = mse / nunlab;
        PLSrates_lab(l,t) = norm(PLSYlab - Ylab, 'fro')^2 / nlab;
        PLSvar(l,t) = norm(Xunlab*Lpls', 'fro') / norm(Xlab, 'fro');
        PLSvar_lab(l,t) = sum(pctVar(1,:));
    end
    
    
    %% LSPCA
    
    %Lambda = 5;
    Lambdas = linspace(1,10,5);
    %Gammas = logspace(log10(3), log10(0.5), 51);
    Gammas = [linspace(1, 0.7, 10), logspace(log10(0.7),log10(0.5058), 30)];
    
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nlab, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nunlab, ~] = size(Yunlab);
        
        %solve
        for jj = 1:length(Lambdas)
            Lambda = Lambdas(jj)
            for ii = 1:length(Gammas)
                Gamma = Gammas(ii);
                if ii == 1
                    [LSPCAXunlab, LSPCAYunlab, Zlspca, Llspca, B] = lspca_gamma_ss(Xunlab, Xlab, Ylab, Lambda, Gamma, k, 0);
                    
                else
                    [LSPCAXunlab, LSPCAYunlab, Zlspca, Llspca, B] = lspca_gamma_ss(Xunlab, Xlab, Ylab, Lambda, Gamma, k, Llspca);
                    
                end
                Ls{l,t,ii, jj} = Llspca;
                %predict
                LSPCAXlab = Zlspca;
                LSPCAYlab = LSPCAXlab*B;
                lspca_mbd_unlab{l,t,ii,jj} = LSPCAXunlab;
                lspca_mbd_lab{l,t,ii,jj} = Zlspca;
                mse = norm(LSPCAYunlab - Yunlab, 'fro')^2 / nunlab;
                % compute error
                lab_err = norm(Ylab - LSPCAYlab, 'fro')^2 / nlab;
                LSPCArates(l,t, ii, jj) = mse ;
                LSPCArates_lab(l,t, ii, jj) = lab_err;
                LSPCAvar(l,t, ii, jj) = norm(LSPCAXunlab, 'fro') / norm(Xunlab, 'fro');
                LSPCAvar_lab(l,t, ii, jj) = norm(Zlspca, 'fro') / norm(Xlab, 'fro');
            end
        end
    end
    
    %% kLSPCA
    %Lambda = 5;
    Lambdas = linspace(1,10,5);
    %Gammas = logspace(log10(3), log10(0.5), 51);
    Gammas = [linspace(1, 0.7, 10), logspace(log10(0.7),log10(0.5058), 30)];
    
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nlab, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nunlab, ~] = size(Yunlab);
        
        for kk = 1:length(sigmas)
            sigma = sigmas(kk)
            for jj = 1:length(Lambdas)
                Lambda = Lambdas(jj);
                for ii = 1:length(Gammas)
                    Gamma = Gammas(ii);
                    if ii == 1
                        [kLSPCAXunlab, kLSPCAYunlab, Zklspca, Lorth, B, Klspca] = klspca_gamma_ss(Xunlab, Xlab, Ylab, Lambda, Gamma, sigma, k, 0, 0);
                        
                    else
                        [kLSPCAXunlab, kLSPCAYunlab, Zklspca, Lorth, B, Klspca] = klspca_gamma_ss(Xunlab, Xlab, Ylab, Lambda, Gamma, sigma, k, Lorth, Klspca);
                        
                    end
                    %             Lorth = Lorth';
                    kLs{l,t,ii,jj,kk} = Lorth;
                    klspca_mbd_unlab{l,t,ii,jj,kk} = kLSPCAXunlab;
                    kLSPCAXlab = Zklspca;
                    klspca_mbd_lab{l,t,ii,jj,kk} = Zklspca;
                    kLSPCAYlab = kLSPCAXlab*B;
                    mse = norm(kLSPCAYunlab - Yunlab, 'fro')^2 / nunlab;
                    lab_err = norm(Ylab - kLSPCAYlab, 'fro')^2 / nlab;
                    kLSPCArates(l,t,ii,jj,kk) = mse ;
                    kLSPCArates_lab(l,t,ii,jj,kk) = lab_err;
                    kLSPCAvar(l,t,ii,jj,kk) = norm(kLSPCAXunlab, 'fro') / norm(gaussian_kernel(Xunlab,Xlab,sigma), 'fro');
                    kLSPCAvar_lab(l,t,ii,jj,kk) = norm(Zklspca, 'fro') / norm(Klspca, 'fro');
                end
            end
        end
    end
    
    %% KPCA
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nlab, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nunlab, ~] = size(Yunlab);
        for jj = 1:length(sigmas)
            sigma = sigmas(jj)
            K = gaussian_kernel([Xlab;Xunlab], [Xlab;Xunlab], sigma);
            Klab = K(1:nlab,:);
            Kunlab = K(nlab+1:end,:);
            tic
            [Lkpca, Zkpca] = pca(K, 'NumComponents', k);
            kPCAtimes(l,t,t,jj) = toc;
            Zkpcas{l,t,jj} = Zkpca(1:nlab,:);
            Lkpca = Lkpca';
            % compute embedding for unlab data
            kpcaXunlab = Zkpca(nlab+1:end,:);
            kpcaXunlabs{l,t,jj} = kpcaXunlab;
            % compute error
            B = Zkpca(1:nlab,:) \ Ylab;
            kpcaYunlab = kpcaXunlab*B;
            kpcaYlab = Zkpca(1:nlab,:)*B;
            mse = norm(kpcaYunlab - Yunlab, 'fro')^2 / nunlab;
            lab_err = norm(Ylab - kpcaYlab, 'fro')^2 / nlab;
            kPCArates(l,t,jj) = mse;
            kPCArates_lab(l,t,jj) = lab_err;
            kPCAvar(l,t,jj) = norm(kpcaXunlab, 'fro') / norm(Kunlab, 'fro');
            kPCAvar_lab(l,t,jj) = norm(Zkpca(1:nlab,:), 'fro') / norm(K, 'fro');
        end
    end
    
    %% R4
    
    if ~strcmp(dataset, 'DLBCL')
        params = logspace(-6, 6, 101);
        params(1) = 0; %edge case is just RRR
        ridge_solns = [];
        for l = 1:kfold
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            [nlab, ~] = size(Ylab);
            Xunlab = Xunlabs{l};
            Yunlab = Yunlabs{l};
            [nunlab, ~] = size(Yunlab);
            
            for pp = 1:length(params)
                param = params(pp)
                [Mrrr, Lrrr] = rrr_ridge(Xlab, Ylab, k, param);
                ridge_solns(:,:,pp) = Lrrr;
            end
            ridge_rrr_solns{l,t} = ridge_solns
            Ls_rrr = ridge_rrr_solns{l,t};
            [~, ~, numsols] = size(Ls_rrr);
            for tt = 1:numsols
                b = (Xlab*Ls_rrr(:,:,tt)') \ Ylab;
                ridge_rrr_err = norm(Yunlab - Xunlab*Ls_rrr(:,:,tt)'*b, 'fro')^2 / nunlab;
                ridge_rrr_err_lab = norm(Ylab - Xlab*Ls_rrr(:,:,tt)'*b, 'fro')^2 / nlab;
                ridge_rrr_var = norm(Xunlab*Ls_rrr(:,:,tt)', 'fro') / norm(Xunlab, 'fro');
                ridge_rrr_var_lab = norm(Xlab*Ls_rrr(:,:,tt)', 'fro') / norm(Xlab, 'fro');
                ridge_rrr_rates(l,t, tt) = ridge_rrr_err;
                ridge_rrr_rates_lab(l,t,tt) = ridge_rrr_err_lab;
                ridge_rrrvars(l,t,tt) = ridge_rrr_var;
                ridge_rrrvars_lab(l,t, tt) = ridge_rrr_var_lab;
                Lridge_rrrs{l, t, tt} = Ls_rrr(:,:,tt);
            end
        end
    else
        ridge_rrr_rates(l,t, tt) = nan;
        ridge_rrr_rates_lab(l,t,tt) = nan;
        ridge_rrrvars(l,t,tt) = nan;
        ridge_rrrvars_lab(l,t, tt) = nan;
    end
    
    %% ISPCA
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nlab, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nunlab, ~] = size(Yunlab);
        %find basis
        tic
        [Zispca, Lispca, B] = ISPCA(Xlab,Ylab,k);
        ISPCAtimes(l,t) = toc;
        % predict
        ISPCAXunlab = Xunlab*Lispca';
        ISPCAYunlab = ISPCAXunlab*B;
        ISPCAYlab = Zispca*B;
        % compute error
        mse = norm(ISPCAYunlab - Yunlab, 'fro')^2 / nunlab;
        ISPCArates(l,t) = mse ;
        ISPCArates_lab(l,t) = norm(ISPCAYlab - Ylab, 'fro')^2 / nlab;
        ISPCAvar(l,t) = norm(Xunlab*Lispca', 'fro') / norm(Xunlab, 'fro');
        ISPCAvar_lab(l,t) = norm(Xlab*Lispca', 'fro') / norm(Xlab, 'fro');
    end
    
    %% SPPCA
    if ~strcmp(dataset, 'DLBCL')
        for l = 1:kfold
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            [nlab, ~] = size(Ylab);
            Xunlab = Xunlabs{l};
            Yunlab = Yunlabs{l};
            [nunlab, ~] = size(Yunlab);
            
            tic
            SPPCAXunlab = {};
            SPPCAYunlab = {};
            SPPCAYlab = {};
            sppca_err = [];
            % solve
            for count = 1:10 %do 10 initializations and take the best b/c ends up in bad local minima a lot
                [SPPCAXunlab, SPPCAYunlab, Zsppca, Lsppca, B] = SPPCA_SS(Xunlab, Xlab,Ylab,k,exp(-10), randn(p,k), randn(q,k));
                Zsppcas{count} = Zsppca;
                Lsppcas{count} = Lsppca;
                SPPCAXunlabs{count} = SPPCAXunlab;
                SPPCAYunlabs{count} = SPPCAYunlab;
                SPPCAYlabs{count} = Zsppca*B;
                sppca_err(count) =  norm(SPPCAYlabs{count} - Ylab, 'fro')^2;
            end
            [~, loc] = min(sppca_err);
            Zsppca = Zsppcas{loc};
            Lsppca = orth(Lsppcas{loc}')';
            % Predict
            SPPCAXunlab = SPPCAXunlabs{loc};
            SPPCAYunlab = SPPCAYunlabs{loc};
            SPPCAYlab = SPPCAYlabs{loc};
            SPPCAtimes(l,t) = toc;
            % compute error
            mse = norm(SPPCAYunlab - Yunlab, 'fro')^2/ nunlab;
            SPPCArates(l,t) = mse ;
            SPPCArates_lab(l,t) = norm(SPPCAYlab - Ylab, 'fro')^2 / nlab;
            SPPCAvar(l,t) = norm(SPPCAXunlab, 'fro') / norm(Xunlab, 'fro');
            SPPCAvar_lab(l,t) = norm(Zsppca, 'fro') / norm(Xlab, 'fro');
        end
    else
        SPPCArates(l,t) = nan ;
        SPPCArates_lab(l,t) = nan;
        SPPCAvar(l,t) = nan;
        SPPCAvar_lab(l,t) = nan;
    end
    
    
    %% Barshan
    
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nlab, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nunlab, ~] = size(Yunlab);
        
        %learn basis
        [Zspca Lspca] = SPCA(Xlab', Ylab', k);
        spcaXunlab = Xunlab*Lspca';
        % predict
        betaSPCA = Zspca \Ylab;
        SPCAYunlab = spcaXunlab * betaSPCA;
        SPCAYlab = Zspca*betaSPCA;
        %compute error
        mse = norm(SPCAYunlab - Yunlab, 'fro')^2/ nunlab;
        SPCArates(l,t) = mse ;
        SPCArates_lab(l,t) = norm(SPCAYlab - Ylab, 'fro')^2 / nlab;
        SPCAvar(l,t) = norm(Xunlab*Lspca', 'fro') / norm(Xunlab, 'fro');
        SPCAvar_lab(l,t) = norm(Xlab*Lspca', 'fro') / norm(Xlab, 'fro');
    end
    
    %% Perform Barshan's KSPCA based 2D embedding
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nlab, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nunlab, ~] = size(Yunlab);
        
        for jj = 1:length(sigmas)
            sigma = sigmas(jj);
            % calc with best param on full labing set
            barshparam.ktype_y = 'linear';
            barshparam.kparam_y = 1;
            barshparam.ktype_x = 'rbf';
            barshparam.kparam_x = sigma;
            [Zkspca Lkspca] = KSPCA(Xlab', Ylab', k, barshparam);
            Zkspca = Zkspca';
            %do regression in learned basis
            betakSPCA = Zkspca \ Ylab;
            Kunlab = gaussian_kernel(Xunlab, Xlab, sigma);
            K = gaussian_kernel(Xlab, Xlab, sigma);
            kspcaXunlab = Kunlab*Lkspca;
            kSPCAYunlab = kspcaXunlab*betakSPCA;
            kSPCAYlab = Zkspca*betakSPCA;
            kspca_mbd_unlab{l,t,jj} = kspcaXunlab;
            kspca_mbd_lab{l,t,jj} = Zkspca;
            % compute error
            mse = norm(kSPCAYunlab - Yunlab, 'fro')^2 / nunlab;
            kSPCArates(l,t,jj) = mse;
            kSPCArates_lab(l,t,jj) = norm(kSPCAYlab - Ylab, 'fro')^2  / nlab;
            kSPCAvar(l,t,jj) = norm(kspcaXunlab, 'fro') / norm(Kunlab, 'fro');
            kSPCAvar_lab(l,t,jj) = norm(Zkspca, 'fro') / norm(K, 'fro');
        end
    end
    
    
%     %% Sup SVD
%     for l = 1:kfold
%         test_num = l
%         % get lth fold
%         Xlab = Xlabs{l};
%         Ylab = Ylabs{l};
%         [nlab, ~] = size(Ylab);
%         Xunlab = Xunlabs{l};
%         Yunlab = Yunlabs{l};
%         [nunlab, ~] = size(Yunlab);
%         % solve
%         [~,V,Zssvd,~,~]=SupPCA(Ylab,Xlab,k);
%         Lssvd = V';
%         % Predict
%         Bssvd = Zssvd \ Ylab;
%         SSVDXunlab = Xunlab*Lssvd';
%         SSVDYunlab = SSVDXunlab*Bssvd;
%         SSVDYlab = Zssvd*Bssvd;
%         %compute error
%         mse = norm(SSVDYunlab - Yunlab, 'fro')^2 / nunlab;
%         SSVDrates(l,t) = mse ;
%         SSVDrates_lab(l,t) = norm(SSVDYlab - Ylab, 'fro')^2 / nlab;
%         SSVDvar(l,t) = norm(SSVDXunlab, 'fro') / norm(Xunlab, 'fro');
%         SSVDvar_lab(l,t) = norm(Zssvd, 'fro') / norm(Xlab, 'fro');
%     end
    
    
end

%% save all data
save(strcat(dataset, '_results_ss'))

% %% compute avg performance for each k
% 
% %means
% avgPCA(t) = mean(PCArates);
% avgPCA_lab(t) = mean(PCArates_lab);
% avgkPCA(t, :) = mean(kPCArates);
% avgkPCA_lab(t, :) = mean(kPCArates_lab);
% avgPLS(t) = mean(PLSrates);
% avgPLS_lab(t) = mean(PLSrates_lab);
% avgLSPCA(t, :, :) = mean(LSPCArates, 1);
% avgLSPCA_lab(t, :, :) = mean(LSPCArates_lab, 1);
% avgkLSPCA(t, :, :, :) = mean(kLSPCArates, 1);
% avgkLSPCA_lab(t, :, :, :) = mean(kLSPCArates_lab, 1);
% avgSPCA(t) = mean(SPCArates);
% avgSPCA_lab(t) = mean(SPCArates_lab);
% avgkSPCA(t,:) = mean(kSPCArates);
% avgkSPCA_lab(t,:) = mean(kSPCArates_lab);
% avgISPCA(t) = mean(ISPCArates);
% avgISPCA_lab(t) = mean(ISPCArates_lab);
% avgSPPCA(t) = mean(SPPCArates);
% avgSPPCA_lab(t) = mean(SPPCArates_lab);
% avgR4(t, :) = mean(ridge_rrr_rates, 1);
% avgR4_lab(t, :) = mean(ridge_rrr_rates_lab, 1);
% % avgSSVD(t) = mean(SSVDrates);
% % avgSSVD_lab(t) = mean(SSVDrates_lab);
% 
% avgPCAvar(t) = mean(PCAvar);
% avgkPCAvar(t, :) = mean(kPCAvar);
% avgPLSvar(t) = mean(PLSvar);
% avgLSPCAvar(t, :, :) = mean(LSPCAvar, 1);
% avgkLSPCAvar(t, :, :, :) = mean(kLSPCAvar, 1);
% avgSPCAvar(t) = mean(SPCAvar);
% avgkSPCAvar(t, :) = mean(kSPCAvar);
% avgISPCAvar(t) = mean(ISPCAvar);
% avgSPPCAvar(t) = mean(SPPCAvar);
% avgR4var(t, :) = mean(ridge_rrrvars, 1);
% % avgSSVDvar(t) = mean(SSVDvar);
% 
% 
% avgPCAvar_lab(t) = mean(PCAvar_lab);
% avgkPCAvar_lab(t, :) = mean(kPCAvar_lab);
% avgPLSvar_lab(t) = mean(PLSvar_lab);
% avgLSPCAvar_lab(t, :, :) = mean(LSPCAvar_lab, 1);
% avgkLSPCAvar_lab(t, :, :, :) = mean(kLSPCAvar_lab, 1);
% avgSPCAvar_lab(t) = mean(SPCAvar_lab);
% avgkSPCAvar_lab(t, :) = mean(kSPCAvar_lab);
% avgISPCAvar_lab(t) = mean(ISPCAvar_lab);
% avgSPPCAvar_lab(t) = mean(SPPCAvar_lab);
% avgR4var_lab(t, :) = mean(ridge_rrrvars_lab, 1);
% % avgSSVDvar_lab(t) = mean(SSVDvar_lab);
% 
% %% print mean performance with std errors
% 
% 
% m = mean(PCArates);
% v = mean(PCAvar);
% sm = std(PCArates);
% sv = std(PCAvar);
% sprintf('PCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
% 
% [~, gamloc] = min(min(avgLSPCA,[], 3),[], 2);
% [~, lamloc] = min(min(avgLSPCA,[], 2),[], 3);
% m = mean(LSPCArates(:,:,gamloc,lamloc), 1);
% v = mean(LSPCAvar(:,:,gamloc,lamloc), 1);
% sm = std(LSPCArates(:,:,gamloc,lamloc), 1);
% sv = std(LSPCAvar(:,:,gamloc,lamloc), 1);
% sprintf('LSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
% 
% [~, gamloc] = min(min(min(avgkLSPCA,[], 4),[], 3), [], 2);
% [~, lamlock] = min(min(min(avgkLSPCA,[], 2),[], 4), [], 3);
% [~, siglock] = min(min(min(avgkLSPCA,[], 2),[], 3), [], 4);
% m = mean(kLSPCArates(:,:,gamloc,lamlock,siglock), 1);
% v = mean(kLSPCAvar(:,:,gamloc,lamlock,siglock), 1);
% sm = std(kLSPCArates(:,:,gamloc,lamlock,siglock), 1);
% sv = std(kLSPCAvar(:,:,gamloc,lamlock,siglock), 1);
% sprintf('kLSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
% 
% m = mean(ISPCArates);
% v = mean(ISPCAvar);
% sm = std(ISPCArates);
% sv = std(ISPCAvar);
% sprintf('ISPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
% 
% m = mean(SPPCArates);
% v = mean(SPPCAvar);
% sm = std(SPPCArates);
% sv = std(SPPCAvar);
% sprintf('SPPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
% 
% m = mean(SPCArates);
% v = mean(SPCAvar);
% sm = std(SPCArates);
% sv = std(SPCAvar);
% sprintf('Barshanerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
% 
% [~, locb] = min(avgkSPCA,[], 2);
% m = mean(kSPCArates(:,:,locb));
% v = mean(kSPCAvar(:,:,locb));
% sm = std(kSPCArates(:,:,locb));
% sv = std(kSPCAvar(:,:,locb));                                                                            
% sprintf('kBarshanerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
% 
% m = mean(SSVDrates);
% v = mean(SSVDvar);
% sm = std(SSVDrates);
% sv = std(SSVDvar);
% sprintf('SSVD: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
% 
% loc = 1; % RRR with parameter value 0
% m = mean(ridge_rrr_rates(:,loc), 1);
% v = mean(ridge_rrrvars(:,loc), 1);
% sm = std(ridge_rrr_rates(:,loc), 1);
% sv = std(ridge_rrrvars(:,loc), 1);
% sprintf('RRR: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
% 
% [~, locr4] = min(avgR4_lab);
% m = mean(ridge_rrr_rates(:,locr4), 1);
% v = mean(ridge_rrrvars(:,locr4), 1);
% sm = std(ridge_rrr_rates(:,locr4), 1);
% sv = std(ridge_rrrvars(:,locr4), 1);
% sprintf('R4: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)
% 
% 
% 
% %% plot error - var tradeoff curves
% 
% 
% for t = 1:length(ks)
%     figure()
%     hold on
%     plot(avgPCAvar_lab(t), avgPCA(t), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
%     %plot(avgkPCAvar_lab(t), avgkPCA(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgLSPCAvar_lab(t,:,lamloc), avgLSPCA(t, :,lamloc), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgkLSPCAvar_lab(t,:,lamlock,siglock), avgkLSPCA(t, :,lamlock,siglock), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgISPCAvar_lab(t), avgISPCA(t), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPPCAvar_lab(t), avgSPPCA(t), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPCAvar_lab(t), avgSPCA(t), '+', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgkSPCAvar_lab(t,locb), avgkSPCA(t), '>', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgR4var_lab(t,2:end), avgR4(t, 2:end), ':', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgR4var_lab(t,1), avgR4(t, 1), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgPLSvar_lab(t), avgPLS(t), 'h', 'MarkerSize', 20, 'LineWidth', 2)
% %     plot(avgSSVDvar_lab(t), avgSSVD(t), 'd', 'MarkerSize', 20, 'LineWidth', 2)
%     
%     xlabel('Variation Explained', 'fontsize', 25)
%     %title('unlab', 'fontsize', 25)
%     ylabel('MSE', 'fontsize', 25)
%     %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
%     set(gca, 'fontsize', 25)
%     lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'Location', 'best'); lgd.FontSize = 15;
%     %lgd = legend('LSPCA', 'R4', 'PLS', 'SPPCA', 'Barshan', 'SSVD', 'PCA', 'Location', 'southeast'); lgd.FontSize = 15;
%     %ylim([0, 0.12])
%     %set(gca, 'YScale', 'log')
%     xlim([0,1])
% end
% saveas(gcf, strcat(dataset, 'multi_obj_gamma_ss.jpg'))
% 
% 
% %% plot labing error - var tradeoff curves
% for t = 1:length(ks)
%     figure()
%     hold on
%     plot(avgPCAvar_lab(t), avgPCA_lab(t), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
%     %    plot(avgkPCAvar_lab(t), avgkPCA_lab(t), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgLSPCAvar_lab(t,:,lamloc), avgLSPCA_lab(t, :,lamloc), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgkLSPCAvar_lab(t,:,lamlock,siglock), avgkLSPCA_lab(t, :,lamlock,siglock), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgISPCAvar_lab(t), avgISPCA_lab(t), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPPCAvar_lab(t), avgSPPCA_lab(t), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPCAvar_lab(t), avgSPCA_lab(t), '+', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgkSPCAvar_lab(t), avgkSPCA_lab(t), '>', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgR4var_lab(t,2:end), avgR4_lab(t, 2:end), ':', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgR4var_lab(t,1), avgR4_lab(t, 1), 'k*', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgPLSvar_lab(t), avgPLS_lab(t), 'h', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSSVDvar_lab(t), avgSSVD_lab(t), 'd', 'MarkerSize', 20, 'LineWidth', 2)
%     
%     xlabel('Variation Explained', 'fontsize', 25)
%     %title('lab', 'fontsize', 25)
%     ylabel('MSE', 'fontsize', 25)
%     %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
%     set(gca, 'fontsize', 25)
%     lgd = legend('PCA', 'LSPCA', 'kLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', 'R4', 'RRR', 'PLS', 'SSVD', 'Location', 'best'); lgd.FontSize = 15;
%     %ylim([0, 0.12])
%     %set(gca, 'YScale', 'log')
%     xlim([0,1])
% end
% saveas(gcf, strcat(dataset, 'multi_obj_lab_gamma_ss.jpg'))
% 






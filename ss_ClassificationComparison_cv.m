
%% setup and load data
%rng(0);
ks = 2;
load(strcat(dataset, '.mat'));
[n, p] = size(X);
[~, q] = size(Y);


% create same splits to use every time and center
kfold = 20;
cvidx = crossvalind('kfold',Y,kfold,'classes',unique(Y),'min',3);
for l = 1:kfold
    Xlab = X(cvidx==l, :); %lth labled set
    Ylab = Y(cvidx==l, :);
    Xunlab = X(cvidx~=l, :); %lth unlabled set
    Yunlab = Y(cvidx~=l, :);
    [Xlab,Xunlab,Ylab,Yunlab] = ss_center_data(Xlab,Xunlab,Ylab,Yunlab,'classification');
    Xlabs{l} = Xlab; %lth centered labled set
    Ylabs{l} = Ylab;
    Xunlabs{l} = Xunlab; %lth centered unlabled set
    Yunlabs{l} = Yunlab;
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
        B = mnrfit(Xlab*Lpca',Ylab, 'Model', 'nominal', 'Interactions', 'on');
        [~, Yhat] = max(mnrval(B,pcaXunlab),[], 2);
        [~,pcaYlab] = max(mnrval(B,Zpca),[], 2);
        PCArates(l,t) = 1 - sum(Yhat == Yunlab) / nunlab;
        PCArates_lab(l,t) = 1 - sum(pcaYlab == Ylab) / nlab;
        PCAvar(l,t) = norm(Xunlab*Lpca'*Lpca, 'fro') / norm(Xunlab, 'fro');
        PCAvar_lab(l,t) = norm(Xlab*Lpca'*Lpca, 'fro') / norm(Xlab, 'fro');
    end
    
    %         %% kPCA
    %     for l = 1:kfold
    %         test_num = l
    %         % get lth fold
    %         Xlab = Xlabs{l};
    %         Ylab = Ylabs{l};
    %         Xunlab = Xunlabs{l};
    %         Yunlab = Yunlabs{l};
    %
    %         if l == 1
    %             for jj = 1:length(sigmas)
    %                 sigma = sigmas(jj)
    %                 K = gaussian_kernel(Xlabh, Xlabh, sigma);
    %                 [Lkpca, Zkpca] = pca(K, 'NumComponents', k);
    %                 Lkpca = Lkpca';
    %                 B = mnrfit(Zkpca,Ylabh, 'Model', 'nominal', 'Interactions', 'on');
    %                 [~, Yhat] = max(mnrval(B,gaussian_kernel(Xunlabh, Xlabh, sigma)*Lkpca'),[], 2);
    %                 PCAratesh(jj) = 1 - sum(Yhat == Yunlabh) / nHoldunlab;
    %             end
    %             sigloc = find(PCAratesh==min(PCAratesh,[],'all'),1,'last');
    %             bestSigma = sigmas(sigloc);
    %         end
    %
    %         K = gaussian_kernel(Xlab, Xlab, bestSigma);
    %         Kunlab = gaussian_kernel(Xunlab, Xlab, sigma);
    %         tic
    %         [Lkpca, Zkpca] = pca(K, 'NumComponents', k);
    %         kPCAtimes(l,t) = toc;
    %         Zkpcas{l,t} = Zkpca;
    %         Lkpca = Lkpca';
    %         % compute embedding for unlab data
    %         kpcaXunlab = Kunlab*Lkpca';
    %         kpcaXunlabs{l,t} = kpcaXunlab;
    %         % compute error
    %         B = mnrfit(Zkpca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
    %         [~, Yhat] = max(mnrval(B,kpcaXunlab),[], 2);
    %         [~,kpcaYlab] = max(mnrval(B,Zkpca),[], 2);
    %         kPCArates(l,t) = 1 - sum(Yhat == Yunlab) / nunlab
    %         kPCArates_lab(l,t) = 1 - sum(kpcaYlab == Ylab) / nlab;
    %         kPCAvar(l,t) = norm(kpcaXunlab, 'fro') / norm(Kunlab, 'fro');
    %         kPCAvar_lab(l,t) = norm(Zkpca, 'fro') / norm(K, 'fro');
    %     end
    
    
    
    %% LSPCA
    Gammas = [linspace(1, 0.7, 10), logspace(log10(0.7),log10(0.5058), 30)];
    for l = 1:kfold
        test_num = l
        r = zeros(length(Gammas), length(sigmas));
        rt = zeros(length(Gammas), length(sigmas));
        v = zeros(length(Gammas), length(sigmas));
        vt = zeros(length(Gammas), length(sigmas));
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nlab, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nunlab, ~] = size(Yunlab);
        
        Lstemp = {};
        lspca_mbd_unlabtemp = {};
        lspca_mbd_labtemp = {};
        r = []; rt = []; v = []; vt = [];
        %solve
        tic
        %        if n>=p %use ilrpca
        for ii = 1:length(Gammas)
            Gamma = Gammas(ii);
            if ii == 1
                [LSPCAXunlab, LSPCAYunlab, Zlspca, Llspca, B] = ilrpca_ss(Xunlab, Xlab, Ylab, Gamma, k, 0);
            else
                [LSPCAXunlab, LSPCAYunlab, Zlspca, Llspca, B] = ilrpca_ss(Xunlab, Xlab, Ylab, Gamma, k, Llspca');
            end
            Llspca = Llspca';
            Lstemp{ii} = Llspca;
            %predict
            LSPCAXlab = Zlspca;
            [~, LSPCAYlab] = max(Xlab*Llspca'*B(2:end,:) + B(1,:), [], 2);
            lspca_mbd_unlabtemp{ii} = LSPCAXunlab;
            lspca_mbd_labtemp{ii} = Zlspca;
            % compute error
            err = 1 - sum(Yunlab == LSPCAYunlab) / nunlab;
            lab_err = 1 - sum(Ylab == LSPCAYlab) / nlab;
            r(ii) = err;
            rt(ii) = lab_err;
            v(ii) = norm(LSPCAXunlab, 'fro') / norm(Xunlab, 'fro');
            vt(ii) = norm(Zlspca, 'fro') / norm(Xlab, 'fro');
        end
        %         else %use iklrpca with linear kernel (faster)
        %             for ii = 1:length(Gammas)
        %                 Gamma = Gammas(ii);
        %                 if ii == 1
        %                     [ Zlspca, Lorth, B, Klspca] = iklrpca_gamma(Xlab, Ylab, Gamma, 0, k, 0, 0);
        %                 else
        %                     [ Zlspca, Lorth, B, Klspca] = iklrpca_gamma(Xlab, Ylab, Gamma, 0, k, Lorth', Klspca);
        %                 end
        %                 Lorth = Lorth';
        %                 Lstemp{ii} = Lorth;
        %                 Kunlab = Xunlab*Xlab';
        %                 LSPCAXunlab = Kunlab*Lorth';
        %                 LSPCAXlab = Zlspca;
        %                 lspca_mbd_unlabtemp{ii} = LSPCAXunlab;
        %                 lspca_mbd_labtemp{ii} = Zlspca;
        %                 [~, LSPCAYunlab] = max(LSPCAXunlab*B(2:end,:) + B(1,:), [], 2);
        %                 [~, LSPCAYlab] = max(Zlspca*B(2:end,:) + B(1,:), [], 2);
        %                 err = 1 - sum(Yunlab == LSPCAYunlab)/nunlab;
        %                 lab_err = 1 - sum(Ylab == LSPCAYlab)/nlab;
        %                 r(ii) = err;
        %                 rt(ii) = lab_err;
        %                 v(ii) = norm(LSPCAXunlab, 'fro') / norm(Kunlab, 'fro');
        %                 vt(ii) = norm(Zlspca, 'fro') / norm(Klspca, 'fro');
        %             end
        %         end
        LSPCAminrate = min(r)
        Ls(l,t,:) = Lstemp;
        lspca_mbd_unlab(l,t,:) = lspca_mbd_unlabtemp;
        lspca_mbd_lab(l,t,:) = lspca_mbd_labtemp;
        LSPCArates(l,t,:) = r;
        LSPCArates_lab(l,t,:) = rt;
        LSPCAvar(l,t,:) = v;
        LSPCAvar_lab(l,t,:) = vt;
        avgLSPCAtimes(l,t) = toc/length(Gammas);
        avgLSPCAtimes(l,t)
    end
    
    %         %% ILSPCA
    %
    %         Lstemp = {};
    %         lspca_mbd_unlabtemp = {};
    %         lspca_mbd_labtemp = {};
    %         r = []; rt = []; v = []; vt = [];
    %         %Gammas = linspace(2,0.5, 51);
    %         %Gammas = 0.5;
    %         %Gammas = [linspace(2, 0.71, 20), logspace(log10(0.7), log10(0.6), 30), 0.99 - logspace(log10(0.4), log10(0.6), 30)];
    %         Gammas = [3, linspace(1, 0.71, 20), logspace(log10(0.7), log10(0.5), 30)];
    %         %solve
    %         tic
    %         if n>=p %use ilrpca
    %             for ii = 1:length(Gammas)
    %                 Gamma = Gammas(ii);
    %                 if ii == 1
    %                     [Zlspca, Llspca, B] = ilrpca_gamma(Xlab, Ylab, Gamma, k, 0);
    %                 else
    %                     [Zlspca, Llspca, B] = ilrpca_gamma(Xlab, Ylab, Gamma, k, Llspca');
    %                 end
    %                 Llspca = Llspca';
    %                 Lstemp{ii} = Llspca;
    %                 %predict
    %                 LSPCAXunlab = Xunlab*Llspca';
    %                 LSPCAXlab = Zlspca;
    %                 [~, LSPCAYunlab] = max(LSPCAXunlab*B(2:end,:) + B(1,:), [], 2);
    %                 [~, LSPCAYlab] = max(Xlab*Llspca'*B(2:end,:) + B(1,:), [], 2);
    %                 lspca_mbd_unlabtemp{ii} = LSPCAXunlab;
    %                 lspca_mbd_labtemp{ii} = Zlspca;
    %                 % compute error
    %                 err = 1 - sum(Yunlab == LSPCAYunlab) / nunlab;
    %                 lab_err = 1 - sum(Ylab == LSPCAYlab) / nlab;
    %                 r(ii) = err;
    %                 rt(ii) = lab_err;
    %                 v(ii) = norm(LSPCAXunlab, 'fro') / norm(Xunlab, 'fro');
    %                 vt(ii) = norm(Zlspca, 'fro') / norm(Xlab, 'fro');
    %             end
    %         else %use iklrpca with linear kernel (faster)
    %             for ii = 1:length(Gammas)
    %                 Gamma = Gammas(ii);
    %                 if ii == 1
    %                     [ Zlspca, Lorth, B, Klspca] = iklrpca_gamma(Xlab, Ylab, Gamma, 0, k, 0, 0);
    %                 else
    %                     [ Zlspca, Lorth, B, Klspca] = iklrpca_gamma(Xlab, Ylab, Gamma, 0, k, Lorth', Klspca);
    %                 end
    %                 Lorth = Lorth';
    %                 Lstemp{ii} = Lorth;
    %                 Kunlab = Xunlab*Xlab';
    %                 LSPCAXunlab = Kunlab*Lorth';
    %                 LSPCAXlab = Zlspca;
    %                 lspca_mbd_unlabtemp{ii} = LSPCAXunlab;
    %                 lspca_mbd_labtemp{ii} = Zlspca;
    %                 [~, LSPCAYunlab] = max(LSPCAXunlab*B(2:end,:) + B(1,:), [], 2);
    %                 [~, LSPCAYlab] = max(Zlspca*B(2:end,:) + B(1,:), [], 2);
    %                 err = 1 - sum(Yunlab == LSPCAYunlab)/nunlab;
    %                 lab_err = 1 - sum(Ylab == LSPCAYlab)/nlab;
    %                 r(ii) = err;
    %                 rt(ii) = lab_err;
    %                 v(ii) = norm(LSPCAXunlab, 'fro') / norm(Kunlab, 'fro');
    %                 vt(ii) = norm(Zlspca, 'fro') / norm(Klspca, 'fro');
    %             end
    %         end
    %         ILSPCAminrate = min(r)
    %         ILs(l,t,:) = Lstemp;
    %         Ilspca_mbd_unlab(l,t,:) = lspca_mbd_unlabtemp;
    %         Ilspca_mbd_lab(l,t,:) = lspca_mbd_labtemp;
    %         ILSPCArates(l,t,:) = r;
    %         ILSPCArates_lab(l,t,:) = rt;
    %         ILSPCAvar(l,t,:) = v;
    %         ILSPCAvar_lab(l,t,:) = vt;
    %         avgILSPCAtimes(l,t) = toc/length(Gammas);
    %         avgILSPCAtimes(l,t)
    %
    
    
    %         %% Gamma Lambda LSPCA
    %         Lambdas = linspace(0.1, 5, 20);
    %         Lstemp = {};
    %         lspca_mbd_unlabtemp = {};
    %         lspca_mbd_labtemp = {};
    %         r = []; rt = []; v = []; vt = [];
    %         Gammas = [3, linspace(1, 0.71, 20), logspace(log10(0.7), log10(0.5), 30)];
    %
    %         kLSPCAratesh = []; kLSPCAvarh = []; kLSPCAvar_labh = [];
    %         if l == 1
    %             for jj = 1:length(Lambdas)
    %                 Lambda = Lambdas(jj)
    %                 for ii = 1:length(Gammas)
    %                     Gamma = Gammas(ii);
    %                     if ii == 1
    %                         [Zlspca, Llspca, B] = lrpca_gamma_lambda(Xlabh, Ylabh, Gamma, Lambda, k, 0);
    %                     else
    %                         [Zlspca, Llspca, B] = lrpca_gamma_lambda(Xlabh, Ylabh, Gamma, Lambda, k, Llspca');
    %                     end
    %                     Llspca = Llspca';
    %                     %predict
    %                     LSPCAXunlab = Xunlabh*Llspca';
    %                     LSPCAXlab = Zlspca;
    %                     [~, LSPCAYunlab] = max(LSPCAXunlab*B(2:end,:) + B(1,:), [], 2);
    %                     err = 1 - sum(Yunlabh == LSPCAYunlab)/nHoldunlab;
    %                     LSPCAratesh(jj,ii) = err;
    %                 end
    %             end
    %             [lamloc, gamloc] = ind2sub( size(LSPCAratesh), find(LSPCAratesh==min(LSPCAratesh,[],'all'),1,'last') );
    %             bestLambda = Lambdas(lamloc)
    %         end
    %
    %         %solve
    %         tic
    %         for ii = 1:length(Gammas)
    %             Gamma = Gammas(ii);
    %             if ii == 1
    %                 [Zlspca, Llspca, B] = lrpca_gamma_lambda(Xlab, Ylab, Gamma, bestLambda, k, 0);
    %             else
    %                 [Zlspca, Llspca, B] = lrpca_gamma_lambda(Xlab, Ylab, Gamma, bestLambda, k, Llspca');
    %             end
    %             Llspca = Llspca';
    %             Lstemp{ii} = Llspca;
    %             %predict
    %             LSPCAXunlab = Xunlab*Llspca';
    %             LSPCAXlab = Zlspca;
    %             [~, LSPCAYunlab] = max(LSPCAXunlab*B(2:end,:) + B(1,:), [], 2);
    %             [~, LSPCAYlab] = max(Xlab*Llspca'*B(2:end,:) + B(1,:), [], 2);
    %             lspca_mbd_unlabtemp{ii} = LSPCAXunlab;
    %             lspca_mbd_labtemp{ii} = Zlspca;
    %             % compute error
    %             err = 1 - sum(Yunlab == LSPCAYunlab) / nunlab;
    %             lab_err = 1 - sum(Ylab == LSPCAYlab) / nlab;
    %             r(ii) = err;
    %             rt(ii) = lab_err;
    %             v(ii) = norm(LSPCAXunlab, 'fro') / norm(Xunlab, 'fro');
    %             vt(ii) = norm(Zlspca, 'fro') / norm(Xlab, 'fro');
    %         end
    %         lambda_LSPCAminrate = min(r)
    %         lambda_Ls(l,t,:) = Lstemp;
    %         lambda_lspca_mbd_unlab(l,t,:) = lspca_mbd_unlabtemp;
    %         lambda_lspca_mbd_lab(l,t,:) = lspca_mbd_labtemp;
    %         lambda_LSPCArates(l,t,:) = r;
    %         lambda_LSPCArates_lab(l,t,:) = rt;
    %         lambda_LSPCAvar(l,t,:) = v;
    %         lambda_LSPCAvar_lab(l,t,:) = vt;
    %         lambda_avgLSPCAtimes(l,t) = toc/length(Gammas);
    %         lambda_avgLSPCAtimes(l,t)
    
    %% kLSPCA
    Gammas = [linspace(1, 0.7, 10), logspace(log10(0.7),log10(0.5), 30)];
    for l = 1:kfold
        test_num = l
        r = zeros(length(Gammas), length(sigmas));
        rt = zeros(length(Gammas), length(sigmas));
        v = zeros(length(Gammas), length(sigmas));
        vt = zeros(length(Gammas), length(sigmas));
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nlab, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nunlab, ~] = size(Yunlab);
        tic
        for jj = 1:length(sigmas)
            sigma = sigmas(jj)
            for ii = 1:length(Gammas)
                Gamma = Gammas(ii);
                if ii == 1
                    [kLSPCAXunlab, kLSPCAYunlab, Zklspca, Lorth, B, Klspca] = iklrpca_ss(Xunlab, Xlab, Ylab, Gamma, sigma, k, 0, 0);
                else
                    [kLSPCAXunlab, kLSPCAYunlab, Zklspca, Lorth, B, Klspca] = iklrpca_ss(Xunlab, Xlab, Ylab, Gamma, sigma, k, Lorth', Klspca);
                end
                Lorth = Lorth';
                kLstemp{ii,jj} = Lorth;
                kLSPCAXlab = Zklspca;
                klspca_mbd_unlabtemp{ii,jj} = kLSPCAXunlab;
                klspca_mbd_labtemp{ii,jj} = Zklspca;
                [~, kLSPCAYlab] = max(Zklspca*B(2:end,:) + B(1,:), [], 2);
                err = 1 - sum(Yunlab == kLSPCAYunlab)/nunlab;
                lab_err = 1 - sum(Ylab == kLSPCAYlab)/nlab;
                r(ii,jj) = err;
                rt(ii,jj) = lab_err;
                Klab = Klspca(1:nlab,:);
                Kunlab = Klspca(nlab+1:end,:);
                v(ii,jj) = norm(kLSPCAXunlab, 'fro') / norm(Kunlab, 'fro');
                vt(ii,jj) = norm(Zklspca, 'fro') / norm(Klab, 'fro');
            end
        end
        
        kLSPCAminrate = min(r)
        kLs(l,t,:,:) = kLstemp;
        klspca_mbd_unlab(l,t,:,:) = klspca_mbd_unlabtemp;
        klspca_mbd_lab(l,t,:,:) = klspca_mbd_labtemp;
        kLSPCArates(l,t,:,:) = r;
        kLSPCArates_lab(l,t,:,:) = rt;
        kLSPCAvar(l,t,:,:) = v;
        kLSPCAvar_lab(l,t,:,:) = vt;
        avgkLSPCAtimes(l,t) = toc/(length(Gammas)*length(sigmas));
        avgkLSPCAtimes(l,t)
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
            K = gaussian_kernel(Xlab, Xlab, sigma);
            Kunlab = gaussian_kernel(Xunlab, Xlab, sigma);
            tic
            [Lkpca, Zkpca] = pca(K, 'NumComponents', k);
            kPCAtimes(l,t,jj) = toc;
            Zkpcas{l,t,jj} = Zkpca;
            Lkpca = Lkpca';
            % compute embedding for unlab data
            kpcaXunlab = Kunlab*Lkpca';
            kpcaXunlabs{l,t,jj} = kpcaXunlab;
            % compute error
            B = mnrfit(Zkpca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
            [~, Yhat] = max(mnrval(B,kpcaXunlab),[], 2);
            [~,kpcaYlab] = max(mnrval(B,Zkpca),[], 2);
            kPCArates(l,t,jj) = 1 - sum(Yhat == Yunlab) / nunlab;
            kPCArates_lab(l,t,jj) = 1 - sum(kpcaYlab == Ylab) / nlab;
            kPCAvar(l,t,jj) = norm(kpcaXunlab, 'fro') / norm(Kunlab, 'fro');
            kPCAvar_lab(l,t,jj) = norm(Zkpca, 'fro') / norm(K, 'fro');
        end
    end
    
    %% ISPCA
    %find basis
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
        [Zispca, Lispca, B] = ISPCA(Xlab,Ylab,k);
        Zispcas{l,t} = Zispca;
        ISPCAtimes(l,t) = toc;
        % predict
        ISPCAXunlab = Xunlab*Lispca';
        ISPCAXunlabs{l,t} = ISPCAXunlab;
        B = mnrfit(Xlab*Lispca',Ylab, 'Model', 'nominal', 'Interactions', 'on');
        [~,Yhat] = max(mnrval(B,ISPCAXunlab),[], 2);
        [~, ISPCAYlab] = max(mnrval(B,Zispca),[], 2);
        % compute error
        ISPCArates(l,t) = 1 - sum(Yhat == Yunlab) / nunlab;
        ISPCArates(l,t)
        ISPCArates_lab(l,t) = 1 - sum(ISPCAYlab == Ylab) / nlab;
        ISPCAvar(l,t) = norm(Xunlab*Lispca', 'fro') / norm(Xunlab, 'fro');
        ISPCAvar_lab(l,t) = norm(Xlab*Lispca', 'fro') / norm(Xlab, 'fro');
        ISPCAtimes(l,t) = toc;
    end
    
    %% SPPCA
    % solve
    if ~strcmp(dataset, 'Arcene')
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
            Zsppcasin = {};
            Lsppcasin = {};
            SPPCAXunlabin = {};
            SPPCAYunlabin = {};
            SPPCAYlabin = {};
            sppca_err = [];
            for count = 1%:10 %do 10 initializations and take the best b/c ends up in bad local minima a lot
                [SPPCAXunlab, SPPCAYunlab, Zsppca, Lsppca, ~] = SPPCA_SS(Xunlab, Xlab,Ylab,k,exp(-10), randn(p,k), randn(q,k));
                Zsppcasin{count} = Zsppca;
                Lsppcasin{count} = Lsppca;
                SPPCAXunlabin{count} = SPPCAXunlab;
                B = mnrfit(Zsppca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
                SPPCAYunlabin{count} = SPPCAYunlab;
                SPPCAYlabin{count} = max(mnrval(B,Zsppca),[], 2);
                sppca_err(count) =  norm(SPPCAYlabin{count} - Ylab, 'fro');
            end
            [~, loc] = min(sppca_err);
            Zsppca = Zsppcasin{loc};
            Zsppcas{l,t} = Zsppca;
            Lsppca = orth(Lsppcasin{loc}')';
            % Predict
            SPPCAXunlab = SPPCAXunlabin{loc};
            SPPCAXunlabs{l,t} = SPPCAXunlab;
            SPPCAYunlab = SPPCAYunlabin{loc};
            SPPCAYlab = SPPCAYlabin{loc};
            % compute error
            B = mnrfit(Zsppca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
            [~,SPPCAYlab] = max(mnrval(B,Zsppca),[], 2);
            [~,Yhat] = max(mnrval(B,SPPCAXunlab),[], 2);
            SPPCArates(l,t) = 1 - sum(Yhat == Yunlab) / nunlab;
            SPPCArates(l,t)
            SPPCArates_lab(l,t) = 1 - sum(SPPCAYlab == Ylab) / nlab;
            Lsppca_orth = orth(Lsppca'); %normalize latent directions for variation explained comparison
            Lsppca_orth = Lsppca_orth';
            SPPCAvar(l,t) = norm(Xunlab*Lsppca_orth', 'fro') / norm(Xunlab, 'fro');
            SPPCAvar_lab(l,t) = norm(Xlab*Lsppca_orth', 'fro') / norm(Xlab, 'fro');
            
            SPPCAtimes(l,t) = toc;
        end
    else
        SPPCArates(l,t) = nan;
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
        tic
        barshparam = struct;
        if n>p
            %learn basis
            [Zspca Lspca] = SPCA(Xlab', Ylab', k);
            spcaXunlab = Xunlab*Lspca';
            % predict
            B = mnrfit(Zspca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
            [~,Yhat] = max(mnrval(B,spcaXunlab),[], 2);
            %compute error
            SPCArates(l,t) = 1 - sum(Yhat == Yunlab) / nunlab;
            SPCArates(l,t)
            [~,SPCAYlab] = max(mnrval(B,Zspca),[], 2);
            SPCArates_lab(l,t) = 1 - sum(SPCAYlab == Ylab) / nlab;
            SPCAvar(l,t) = norm(Xunlab*Lspca', 'fro') / norm(Xunlab, 'fro');
            SPCAvar_lab(l,t) = norm(Xlab*Lspca', 'fro') / norm(Xlab, 'fro');
        else
            % kernel version faster in this regime
            barshparam.ktype_y = 'delta';
            barshparam.kparam_y = 1;
            barshparam.ktype_x = 'linear';
            barshparam.kparam_x = 1;
            [Zspca Lspca] = KSPCA(Xlab', Ylab', k, barshparam);
            Zspca = Zspca';
            %do prediction in learned basis
            betaSPCA = Zspca \ Ylab;
            Klab = Xlab*Xlab';
            Kunlab = Xunlab*Xlab';
            spcaXunlab = Kunlab*Lspca;
            spca_mbd_unlab{l,t} = spcaXunlab;
            spca_mbd_lab{l,t} = Zspca;
            B = mnrfit(Zspca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
            [~,Yhat] = max(mnrval(B,spcaXunlab),[], 2);
            [~,SPCAYlab] = max(mnrval(B,Zspca),[], 2);
            % compute error
            SPCArates(l,t) = 1 - sum(Yhat == Yunlab) / nunlab;
            SPCArates(l,t)
            SPCArates_lab(l,t) = 1 - sum(SPCAYlab == Ylab) / nlab;
            SPCAvar(l,t) = norm(spcaXunlab, 'fro') / norm(Kunlab, 'fro');
            SPCAvar_lab(l,t) = norm(Zspca, 'fro') / norm(Klab, 'fro');
        end
        spcaXunlabs{l,t} = spcaXunlab;
        Zspcas{l,t} = Zspca;
        Barshantimes(l,t) = toc;
    end
    
    %% Perform Barshan's KSPCA based 2D embedding
    %learn basis
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
            Klab = gaussian_kernel(Xlab, Xlab, sigma);
            Kunlab = gaussian_kernel(Xunlab, Xlab, sigma);
            kspcaXunlab = Kunlab*Lkspca;
            kspca_mbd_unlab{l,t,jj} = kspcaXunlab;
            kspca_mbd_lab{l,t,jj} = Zkspca;
            B = mnrfit(Zkspca,Ylab, 'Model', 'nominal', 'Interactions', 'on');
            [~,Yhat] = max(mnrval(B,kspcaXunlab),[], 2);
            [~,kSPCAYlab] = max(mnrval(B,Zkspca),[], 2);
            %compute error
            kSPCArates(l,t,jj) = 1 - sum(Yhat == Yunlab) / nunlab;
            kSPCArates(l,t,jj)
            kSPCArates_lab(l,t,jj) = 1 - sum(kSPCAYlab == Ylab) / nlab;
            kSPCAvar(l,t,jj) = norm(kspcaXunlab, 'fro') / norm(Kunlab, 'fro');
            kSPCAvar_lab(l,t,jj) = norm(Zkspca, 'fro') / norm(Klab, 'fro');
            kBarshantimes(l,t,jj) = toc;
            kspcaXunlabs{l,t,jj} = kspcaXunlab;
            Zkspcas{l,t,jj} = Zkspca;
        end
    end
    
    %% LDA
    % solve
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nlab, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nunlab, ~] = size(Yunlab);
        Mdl = fitcdiscr(Xlab,Ylab, 'DiscrimType', 'pseudolinear');
        LDAYunlab = predict(Mdl,Xunlab);
        % Predict
        LDAYlab = predict(Mdl,Xlab);
        %compute error
        LDArates(l,t) = 1 - sum(LDAYunlab == Yunlab) / nunlab;
        LDArates(l,t)
        LDArates_lab(l,t) = 1 - sum(LDAYlab == Ylab) / nlab;
        lin = Mdl.Coeffs(1,2).Linear / norm([Mdl.Coeffs(1,2).Const; Mdl.Coeffs(1,2).Linear]);
        const = Mdl.Coeffs(1,2).Const / norm([Mdl.Coeffs(1,2).Const; Mdl.Coeffs(1,2).Linear]);
        Zlda = Xlab*lin + const;
        LDAXunlab = Xunlab*lin + const;
        LDAvar(l,t) = norm(LDAXunlab, 'fro') / norm(Xunlab, 'fro');
        LDAvar_lab(l,t) = norm(Zlda, 'fro') / norm(Xlab, 'fro');
    end
    
    
    %% Supervised Discriminant Analysis
    % solve
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nl, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nu, ~] = size(Yunlab);
        
        options = struct; %use defaults
        [Lsda, ~] = SDA([Ylab; zeros(nu,1)],X,~~[ones(nl,1);zeros(nu,1)], options);
        Zself = X*Lsda;
        Zl = Zself(1:nl,:);
        Zu = Zself(nl+1:end,:);
        B = mnrfit(Zl,Ylab, 'Model', 'nominal', 'Interactions', 'on');
        [~,Yhat] = max(mnrval(B,Zu),[], 2);
        [~, SDAYl] = max(mnrval(B,Zl),[], 2);
        % compute error
        SDArates(l) = 1 - sum(Yhat == Yunlab) / nu
        SDArates_lab(l) = 1 - sum(SDAYl == Ylab) / nl;
        SDAvar(l) = norm(Zu, 'fro') / norm(Xunlab, 'fro');
        SDAvar_lab(l) = norm(Zl, 'fro') / norm(Xlab, 'fro');
    end
    
    
    
    %         %% QDA
    %         % solve
    %         QMdl = fitcdiscr(Xlab,Ylab, 'DiscrimType', 'pseudoquadratic');
    %         QDAYunlab = predict(QMdl,Xunlab);
    %         % Predict
    %         QDAYlab = predict(QMdl,Xlab);
    %         %compute error
    %         QDArates(l,t) = 1 - sum(QDAYunlab == Yunlab) / nunlab;
    %         QDArates(l,t)
    %         QDArates_lab(l,t) = 1 - sum(QDAYlab == Ylab) / nlab;
    
    %% Local Fisher Discriminant Analysis (LFDA)
    if ~strcmp(dataset, 'colon')
        for l = 1:kfold
            test_num = l
            % get lth fold
            Xlab = Xlabs{l};
            Ylab = Ylabs{l};
            [nlab, ~] = size(Ylab);
            Xunlab = Xunlabs{l};
            Yunlab = Yunlabs{l};
            [nunlab, ~] = size(Yunlab);
            K = Xlab*Xlab';
            Kunlab = Xunlab*Xlab';
            %if ~strcmp(dataset, 'colon') && ~strcmp(dataset, 'Arcene')
            %lab
            %[Llfda,Zlfda] = LFDA(Xlab',Ylab,k, 'plain');
            [Llfda,~] = KLFDA(K,Ylab,k, 'plain', 1);
            %predict
            Llfda = orth(Llfda);
            Zlfda = K*Llfda;
            B = mnrfit(Zlfda,Ylab, 'Model', 'nominal', 'Interactions', 'on');
            %LFDAXunlab = Xunlab*Llfda;
            LFDAXunlab = Kunlab*Llfda;
            [~,Yhat] = max(mnrval(B,LFDAXunlab),[], 2);
            [~,LFDAYlab] = max(mnrval(B,Zlfda),[], 2);
            %compute error
            LFDArates(l,t) = 1 - sum(Yhat == Yunlab) / nunlab;
            LFDArates(l,t)
            LFDArates_lab(l,t) = 1 - sum(LFDAYlab == Ylab) / nlab;
            LFDAvar(l,t) = norm(LFDAXunlab, 'fro') / norm(Kunlab, 'fro');
            LFDAvar_lab(l,t) = norm(Zlfda, 'fro') / norm(K, 'fro');
        end
    else
        LFDArates(l,t) = nan;
        LFDArates_lab(l,t) = nan;
        LFDAvar(l,t) = nan;
        LFDAvar_lab(l,t) = nan;
    end
    
    %% Kernel Local Fisher Discriminant Analysis (KLFDA)
    %choose kernel param
    if ~strcmp(dataset, 'colon')
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
                %lab
                K = gaussian_kernel(Xlab, Xlab, sigma);
                [Llfda,~] = KLFDA(K,Ylab,k, 'plain', 1);
                Llfda = orth(Llfda);
                Zlfda = K*Llfda;
                %predict
                Kunlab = gaussian_kernel(Xunlab, Xlab, sigma);
                LFDAXunlab = Kunlab*Llfda;
                B = mnrfit(Zlfda,Ylab, 'Model', 'nominal', 'Interactions', 'on');
                [~,Yhat] = max(mnrval(B,LFDAXunlab),[], 2);
                [~,LFDAYlab] = max(mnrval(B,Zlfda),[], 2);
                %compute error
                kLFDArates(l,t,jj) = 1 - sum(Yhat == Yunlab) / nunlab;
                kLFDArates(l,t,jj)
                kLFDArates_lab(l,t,jj) = 1 - sum(LFDAYlab == Ylab) / nlab;
                kLFDAvar(l,t,jj) = norm(LFDAXunlab, 'fro') / norm(Kunlab, 'fro');
                kLFDAvar_lab(l,t,jj) = norm(Zlfda, 'fro') / norm(K, 'fro');
            end
        end
    else
        kLFDArates(l,t) = nan;
        kLFDArates_lab(l,t) = nan;
        kLFDAvar(l,t) = nan;
        kLFDAvar_lab(l,t) = nan;
    end
    
        %% Semi Supervised Local Fisher Discriminant Analysis (SELF)
    for l = 1:kfold
        test_num = l
        % get lth fold
        Xlab = Xlabs{l};
        Ylab = Ylabs{l};
        [nlab, ~] = size(Ylab);
        Xunlab = Xunlabs{l};
        Yunlab = Yunlabs{l};
        [nunlab, ~] = size(Yunlab);
        
        [Lself,Zself]=SELF(X',[Ylab; zeros(nunlab,1)],0.5,k);
        Zself = Zself';
        Zl = Zself(1:nlab,:);
        Zu = Zself(nlab+1:end,:);
        B = mnrfit(Zl,Ylab, 'Model', 'nominal', 'Interactions', 'on');
        [~,Yhat] = max(mnrval(B,Zu),[], 2);
        [~, SELFYl] = max(mnrval(B,Zl),[], 2);
        % compute error
        SELFrates(l) = 1 - sum(Yhat == Yunlab) / nu;
        SELFrates(l)
        SELFrates_lab(l) = 1 - sum(SELFYl == Ylab) / nl;
        SELFvar(l) = norm(Zu, 'fro') / norm(Xunlab, 'fro');
        SELFvar_lab(l) = norm(Zl, 'fro') / norm(Xlab, 'fro');
        
        SELFtimes(l) = toc;
    end
    
%     %% FME
%     for l = 1:kfold
%         test_num = l
%         % get lth fold
%         Xlab = Xlabs{l};
%         Ylab = Ylabs{l};
%         [nlab, ~] = size(Ylab);
%         Xunlab = Xunlabs{l};
%         Yunlab = Yunlabs{l};
%         [nunlab, ~] = size(Yunlab);
%         
%         Xh = [Xlh; Xuh];
%         adj = max(Xh*Xh', 0.5).*(1-eye(nHoldl + nHoldu));
%         Lapl = diag(sum(adj,1)) - adj;
%         Ymask = zeros(nHoldl,length(unique(Yl))); Ymask(sub2ind(size(Ymask), (1:nHoldl)', Ylh)) = 1;
%         T = [Ymask; zeros(nHoldu, length(unique(Yl)))];
%         para.ul = 10000;
%         para.uu = 0;
%         mus = logspace(-9,9,7);
%         lambdas = mus;
%         bestrate = inf;
%         for mu = mus
%             for lambda = lambdas
%                 para.mu = mu;
%                 para.lamda = lambda;
%                 [W, b, F] = FME_semi(Xh', Lapl, T, para);
%                 [~, FMEYu] = max(F(nHoldl+1:end,:),[], 2);
%                 rate = 1 - sum(FMEYu == Yuh) / nu;
%                 if rate < bestrate
%                    bestrate=rate;
%                    bestmu = mu;
%                    bestlambda=lambda;
%                 end
%             end
%         end
%         adj = max(X*X', 0.5).*(1-eye(nl+nu));
%         Lapl = diag(sum(adj,1)) - adj;
%         Ymask = zeros(nl,length(unique(Yl))); Ymask(sub2ind(size(Ymask), (1:nl)', Yl)) = 1;
%         T = [Ymask; zeros(nu, length(unique(Yl)))];
%         para.mu = bestmu;
%         para.lamda = bestlambda;
%         [W, b, F] = FME_semi(X', Lapl, T, para);
%         Zl = Xl*W;
%         Zu = Xu*W;
%         [~, FMEYl] = max(F(1:nl,:),[], 2);
%         [~, FMEYu] = max(F(nl+1:end,:),[], 2);
%         rate = 1 - sum(FMEYu == Yu) / nu;
%         FMErates(l) = 1 - sum(FMEYu == Yu) / nu
%         FMErates_l(l) = 1 - sum(FMEYl == Yl) / nl;
%         FMEvar(l) = norm(Zu, 'fro') / norm(Xu, 'fro');
%         FMEvar_l(l) = norm(Zl, 'fro') / norm(Xl, 'fro');
%     end
    
    
    
    
end
%% save all data
save(strcat(dataset, '_results_ss'))

%% compute avg performance accross folds

avgPCA(t) = mean(PCArates);
avgPCA_lab(t) = mean(PCArates_lab);
avgkPCA(t,:) = mean(kPCArates);
avgkPCA_lab(t,:) = mean(kPCArates_lab);
avgLSPCA(t, :) = mean(LSPCArates);
avgLSPCA_lab(t, :) = mean(LSPCArates_lab);
%     lambda_avgLSPCA(t, :) = mean(lambda_LSPCArates, 1);
%     lambda_avgLSPCA_lab(t, :) = mean(lambda_LSPCArates_lab, 1);
avgkLSPCA(t, :, :) = mean(kLSPCArates);
avgkLSPCA_lab(t, :, :) = mean(kLSPCArates_lab);
%     avgILSPCA(t, :) = mean(ILSPCArates);
%     avgILSPCA_lab(t, :) = mean(ILSPCArates_lab);
avgSPCA(t) = mean(SPCArates);
avgSPCA_lab(t) = mean(SPCArates_lab);
avgkSPCA(t,:) = mean(kSPCArates);
avgkSPCA_lab(t,:) = mean(kSPCArates_lab);
avgISPCA(t) = mean(ISPCArates);
avgISPCA_lab(t) = mean(ISPCArates_lab);
avgSPPCA(t) = mean(SPPCArates);
avgSPPCA_lab(t) = mean(SPPCArates_lab);
avgLDA(t) = mean(LDArates);
avgLDA_lab(t) = mean(LDArates_lab);
%     avgQDA(t) = mean(QDArates);
%     avgQDA_lab(t) = mean(QDArates_lab);
avgLFDA(t) = mean(LFDArates);
avgLFDA_lab(t) = mean(LFDArates_lab);
avgkLFDA(t,:) = mean(kLFDArates);
avgkLFDA_lab(t,:) = mean(kLFDArates_lab);
avgSELF(t) = mean(SELFrates);
avgSELF_lab(t) = mean(SELFrates_lab);
avgSDA(t,:) = mean(SDArates);
avgSDA_lab(t,:) = mean(SDArates_lab);


%
avgPCAvar(t) = mean(PCAvar);
avgkPCAvar(t, :) = mean(kPCAvar);
avgLSPCAvar(t, :) = mean(LSPCAvar);
%     lambda_avgLSPCAvar(t, :) = mean(lambda_LSPCAvar);
avgkLSPCAvar(t, :, :) = mean(kLSPCAvar);
%     avgILSPCAvar(t, :) = mean(ILSPCAvar);
avgSPCAvar(t) = mean(SPCAvar);
avgkSPCAvar(t,:) = mean(kSPCAvar);
avgISPCAvar(t) = mean(ISPCAvar);
avgSPPCAvar(t) = mean(SPPCAvar);
avgLDAvar(t) = mean(LDAvar);
avgLFDAvar(t) = mean(LFDAvar);
avgkLFDAvar(t,:) = mean(kLFDAvar);
avgSELFvar(t) = mean(SELFvar);
avgSDAvar(t,:) = mean(SDAvar);

avgPCAvar_lab(t) = mean(PCAvar_lab);
avgkPCAvar_lab(t,:) = mean(kPCAvar_lab);
avgLSPCAvar_lab(t, :) = mean(LSPCAvar_lab);
%     lambda_avgLSPCAvar_lab(t, :) = mean(lambda_LSPCAvar_lab);
avgkLSPCAvar_lab(t, :, :) = mean(kLSPCAvar_lab);
%     avgILSPCAvar_lab(t, :) = mean(ILSPCAvar_lab);
avgSPCAvar_lab(t) = mean(SPCAvar_lab);
avgkSPCAvar_lab(t,:) = mean(kSPCAvar_lab);
avgISPCAvar_lab(t) = mean(ISPCAvar_lab);
avgSPPCAvar_lab(t) = mean(SPPCAvar_lab);
avgLDAvar_lab(t) = mean(LDAvar_lab);
avgLFDAvar_lab(t) = mean(LFDAvar_lab);
avgkLFDAvar_lab(t, :) = mean(kLFDAvar_lab);
avgSELFvar_lab(t) = mean(SELFvar_lab);
avgSDAvar_lab(t, :) = mean(SDAvar_lab);

%% print mean performance with std errors


m = mean(PCArates);
v = mean(PCAvar);
sm = std(PCArates);
sv = std(PCAvar);
sprintf('PCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

[~, gamloc] = min(avgLSPCA,[], 2);
m = mean(LSPCArates(:,:,gamloc), 1);
v = mean(LSPCAvar(:,:,gamloc), 1);
sm = std(LSPCArates(:,:,gamloc), 1);
sv = std(LSPCAvar(:,:,gamloc), 1);
sprintf('LSPCAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

[~, gamlock] = min(min(avgkLSPCA,[], 3), [], 2);
[~, siglock] = min(min(avgkLSPCA,[], 2), [], 3);
m = mean(kLSPCArates(:,:,gamlock,siglock), 1);
v = mean(kLSPCAvar(:,:,gamlock,siglock), 1);
sm = std(kLSPCArates(:,:,gamlock,siglock), 1);
sv = std(kLSPCAvar(:,:,gamlock,siglock), 1);
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

m = mean(LDArates);
v = mean(LDAvar);
sm = std(LDArates);
sv = std(LDAvar);
sprintf('LDAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(LFDArates);
v = mean(LFDAvar);
sm = std(LFDArates);
sv = std(LFDAvar);
sprintf('LFDAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

[~, locl] = min(avgkLFDA,[], 2);
m = mean(kLFDArates(:,:,locl));
v = mean(kLFDAvar(:,:,locl));
sm = std(kLFDArates(:,:,locl));
sv = std(kLFDAvar(:,:,locl));
sprintf('kLFDAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(SDArates);
v = mean(SDAvar);
sm = std(SDArates);
sv = std(SDAvar);
sprintf('SDAerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

m = mean(SELFrates);
v = mean(SELFvar);
sm = std(SELFrates);
sv = std(SELFvar);
sprintf('SELFerr: $%0.3f \\pm %0.3f$ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

%% plot error - var tradeoff curves

for t = 1:length(ks)
    figure()
    hold on
    plot(avgPCAvar(t), avgPCA(t), 'sb', 'MarkerSize', 20, 'LineWidth', 2)
    %plot(avgkPCAvar_lab(t), avgkPCA(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
    %plot(avgkLSPCAvar_lab(t,1), avgkLSPCA(t, 1), 'sr', 'LineWidth', 2, 'MarkerSize', 20)
    plot(avgLSPCAvar(t,:), avgLSPCA(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
    % plot(avgILSPCAvar_lab(t,:), avgILSPCA(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
    % plot(lambda_avgLSPCAvar_lab(t,:), lambda_avgLSPCA(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
    plot(avgkLSPCAvar(t,1:end-1,siglock), avgkLSPCA(t, 1:end-1,siglock), '.-', 'LineWidth', 2, 'MarkerSize', 20)
    plot(avgISPCAvar(t), avgISPCA(t), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgSPPCAvar(t), avgSPPCA(t), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgSPCAvar(t), avgSPCA(t), '+', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgkSPCAvar(t,locb), avgkSPCA(t,locb), '>', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgLDAvar(t), avgLDA(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
    % plot([0,1], [1,1]*avgQDA(t), ':', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgLFDAvar(t), avgLFDA(t), '^', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgkLFDAvar(t,locl), avgkLFDA(t,locl), '<', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgSELFvar(t), avgSELF(t), 'o', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgSDAvar(t,locl), avgSDA(t,locl), 'd', 'MarkerSize', 20, 'LineWidth', 2)
    
    %plot([0,1], avgLR*[1,1], 'k', 'LineWidth', 4)
    
    %xlabel('Variation Explained', 'fontsize', 25)
    %title('unlab', 'fontsize', 25)
    %ylabel('Classification Error', 'fontsize', 25)
    %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
    set(gca, 'fontsize', 25)
    lgd = legend('PCA', 'LSPCA', 'KLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', ...
        'LDA', 'LFDA', 'kLFDA', 'Location', 'best'); lgd.FontSize = 15;
    xlim([0,1.01])
    %ylim([0,0.5])
    saveas(gcf, strcat(dataset, 'multi_obj_gamma_ss.jpg'))
end

%% plot error - var tradeoff curves

for t = 1:length(ks)
    figure()
    hold on
    plot(avgPCAvar_lab(t), avgPCA_lab(t), 'sb', 'MarkerSize', 20, 'LineWidth', 2)
    % plot(avgkPCAvar_lab(t), avgkPCA_lab(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
    %plot(avgkLSPCAvar_lab(t,1), avgkLSPCA_lab(t, 1), 'sr', 'LineWidth', 2, 'MarkerSize', 20)
    plot(avgLSPCAvar_lab(t,:), avgLSPCA_lab(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
    % plot(avgILSPCAvar_lab(t,:), avgILSPCA_lab(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
    % plot(lambda_avgLSPCAvar_lab(t,:), lambda_avgLSPCA_lab(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
    plot(avgkLSPCAvar_lab(t,1:end-2,siglock), avgkLSPCA_lab(t, 1:end-2,siglock), '.-', 'LineWidth', 2, 'MarkerSize', 20)
    plot(avgISPCAvar_lab(t), avgISPCA_lab(t), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgSPPCAvar_lab(t), avgSPPCA_lab(t), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgSPCAvar_lab(t), avgSPCA_lab(t), '+', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgkSPCAvar_lab(t,locb), avgkSPCA_lab(t,locb), '>', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgLDAvar_lab(t), avgLDA_lab(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
    %plot([0,1], [1,1]*avgQDA_lab(t), ':', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgLFDAvar_lab(t), avgLFDA_lab(t), '^', 'MarkerSize', 20, 'LineWidth', 2)
    plot(avgkLFDAvar_lab(t,locl), avgkLFDA_lab(t,locl), '<', 'MarkerSize', 20, 'LineWidth', 2)
    
    %plot([0,1], avgLR*[1,1], 'k', 'LineWidth', 4)
    
    xlabel('Variation Explained', 'fontsize', 25)
    %title('lab', 'fontsize', 25)
    ylabel('Classification Error', 'fontsize', 25)
    %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
    set(gca, 'fontsize', 25)
    lgd = legend('PCA', 'LSPCA', 'KLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', ...
        'LDA', 'LFDA', 'kLFDA', 'Location', 'best'); lgd.FontSize = 15;
    xlim([0,1.01])
    saveas(gcf, strcat(dataset, 'multi_obj_lab_gamma_ss.jpg'))
end



%% Visualize for select methods (only really good for classification)
load('colorblind_colormap.mat')
colormap(colorblind)

%KLSPCA
figure()

[~, gam] = min(min(avgkLSPCA,[], 3), [], 2);
[~, sig] = min(min(avgkLSPCA,[], 2), [], 3);
data = klspca_mbd_lab{gam,sig};
scatter(data(:,1), data(:,2), 100, Ylabs{gam}, 'filled', 'linewidth', 3)
%scatter3(data(:,1), data(:,2), data(:,3), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
hold on
data = klspca_mbd_unlab{row,sig};
scatter(data(:,1), data(:,2), 100, Yunlabs{gam})
%scatter3(data(:,1), data(:,2), data(:,3), 100, Yunlabs(:,:,row))
colormap(colorblind)
set(gca, 'yticklabel', '')
set(gca, 'xticklabel', '')
title(strcat('KLSPCA err:  ', num2str(min(kLSPCArates, [], 'all'))))
grid on; set(gca, 'fontsize', 25)
saveas(gcf, strcat(dataset, 'KLSPCA_gamma_ss.jpg'))

%LSPCA
figure()
[r,row] = min(min(LSPCArates, [], 2));
[r,col] = min(min(LSPCArates, [], 1));
data = lspca_mbd_lab{row,col};
scatter(data(:,1), data(:,2), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
%scatter3(data(:,1), data(:,2), data(:,3), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
hold on
data = lspca_mbd_unlab{row,col};
scatter(data(:,1), data(:,2), 100, Yunlabs(:,:,row))
%scatter3(data(:,1), data(:,2), data(:,3), 100, Yunlabs(:,:,row))
colormap(colorblind)
set(gca, 'yticklabel', '')
set(gca, 'xticklabel', '')
title(strcat('LSPCA err:  ', num2str(min(LSPCArates, [], 'all'))))
grid on; set(gca, 'fontsize', 25)
saveas(gcf, strcat(dataset, 'LSPCA_gamma_ss.jpg'))

% %LSPCA
% figure()
% [r,row] = min(min(ILSPCArates, [], 2));
% [r,col] = min(min(ILSPCArates, [], 1));
% data = Ilspca_mbd_lab{row,col};
% %scatter(data(:,1), data(:,2), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% scatter3(data(:,1), data(:,2), data(:,3), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% hold on
% data = Ilspca_mbd_unlab{row,col};
% %scatter(data(:,1), data(:,2), data(:,3), 100, Yunlabs(:,:,row))
% scatter3(data(:,1), data(:,2), data(:,3), 100, Yunlabs(:,:,row))
% colormap(colorblind)
% set(gca, 'yticklabel', '')
% set(gca, 'xticklabel', '')
% title(strcat('LSPCA err:  ', num2str(min(LSPCArates, [], 'all'))))
% grid on; set(gca, 'fontsize', 25)
% saveas(gcf, strcat(dataset, 'LSPCA_gamma_ss.jpg'))
%
% %lambda LSPCA
% figure()
% [r,row] = min(min(lambda_LSPCArates, [], 2));
% [r,col] = min(min(lambda_LSPCArates, [], 1));
% data = lambda_lspca_mbd_lab{row,col};
% %scatter(data(:,1), data(:,2), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% scatter3(data(:,1), data(:,2), data(:,3), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% hold on
% data = lambda_lspca_mbd_unlab{row,col};
% %scatter(data(:,1), data(:,2), data(:,3), 100, Yunlabs(:,:,row))
% scatter3(data(:,1), data(:,2), data(:,3), 100, Yunlabs(:,:,row))
% colormap(colorblind)
% set(gca, 'yticklabel', '')
% set(gca, 'xticklabel', '')
% title(strcat('lambda LSPCA err:  ', num2str(min(LSPCArates, [], 'all'))))
% grid on; set(gca, 'fontsize', 25)
% saveas(gcf, strcat(dataset, 'LSPCA_gamma_ss.jpg'))

%PCA
figure()
[r,loc] = min(PCArates);
data = Zpcas{1};
scatter(data(:,1), data(:,2), 100, Ylabs(:,:,loc), 'filled', 'linewidth', 3)
hold on
data = pcaXunlabs{1};
scatter(data(:,1), data(:,2), 100, Yunlabs(:,:,loc))
colormap(colorblind)
set(gca, 'yticklabel', '')
set(gca, 'xticklabel', '')
title(strcat('PCA err:  ', num2str(r)))
grid on; set(gca, 'fontsize', 25)
saveas(gcf, strcat(dataset, 'PCA_ss.jpg'))

%Barshan
figure()
[r,loc] = min(SPCArates);
data = Zspcas{loc};
scatter(data(:,1), data(:,2), 100, Ylabs(:,:,loc), 'filled', 'linewidth', 3)
hold on
data = spcaXunlabs{loc};
scatter(data(:,1), data(:,2), 100, Yunlabs(:,:,loc))
colormap(colorblind)
set(gca, 'yticklabel', '')
set(gca, 'xticklabel', '')
title(strcat('Barshan err:  ', num2str(r)))
grid on; set(gca, 'fontsize', 25)
saveas(gcf, strcat(dataset, 'Barshan_ss.jpg'))

%kBarshan
figure()
[r,loc] = min(kSPCArates);
data = Zkspcas{loc};
scatter(data(:,1), data(:,2), 100, Ylabs(:,:,loc), 'filled', 'linewidth', 3)
hold on
data = kspcaXunlabs{loc};
scatter(data(:,1), data(:,2), 100, Yunlabs(:,:,loc))
colormap(colorblind)
set(gca, 'yticklabel', '')
set(gca, 'xticklabel', '')
title(strcat('kBarshan err:  ', num2str(r)))
grid on; set(gca, 'fontsize', 25)
saveas(gcf, strcat(dataset, 'kBarshan_ss.jpg'))

%ISPCA
figure()
[r,loc] = min(ISPCArates);
data = Zispcas{loc};
scatter(data(:,1), data(:,2), 100, Ylabs(:,:,loc), 'filled', 'linewidth', 3)
hold on
data = ISPCAXunlabs{loc};
scatter(data(:,1), data(:,2), 100, Yunlabs(:,:,loc))
colormap(colorblind)
set(gca, 'yticklabel', '')
set(gca, 'xticklabel', '')
title(strcat('ISPCA err:  ', num2str(r)))
grid on; set(gca, 'fontsize', 25)
saveas(gcf, strcat(dataset, 'ISPCA_ss.jpg'))

%SPPCA
figure()
[r,loc] = min(SPPCArates);
data = Zsppcas{loc};
scatter(data(:,1), data(:,2), 100, Ylabs(:,:,loc), 'filled', 'linewidth', 3)
hold on
data = SPPCAXunlabs{loc};
scatter(data(:,1), data(:,2), 100, Yunlabs(:,:,loc))
colormap(colorblind)
set(gca, 'yticklabel', '')
set(gca, 'xticklabel', '')
title(strcat('SPPCA err:  ', num2str(r)))
grid on; set(gca, 'fontsize', 25)
saveas(gcf, strcat(dataset, 'SPPCA_ss.jpg'))





%% setup and load data
%rng(0);
load(strcat(dataset, '.mat'));
[n, p] = size(X);
[~, q] = size(Y);
ks = 2:min(10, p-1);


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


%% LSPCA (MLE)
for t = 1:length(ks) %dimensionality of reduced data
    k = ks(t)
    Lambdas = fliplr(logspace(-2, 0, 40));
    Gammas = [1, 1.5, 2, 5, 10];
    
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
        for ii = 1:length(Gammas)
            Gamma = Gammas(ii);
            for jj = 1:length(Lambdas)
                Lambda = Lambdas(jj);
                
                if jj == 1
                    [LSPCAXunlab, LSPCAYunlab, Zlspca, Llspca, B] = lspca_gamma_ss(Xunlab, Xlab, Ylab, Lambda, Gamma, k, 0);
                    
                else
                    [LSPCAXunlab, LSPCAYunlab, Zlspca, Llspca, B] = lspca_gamma_ss(Xunlab, Xlab, Ylab, Lambda, Gamma, k, Llspca);
                    
                end
                Ls{l,t,ii, jj} = Llspca;
                %predict
                LSPCAXlab = Zlspca;
                LSPCAYlab = LSPCAXlab*B;
                lspca_mbd_unlab{l,t,ii,jj} =LSPCAXunlab;
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
    mean(LSPCArates)
end

%% kLSPCA
for t = 1:length(ks) %dimensionality of reduced data
    k = ks(t)
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
                    kLs{l,t,ii,jj,kk} = Lorth;
                    klspca_mbd_unlab{l,t,ii,jj,kk} = kLSPCAXunlab;
                    kLSPCAXlab = Zklspca;
                    klspca_mbd_lab{l,t,ii,jj,kk} = Zklspca;
                    kLSPCAYlab = kLSPCAXlab*B;
                    mse = norm(kLSPCAYunlab - Yunlab, 'fro')^2 / nunlab;
                    lab_err = norm(Ylab - kLSPCAYlab, 'fro')^2 / nlab;
                    kLSPCArates(l,t,ii,jj,kk) = mse ;
                    kLSPCArates_lab(l,t,ii,jj,kk) = lab_err;
                    Klab = Klspca(1:nlab,:);
                    Kunlab = Klspca(nlab+1:end,:);
                    kLSPCAvar(l,t,ii,jj,kk) = norm(kLSPCAXunlab, 'fro') / norm(gaussian_kernel(Xunlab,Xlab,sigma), 'fro');
                    kLSPCAvar_lab(l,t,ii,jj,kk) = norm(Zklspca, 'fro') / norm(Klspca, 'fro');
                end
            end
        end
        mean(kLSPCArates)
    end
end

%% save all data
save(strcat(dataset, '_results_dim_mle_ss_onlyUs'))

%% compute avg performance for each k

%means
avgLSPCA = mean(LSPCArates, 1);
avgLSPCA_lab = mean(LSPCArates_lab, 1);
avgkLSPCA = mean(kLSPCArates, 1);
avgkLSPCA_lab = mean(kLSPCArates_lab, 1);
avgLSPCAvar = mean(LSPCAvar, 1);
avgkLSPCAvar = mean(kLSPCAvar, 1);
avgLSPCAvar_lab = mean(LSPCAvar_lab, 1);
avgkLSPCAvar_lab = mean(kLSPCAvar_lab, 1);


%% Print results over all subspace dimensions

loc = find(avgLSPCA==min(avgLSPCA,[],'all'),1,'last');
[~,kloc] = ind2sub(size(avgLSPCA), loc);
k = ks(kloc);
m = mean(LSPCArates(:,kloc), 1);
v = mean(LSPCAvar(:,kloc), 1);
sm = std(LSPCArates(:,kloc), 1);
sv = std(LSPCAvar(:,kloc), 1);
sprintf('LSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)


loc = find(avgkLSPCA==min(avgkLSPCA,[],'all'),1,'last');
[~,klock,siglock] = ind2sub(size(avgkLSPCA), loc);
k = ks(klock);
m = mean(kLSPCArates(:,klock,siglock), 1);
v = mean(kLSPCAvar(:,klock,siglock), 1);
sm = std(kLSPCArates(:,klock,siglock), 1);
sv = std(kLSPCAvar(:,klock,siglock), 1);
sprintf('kLSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)


%% print mean performance with std errors for fixed k
k=1
kloc = 2;

m = mean(LSPCArates(:,kloc), 1);
v = mean(LSPCAvar(:,kloc), 1);
sm = std(LSPCArates(:,kloc), 1);
sv = std(LSPCAvar(:,kloc), 1);
sprintf('LSPCAerr: $%0.3f \\pm %0.3f $ & $%0.3f \\pm %0.3f$', m, sm, v, sv)


loc = find(avgkLSPCA(:,kloc,:)==min(avgkLSPCA(:,kloc,:),[],'all'),1,'last');
[~,~,sigloc] = ind2sub(size(avgkLSPCA(:,kloc,:)), loc);
m = mean(kLSPCArates(:,kloc,sigloc), 1);
v = mean(kLSPCAvar(:,kloc,sigloc), 1);
sm = std(kLSPCArates(:,kloc,sigloc), 1);
sv = std(kLSPCAvar(:,kloc,sigloc), 1);
sprintf('kLSPCAerr: $%0.3f \\pm %0.3f $ & $%0.3f \\pm %0.3f$', m, sm, v, sv)

%%% plot error - var tradeoff curves
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
%     plot(avgSSVDvar_lab(t), avgSSVD(t), 'd', 'MarkerSize', 20, 'LineWidth', 2)
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
% saveas(gcf, strcat(dataset, 'multi_obj_gamma.jpg'))
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
% saveas(gcf, strcat(dataset, 'multi_obj_lab_gamma.jpg'))
%
%
%




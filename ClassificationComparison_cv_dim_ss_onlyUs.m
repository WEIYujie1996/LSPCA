%% setup and load data
%rng(0);
ks = 2:10;
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

%% LRPCA
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
                    [LSPCAXunlab, LSPCAYunlab,Zlspca, Llspca, B] = lrpca_gamma_lambda_ss(Xunlab, Xlab, Ylab, Lambda, Gamma, k, 0);
                else
                    [LSPCAXunlab, LSPCAYunlab,Zlspca, Llspca, B] = lrpca_gamma_lambda_ss(Xunlab, Xlab, Ylab, Lambda, Gamma, k, Llspca);
                end
                Ls{l,t,ii, jj} = Llspca;
                %predict
                LSPCAXlab = Xlab*Llspca;
                [~, LSPCAYlab] = max(Xlab*Llspca*B(2:end,:) + B(1,:), [], 2);
                lspca_mbd_unlab{l,t,ii,jj} =LSPCAXunlab;
                lspca_mbd_lab{l,t,ii,jj} = Zlspca;
                % compute error
                err = 1 - sum(Yunlab == LSPCAYunlab) / nunlab;
                lab_err = 1 - sum(Ylab == LSPCAYlab) / nlab;                
                LSPCArates(l,t, ii, jj) = err ;
                LSPCArates_lab(l,t, ii, jj) = lab_err;
                LSPCAvar(l,t, ii, jj) = norm(LSPCAXunlab, 'fro') / norm(Xunlab, 'fro');
                LSPCAvar_lab(l,t, ii, jj) = norm(Zlspca, 'fro') / norm(Xlab, 'fro');
            end
        end
    end
    
end

%% kLRPCA
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
        
        for kk = 1:length(sigmas)
            sigma = sigmas(kk)
            for ii = 1:length(Gammas)
                Gamma = Gammas(ii);
                for jj = 1:length(Lambdas)
                    Lambda = Lambdas(jj);
                    if jj == 1
                        [kLSPCAXunlab, kLSPCAYunlab, Zklspca, Lorth, B, Klspca] = klrpca_gamma_lambda_ss(Xunlab, Xlab, Ylab, Lambda, Gamma, sigma, k, 0, 0);
                    else
                        [kLSPCAXunlab, kLSPCAYunlab, Zklspca, Lorth, B, Klspca] = klrpca_gamma_lambda_ss(Xunlab, Xlab, Ylab, Lambda, Gamma, sigma, k, Lorth, Klspca);
                    end
                    kLs{l,t,ii,jj,kk} = Lorth;
                    klspca_mbd_unlab{l,t,ii,jj,kk} = kLSPCAXunlab;
                    kLSPCAXlab = Zklspca;
                    klspca_mbd_lab{l,t,ii,jj,kk} = Zklspca;
                    [~, kLSPCAYlab] = max(Zklspca*B(2:end,:) + B(1,:), [], 2);
                    err = 1 - sum(Yunlab == kLSPCAYunlab)/nunlab;
                    lab_err = 1 - sum(Ylab == kLSPCAYlab)/nlab;
                    kLSPCArates(l,t,ii,jj,kk) = err ;
                    kLSPCArates_lab(l,t,ii,jj,kk) = lab_err;
                    Klab = Klspca(1:nlab,:);
                    Kunlab = Klspca(nlab+1:end,:);
                    kLSPCAvar(l,t,ii,jj,kk) = norm(kLSPCAXunlab, 'fro') / norm(Kunlab, 'fro');
                    kLSPCAvar_lab(l,t,ii,jj,kk) = norm(Zklspca, 'fro') / norm(Klab, 'fro');
                end
            end
        end
    end
    min(mean(kLSPCArates), [], 'all')
end
    
%% save all data
save(strcat(dataset, '_results_dim_mle_onlyUs'))

%% compute avg performance accross folds


avgLSPCA = mean(LSPCArates);
avgLSPCA_lab = mean(LSPCArates_lab);

avgkLSPCA = mean(kLSPCArates);
avgkLSPCA_lab = mean(kLSPCArates_lab);

avgLSPCAvar = mean(LSPCAvar);
avgkLSPCAvar = mean(kLSPCAvar);
avgLSPCAvar_lab = mean(LSPCAvar_lab);
avgkLSPCAvar_lab = mean(kLSPCAvar_lab);


%% print mean performance with std errors

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


%% print mean performance with std errors for k=2
k = 2;
kloc = 1;
klock=1;


loc = find(avgLSPCA(:,kloc,:)==min(avgLSPCA(:,kloc,:),[],'all'),1,'last');
[~,~,gamloc] = ind2sub(size(avgLSPCA(:,1,:)), loc);
m = mean(LSPCArates(:,kloc,gamloc), 1);
v = mean(LSPCAvar(:,kloc,gamloc), 1);
sm = std(LSPCArates(:,kloc,gamloc), 1);
sv = std(LSPCAvar(:,kloc,gamloc), 1);
sprintf('LSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)


loc = find(avgkLSPCA(:,kloc,:,:)==min(avgkLSPCA(:,kloc,:,:),[],'all'),1,'last');
[~,~,gamlock,siglock] = ind2sub(size(avgkLSPCA(:,kloc,:,:)), loc);
m = mean(kLSPCArates(:,klock,gamlock,siglock), 1);
v = mean(kLSPCAvar(:,klock,gamlock,siglock), 1);
sm = std(kLSPCArates(:,klock,gamlock,siglock), 1);
sv = std(kLSPCAvar(:,klock,gamlock,siglock), 1);
sprintf('kLSPCAerr: $%0.3f \\pm %0.3f \\ (%i)$ & $%0.3f \\pm %0.3f$', m, sm, k, v, sv)


%% plot error - var tradeoff curves
% 
% for t = 1:length(ks)
%     figure()
%     hold on
%     plot(avgPCAvar(t), avgPCA(t), 'sb', 'MarkerSize', 20, 'LineWidth', 2)
%     %plot(avgkPCAvar_lab(t), avgkPCA(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
%     %plot(avgkLSPCAvar_lab(t,1), avgkLSPCA(t, 1), 'sr', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgLSPCAvar(t,:), avgLSPCA(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     % plot(avgILSPCAvar_lab(t,:), avgILSPCA(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     % plot(lambda_avgLSPCAvar_lab(t,:), lambda_avgLSPCA(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
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
%     %title('unlab', 'fontsize', 25)
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
%     plot(avgPCAvar_lab(t), avgPCA_lab(t), 'sb', 'MarkerSize', 20, 'LineWidth', 2)
%     % plot(avgkPCAvar_lab(t), avgkPCA_lab(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
%     %plot(avgkLSPCAvar_lab(t,1), avgkLSPCA_lab(t, 1), 'sr', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgLSPCAvar_lab(t,:), avgLSPCA_lab(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     % plot(avgILSPCAvar_lab(t,:), avgILSPCA_lab(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     % plot(lambda_avgLSPCAvar_lab(t,:), lambda_avgLSPCA_lab(t, :), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgkLSPCAvar_lab(t,1:end-2,siglock), avgkLSPCA_lab(t, 1:end-2,siglock), '.-', 'LineWidth', 2, 'MarkerSize', 20)
%     plot(avgISPCAvar_lab(t), avgISPCA_lab(t), 'mx', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPPCAvar_lab(t), avgSPPCA_lab(t), 'pc', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgSPCAvar_lab(t), avgSPCA_lab(t), '+', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgkSPCAvar_lab(t,locb), avgkSPCA_lab(t,locb), '>', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgLDAvar_lab(t), avgLDA_lab(t), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
%     %plot([0,1], [1,1]*avgQDA_lab(t), ':', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgLFDAvar_lab(t), avgLFDA_lab(t), '^', 'MarkerSize', 20, 'LineWidth', 2)
%     plot(avgkLFDAvar_lab(t,locl), avgkLFDA_lab(t,locl), '<', 'MarkerSize', 20, 'LineWidth', 2)
%     
%     %plot([0,1], avgLR*[1,1], 'k', 'LineWidth', 4)
%     
%     xlabel('Variation Explained', 'fontsize', 25)
%     %title('lab', 'fontsize', 25)
%     ylabel('Classification Error', 'fontsize', 25)
%     %title(sprintf('k = %d', ks(t)), 'fontsize', 30)
%     set(gca, 'fontsize', 25)
%     lgd = legend('PCA', 'LSPCA', 'KLSPCA', 'ISPCA', 'SPPCA', 'Barshan', 'kBarshan', ...
%         'LDA', 'LFDA', 'kLFDA', 'Location', 'best'); lgd.FontSize = 15;
%     xlim([0,1.01])
%     saveas(gcf, strcat(dataset, 'multi_obj_lab.jpg'))
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
% data = klspca_mbd_lab{loc};
% scatter(data(:,1), data(:,2), 100, Ylabs{idx}, 'filled', 'linewidth', 3)
% %scatter3(data(:,1), data(:,2), data(:,3), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% hold on
% data = klspca_mbd_unlab{loc};
% scatter(data(:,1), data(:,2), 100, Yunlabs{idx})
% %scatter3(data(:,1), data(:,2), data(:,3), 100, Yunlabs(:,:,row))
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
% data = lspca_mbd_lab{loc};
% scatter(data(:,1), data(:,2), 100, Ylabs{idx}, 'filled', 'linewidth', 3)
% %scatter3(data(:,1), data(:,2), data(:,3), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% hold on
% data = lspca_mbd_unlab{loc};
% scatter(data(:,1), data(:,2), 100, Yunlabs{idx})
% %scatter3(data(:,1), data(:,2), data(:,3), 100, Yunlabs(:,:,row))
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
% % data = Ilspca_mbd_lab{row,col};
% % %scatter(data(:,1), data(:,2), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% % scatter3(data(:,1), data(:,2), data(:,3), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% % hold on
% % data = Ilspca_mbd_unlab{row,col};
% % %scatter(data(:,1), data(:,2), data(:,3), 100, Yunlabs(:,:,row))
% % scatter3(data(:,1), data(:,2), data(:,3), 100, Yunlabs(:,:,row))
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
% % data = lambda_lspca_mbd_lab{row,col};
% % %scatter(data(:,1), data(:,2), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% % scatter3(data(:,1), data(:,2), data(:,3), 100, Ylabs(:,:,row), 'filled', 'linewidth', 3)
% % hold on
% % data = lambda_lspca_mbd_unlab{row,col};
% % %scatter(data(:,1), data(:,2), data(:,3), 100, Yunlabs(:,:,row))
% % scatter3(data(:,1), data(:,2), data(:,3), 100, Yunlabs(:,:,row))
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
% scatter(data(:,1), data(:,2), 100, Ylabs{loc}, 'filled', 'linewidth', 3)
% hold on
% data = pcaXunlabs{loc};
% scatter(data(:,1), data(:,2), 100, Yunlabs{loc})
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
% scatter(data(:,1), data(:,2), 100, Ylabs{loc}, 'filled', 'linewidth', 3)
% hold on
% data = spcaXunlabs{loc};
% scatter(data(:,1), data(:,2), 100, Yunlabs{loc})
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
% scatter(data(:,1), data(:,2), 100, Ylabs{idx}, 'filled', 'linewidth', 3)
% hold on
% data = kspcaXunlabs{loc};
% scatter(data(:,1), data(:,2), 100, Yunlabs{idx})
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
% scatter(data(:,1), data(:,2), 100, Ylabs{loc}, 'filled', 'linewidth', 3)
% hold on
% data = ISPCAXunlabs{loc};
% scatter(data(:,1), data(:,2), 100, Yunlabs{loc})
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
% scatter(data(:,1), data(:,2), 100, Ylabs{loc}, 'filled', 'linewidth', 3)
% hold on
% data = SPPCAXunlabs{loc};
% scatter(data(:,1), data(:,2), 100, Yunlabs{loc})
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
% scatter(data(:,1), data(:,2), 100, Ylabs{loc}, 'filled', 'linewidth', 3)
% hold on
% data = SPPCAXunlabs{loc};
% scatter(data(:,1), data(:,2), 100, Yunlabs{loc})
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
% scatter(data(:,1), data(:,2), 100, Ylabs{loc}, 'filled', 'linewidth', 3)
% hold on
% data = SPPCAXunlabs{loc};
% scatter(data(:,1), data(:,2), 100, Yunlabs{loc})
% colormap(colorblind)
% set(gca, 'yticklabel', '')
% set(gca, 'xticklabel', '')
% title(strcat('SPPCA err:  ', num2str(r)))
% grid on; set(gca, 'fontsize', 25)
% saveas(gcf, strcat(dataset, 'SPPCA.jpg'))




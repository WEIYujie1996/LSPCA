numExps = 10;
for dd = 1:numExps
    %% setup and load data
    load(strcat(dataset, '.mat'));
    [n, p] = size(X);
    [~, q] = size(Y);
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
        
    end
    
    %% compute avg performance for each k
    
    %means
    avgLSPCA = mean(LSPCArates(1:end-1,:,:), 1);
    avgLSPCA_train = mean(LSPCArates_train(1:end-1,:,:), 1);
    avgkLSPCA = mean(kLSPCArates(1:end-1,:,:,:), 1);
    avgkLSPCA_train = mean(kLSPCArates_train(1:end-1,:,:,:), 1);

    avgLSPCAvar = mean(LSPCAvar(1:end-1,:,:), 1);
    avgkLSPCAvar = mean(kLSPCAvar(1:end-1,:,:,:), 1);
    
    avgLSPCAvar_train = mean(LSPCAvar_train(1:end-1,:,:), 1);
    avgkLSPCAvar_train = mean(kLSPCAvar_train(1:end-1,:,:,:), 1);
    
    %% Calc performance for best model and store
    
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
    
    LSPCAval_track(dd,:,:,:) = LSPCArates;
    LSPCAvar_track(dd,:,:,:) = LSPCAvar;
    kLSPCAval_track(dd,:,:,:) = kLSPCArates;
    kLSPCAvar_track(dd,:,:,:) = kLSPCAvar;
    
end

%% save all data
save(strcat(dataset, '_lambda_curve'))

% %% plot error - var tradeoff curves
% 
% % for t = 1:length(ks)
% figure()
% hold on
% plot(mean(PCAvalVar_fixed), mean(PCAval_fixed), 'sk', 'MarkerSize', 20, 'LineWidth', 2)
% % plot(mean(kPCAvalVar_fixed), mean(kPCAval_fixed), 'sr', 'MarkerSize', 20, 'LineWidth', 2)
% plot(LSPCAvar(end,2,:), LSPCArates(end,2,:), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(kLSPCAvar(end,2,:), kLSPCArates(end,2,:), '.-', 'LineWidth', 2, 'MarkerSize', 20)
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
% plot(mean(LSPCAvalVar), mean(LSPCAval), '.-', 'LineWidth', 2, 'MarkerSize', 20)
% plot(mean(kLSPCAvalVar), mean(kLSPCAval), '.-', 'LineWidth', 2, 'MarkerSize', 20)
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


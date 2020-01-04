function [Xtrain,Xtest,Ytrain,Ytest] = ss_center_data(Xtrain,Xtest,Ytrain,Ytest,type)

[ntrain,~] = size(Xtrain);
[ntest,~] = size(Xtest);


% properly normalize the training data
Xmu = mean([Xtrain;Xtest]); %center using both Xs since we are doing transductive semi-supervised learning
Xstd = std([Xtrain;Xtest]);
Xstd(Xstd==0) = 1; %dont normalize variance of constant features (should probably be removed from data)
Xtrain = Xtrain - ones(ntrain, 1)*Xmu;
Xtrain = Xtrain ./ (ones(ntrain, 1)*Xstd);
%apply the same normalization to the testing data
Xtest = Xtest - ones(ntest, 1)*Xmu;
Xtest = Xtest ./ (ones(ntest, 1)*Xstd);

% normalize Y if doing regression
if strcmp(type, 'regression')
    Ytrainmu = mean(Ytrain);
    Ytrainstd = std(Ytrain);
    Ytrainstd(Ytrainstd==0); %dont normalize variance of constant labels (should probably be removed from data)
    Ytrain = Ytrain - ones(ntrain, 1)*Ytrainmu;
    Ytrain = Ytrain ./ (ones(ntrain, 1)*Ytrainstd);
    Ytest = Ytest - ones(ntest, 1)*Ytrainmu;
    Ytest = Ytest ./ (ones(ntest, 1)*Ytrainstd);
end
end


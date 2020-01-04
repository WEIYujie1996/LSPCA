function [Xtrain,Xtest,Ytrain,Ytest] = center_data(Xtrain,Xtest,Ytrain,Ytest,type)

[ntrain,~] = size(Xtrain);
[ntest,~] = size(Xtest);


% properly normalize the training data
Xtrainmu = mean(Xtrain);
Xtrainstd = std(Xtrain);
Xtrainstd(Xtrainstd==0) = 1; %dont normalize variance of constant features (should probably be removed from data)
Xtrain = Xtrain - ones(ntrain, 1)*Xtrainmu;
Xtrain = Xtrain ./ (ones(ntrain, 1)*Xtrainstd);
%apply the same normalization to the testing data
Xtest = Xtest - ones(ntest, 1)*Xtrainmu;
Xtest = Xtest ./ (ones(ntest, 1)*Xtrainstd);

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


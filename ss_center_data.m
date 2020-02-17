function [Xlab,Xunlab,Xhold,Ylab,Yunlab,Yhold] = ss_center_data(Xlab,Xunlab,Xhold,Ylab,Yunlab,Yhold,type)

[nlab,~] = size(Xlab);
[nunlab,~] = size(Xunlab);
[nhold,~] = size(Xhold);

% properly normalize the training data
Xmu = mean([Xlab;Xunlab]); %center using both Xs since we are doing transductive semi-supervised learning
Xstd = std([Xlab;Xunlab]);
Xstd(Xstd==0) = 1; %dont normalize variance of constant features (should probably be removed from data)
Xlab = Xlab - ones(nlab, 1)*Xmu;
Xlab = Xlab ./ (ones(nlab, 1)*Xstd);
%apply the same normalization to the unlabled data
Xunlab = Xunlab - ones(nunlab, 1)*Xmu;
Xunlab = Xunlab ./ (ones(nunlab, 1)*Xstd);
%apply the same normalization to the holdout data
Xhold = Xhold - ones(nhold, 1)*Xmu;
Xhold = Xhold ./ (ones(nhold, 1)*Xstd);

% normalize Y if doing regression
if strcmp(type, 'regression')
    Ylabmu = mean(Ylab);
    Ylabstd = std(Ylab);
    Ylabstd(Ylabstd==0); %dont normalize variance of constant labels (should probably be removed from data)
    Ylab = Ylab - ones(nlab, 1)*Ylabmu;
    Ylab = Ylab ./ (ones(nlab, 1)*Ylabstd);
    Yunlab = Yunlab - ones(nunlab, 1)*Ylabmu;
    Yunlab = Yunlab ./ (ones(nunlab, 1)*Ylabstd);
    Yhold = Yhold - ones(nhold, 1)*Ylabmu;
    Yhold = Yhold ./ (ones(nhold, 1)*Ylabstd);
end
end


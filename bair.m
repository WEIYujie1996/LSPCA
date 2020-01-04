function [Zbair, Lbair, Lbair_partial, bair_mask, bairXtrain ] = bair(k, ks, p, Xtrain, Xtrainparam, Xtestparam, Ytrain, Ytrainparam, Ytestparam, nHoldTrain)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
if k < (nHoldTrain-1)
            BSPCratesparam = [];
            LScoeffs = (Xtrainparam' * Ytrainparam) ./ vecnorm(Xtrainparam,2,1)';
            theta_max = maxk(abs(LScoeffs), k); theta_max = theta_max(end);
            thetas = linspace(min(abs(LScoeffs)), theta_max, 100);
            thetas;
            for i = 1:length(thetas)
                theta = thetas(i);
                if sum(abs(LScoeffs) >= theta) >= k
                    bair_train = Xtrainparam(:, abs(LScoeffs) >= theta);
                    bair_test = Xtestparam(:, abs(LScoeffs) >= theta);
                    % compute bair's spc
                    [COEFF, Zbair] = pca(bair_train);
                    %size(COEFF)
                    Zbair = Zbair(:, 1:k);
                    % compute embedding for test data
                    bairXtest = bair_test*COEFF(:,1:k);
                    % mse of the regression
                    betaBAIR = pinv(Zbair)*Ytrainparam;
%                     betaBAIR2 = COEFF(:,1:k)*betaBAIR;
%                     size(bairXtest)
%                     size(betaBAIR)
%                     size(betaBAIR2)
                    BAIRYtest = bairXtest * betaBAIR;
                    
                    mse = norm(BAIRYtest - Ytestparam, 'fro');
                    BSPCratesparam(i) = mse;
                
                else
                BSPCratesparam = 1;    
                end
            end
            
               
            %find parameter that resulted in lowest mse
            BSPCratesparam;
            [~, loc] = min(BSPCratesparam);
            loc = loc(end);
            bestTheta = thetas(loc);
            %now that best numfeats has been selected learn on all training data
            
            %note Bair's SPC is based on univariate regression coefficients, therefore Y must be a vector
            LScoeffs = (Xtrain' * Ytrain) ./ vecnorm(Xtrain,2,1)';
            theta_max = maxk(abs(LScoeffs), k);
            theta_max = theta_max(end);
            bestTheta = min(bestTheta, theta_max);
            
            %select the features for the full testing data
            bair_mask = abs(LScoeffs) >= bestTheta;
            bair_train = Xtrain(:, bair_mask);            
            
            % compute bair's spc
            [Lbair, Zbair] = pca(bair_train);
            Zbair = Zbair(:, 1:k);
            Lbair = Lbair(:, 1:k)';
            % compute embedding for test data
            
            
            % convert Lbair
            Lbair_partial = Lbair;
            temp = zeros(k, p);
            temp(:, bair_mask) = Lbair;
            Lbair = temp;
            
        end
end


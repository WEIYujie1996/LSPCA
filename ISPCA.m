function [Z,L,B] = ISPCA(X,Y,k)
%Implementation of ISPCA as described in
% Piironen, Juho, and Aki Vehtari. "Iterative Supervised Principal Components." arXiv preprint arXiv:1710.06229 (2017).

% inputs: X : nxp matrix of independent variables
%         Y : nxq matrix of dependent variables
%         M : as described in section 3.5 of the above paper
%         W : as described in section 3.5 of the above paper
%         k : desired number of components

% outputs: Z : nxk matrix (reduced form of X)
%          L : orthonormal basis s.t. Z = X*L'
%          B : LS regression coefficients s.t. Y ~ X*L'*B
%          feats : index of the features used in the 

% treat score function as univariate regression coefficient


%keep a copy of original X since it will change through this process
X_orig = X;

[n,p] = size(X);
[~, q] = size(Y);
L = zeros(k, p);
M = 200;
W = min(1000, p-1);
for ii = 1:k
    score_gamma_cand = [];
    v_gamma_cand = {};
    idx_gammas = {};
    for kk = 1:q
        % calculate univariate regression coefficients
        norms = vecnorm(X,2,1)';
        norms(norms == 0) = 1; %to prevent dividing by zero
        S = abs((X' * Y(:,kk)) ./ norms);
        % determine gamma_min and gamma_max then perform grid search
        [sorted, idx] = sort(S, 'descend');
        gamma_max = sorted(2);
        gamma_min = sorted(W);
        gammas = linspace(gamma_min, gamma_max, M);
        scores_jj = [];
        for jj = 1:length(gammas)
            idx_jj = idx(sorted >= gammas(jj));
            idx_jj = sort(idx_jj, 'ascend'); %grab the features in order
            X_jj = X(:, idx_jj);
            [~, ~, v_jj] = svds(X_jj, 1);
            z_jj = X_jj*v_jj;
            scores_jj(jj) = z_jj'* Y(:,kk) / vecnorm(z_jj)^2;
        end
        % calculate kth ISPC using found gamma
        gamma = gammas(find(scores_jj == max(scores_jj), 1));
        idx_gamma = idx(sorted >= gamma);
        idx_gamma = sort(idx_gamma, 'ascend');
        idx_gammas{kk} = idx_gamma;
        X_gamma = X(:, idx_gamma);
        [~, ~, v_gamma] = svds(X_gamma, 1);
        %store then find max v_gamma across features in Y
        v_gamma_cand{kk} = v_gamma;
        z_gamma = X_gamma*v_gamma;
        score_gamma_cand = z_gamma' * Y(:,kk) / vecnorm(z_gamma)^2;
    end
    %find v_gamma_cand with maximum score
    best_score_loc = find(score_gamma_cand == max(score_gamma_cand), 1);
    v_gamma = v_gamma_cand{best_score_loc};
    idx_gamma = idx_gammas{best_score_loc};
    % save this ISPC
    L(ii, idx_gamma) = v_gamma';
    % subtract variation explained by this ISPC from X
    z = X*L(ii, :)';
    z = z / vecnorm(z);
    X_new = X - z.*(X'*z)';
    %z'*X_new %should be zero
    X = X_new;
end
% calculate reduced data
Z = X_orig*L';
B = pinv(Z)*Y;
end


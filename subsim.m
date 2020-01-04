function [determinantSimilarity, similarities] = subsim(U1,U2)
%U1 and U2 are matrices with orthonormal rows of the same dimension
[k1, p] = size(U1);
[~, ~, U1] = svds(U1,k1);
U1 = U1';
[k2, ~] = size(U2);
[~, ~, U2] = svds(U2, k2);
U2 = U2';
if k1 <= k2
similarities = eig(U1*U2'*U2*U1');
else
similarities = eig(U2*U1'*U1*U2');    
end
similarities = sort(similarities, 'descend', 'ComparisonMethod', 'abs');
determinantSimilarity = prod(similarities);
end
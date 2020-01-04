function [determinantSimilarity, similarities] = detsim(U1,U2)
%U1 and U2 are matrices of the same dimension with orthonormal rows 
% U1 = U1';
% U2 = U2';
similarities = diag((U1*U2')*(U2*U1')); 
similarities = sort(similarities, 'descend', 'ComparisonMethod', 'abs');
determinantSimilarity = min(1,prod(similarities));

end


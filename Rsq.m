function [r_squared] = Rsq(y_test,y_test_hat)
%calculate coefficient of determination (r squared)
y_bar = mean(y_test);

r_squared = 1 - ((norm(y_test - y_test_hat, 'fro')^2)/(norm(y_test - y_bar, 'fro')^2));

end


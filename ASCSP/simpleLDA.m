% simpleLDA
function y = simpleLDA(X1, X2, test)
X1_mean = mean(X1, 1).';
X2_mean = mean(X2, 1).';
cov_X1 = cov(X1);
cov_X2 = cov(X2);
cov_X = mean(cat(3,cov_X1,cov_X2), 3);
inv_cov_X = pinv(cov_X);
w = inv_cov_X*(X1_mean - X2_mean);
w = w.';
c = w*1/2*(X1_mean+X2_mean);
test_result = w*test.';
class1_idx = find(test_result > c);
y = ones(size(test,1),1)*2;
y(class1_idx(:)) = 1;
end
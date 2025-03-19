% CSP_cov.m
function W = CSP_cov(cov)
% X1 and X2 are in the form of trial by channels by time points
sigma1 = cov{1};
sigma2 = cov{2};
[EVector,~] = eig(sigma1, sigma1+sigma2);
W = [EVector(:, 1:3), EVector(:, 66:end)];
end
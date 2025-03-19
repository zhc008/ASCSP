%TRCSP.m
function [W,sigma1,sigma2] = TRCSP(alpha, X1, X2)
% X1 and X2 are in the form of trial by channels by time points
I1 = size(X1, 1);
I2 = size(X2, 1);
num_channels = size(X1, 2);
sigma1 = zeros(num_channels, num_channels);
sigma2 = zeros(num_channels, num_channels);
for i = 1:I1
    X1_temp = squeeze(X1(i,:,:));
    X1_temp = X1_temp.';
    cov1_temp = cov(X1_temp);
    sigma1 = sigma1 + cov1_temp/trace(cov1_temp);
end
sigma1 = sigma1 / I1;
for i = 1:I2
    X2_temp = squeeze(X2(i,:,:));
    X2_temp = X2_temp.';
    cov2_temp = cov(X2_temp);
    sigma2 = sigma2 + cov2_temp/trace(cov2_temp);
end
sigma2 = sigma2 / I2;
Idt = eye(size(sigma1, 1));
[EVector1,~] = eig(sigma1, sigma1+sigma2+alpha*Idt);
[EVector2,~] = eig(sigma2, sigma2+sigma1+alpha*Idt);
W = [EVector1(:, 1:3), EVector2(:, 1:3)];
end
function shrink_param = cal_shrinkage_modified(X)

% X is of dimension number of trials, number of channels and number of time samples

d = size(X,2);
% mu = mean(X)';
% mX = bsxfun(@minus,X,mu');
mu = mean(X);
mX = bsxfun(@minus,X,mu);
N = size(mX,1);
W = zeros(N,d,d);

for n=1:N
    W(n,:,:) = squeeze(mX(n,:,:)) * squeeze(mX(n,:,:))';
end
WM = mean(W,1);
S = squeeze((N/(N-1)) .* WM);

% Target 'B' of Schafer and Strimmer; choice by Blankertz et al.

VS = squeeze((N/((N-1).^3)) .* sum(bsxfun(@minus,W,WM).^2,1));

v = mean(diag(S));

t = triu(S,1);
shrink_param = sum(VS(:)) / (2*sum(t(:).^2) + sum((diag(S)-v).^2));


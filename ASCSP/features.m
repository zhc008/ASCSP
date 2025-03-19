% features.m
function f_X = features(W, X)
% size of X is trials by channels by time points
trials = size(X, 1);
m = size(W, 2);
f_X = zeros(trials, m);
for i = 1:trials
    var_temp = W.'*squeeze(X(i,:,:));
    var_temp = var(var_temp, 0, 2);
    f_temp = log(var_temp/sum(var_temp));
    f_X(i, :) = f_temp;
end
end
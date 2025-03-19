% extract_bciIV2a_train.m
function [X1_s, X2_s] = extract_bciIV2a_train(s_train, h_train)
% artifacts = h.ArtifactSelection;
% dur = h.EVENT.DUR;
types_train = h_train.EVENT.TYP;
pos_train = h_train.EVENT.POS;
%% butterworth filter the train data
s_train = s_train(:,1:22);
[b,a] = butter(5, [0.064, 0.24]);
s_train(isnan(s_train)) = 0;
s_train_f = filtfilt(b, a, s_train);
x1_pos = pos_train(types_train==769); % class 1 cue
x2_pos = pos_train(types_train==770); % class 2 cue
I1 = size(x1_pos, 1);
I2 = size(x2_pos, 1);
X1_s = zeros(I1, size(s_train_f,2), 2*250);
X2_s = zeros(I2, size(s_train_f,2), 2*250);
for i = 1:I1
    cue = x1_pos(i);
    X1 = s_train_f(cue+126:cue+625, :);
    X1_s(i, :, :) = X1.';
end
for i = 1:I2
    cue = x2_pos(i);
    X2 = s_train_f(cue+126:cue+625, :);
    X2_s(i, :, :) = X2.';
end
end
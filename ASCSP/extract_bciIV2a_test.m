% extract_bciIV2a_test.m
function [X_test_s, test_label] = extract_bciIV2a_test(s_test, h_test, label)
% artifacts = h.ArtifactSelection;
% dur = h.EVENT.DUR;
types_test = h_test.EVENT.TYP;
pos_test = h_test.EVENT.POS;
%% butterworth filter the test data
s_test = s_test(:,1:22);
[b,a] = butter(5, [0.064, 0.24]);
s_test(isnan(s_test)) = 0;
s_test_f = filtfilt(b, a, s_test);
trial_pos = pos_test(types_test==783); % unknown cue
test_pos = trial_pos(label==1|label==2);
test_label = label(label==1|label==2);
test_size = size(test_pos, 1);
X_test_s = zeros(test_size, size(s_test_f,2), 2*250);
for i = 1:test_size
    cue = test_pos(i);
    X_test = s_test_f(cue+126:cue+625, :);
    X_test_s(i, :, :) = X_test.';
end
end
% extract_data.m
%{
before calling this function, load the dataset and true labels
name1 = ['../data/data_set_IVa_aa.mat'];
name2 = ['../data/true_labels_aa.mat'];
load(name1)
load(name2)
%}
function [X1_s, X2_s, X_test_s, true_y_test] = extract_data(mrk, cnt, test_idx, true_y)
true_y_test = true_y(test_idx).';
y = mrk.y;
pos = mrk.pos;
cnt_new = 0.1*double(cnt);
channel_idx = [(15:21),(24:29),(33:39),(43:48),(51:57),(60:65),(69:75),(79:84),(88:94),(98:101),(104:108)];
cnt_68 = cnt_new(:, channel_idx);

%% apply fir1 filter
[b,a] = butter(5, [0.16, 0.6]);
cnt68_filtered = filtfilt(b, a, cnt_68);

%% find X1, X2, and X_test
y1 = find(y==1);
y2 = find(y==2);
y_test = find(isnan(y)==1);
I1 = size(y1, 2);
I2 = size(y2, 2);
test_size = size(y_test, 2);
X1_s = zeros(I1, 68, 200);
X2_s = zeros(I2, 68, 200);
X_test_s = zeros(test_size, 68, 200);
for i = 1:I1
    index = y1(i);
    cue = pos(index);
    X1 = cnt68_filtered(cue+51:cue+250, :);
    X1_s(i, :, :) = X1.';
end
for i = 1:I2
    index = y2(i);
    cue = pos(index);
    X2 = cnt68_filtered(cue+51:cue+250, :);
    X2_s(i, :, :) = X2.';
end
for i = 1:test_size
    index = y_test(i);
    cue = pos(index);
    X_test = cnt68_filtered(cue+51:cue+250, :);
    X_test_s(i, :, :) = X_test.';
end
end
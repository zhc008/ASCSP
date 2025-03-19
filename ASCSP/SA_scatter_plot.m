% SA_scatter_plot.m
clear

csp_per_class = 3;
sub_names = {'a','l','v','w','y'};

% load subject 3 as training
name1 = ['../data/data_set_IVa_a',sub_names{1},'.mat'];
name2 = ['../data/true_labels_a',sub_names{1},'.mat'];
load(name1)
load(name2)
[X1_s, X2_s, X_test_s, y_test_s] = extract_data(mrk, cnt, test_idx, true_y);

% number of trials in one class
num_tmp1 = size(X1_s,1);
num_tmp2 = size(X2_s,1);

data_source = cell(1,2); %1 left 2 right
for idx = 1:num_tmp1
    % left data
    data_source{1}{idx} = squeeze(X1_s(idx,:,:));
end

for idx = 1:num_tmp2
    % left data
    data_source{2}{idx} = squeeze(X2_s(idx,:,:));
end
[~, cov_s1, cov_s2] = CSP(X1_s, X2_s);

gm = cell(1,2); %cm 1 left 2 right
gm{1} = cov_s1;
gm{2} = cov_s2;

% load subject 1 as testing
name1_t = ['../data/data_set_IVa_a',sub_names{4},'.mat'];
name2_t = ['../data/true_labels_a',sub_names{4},'.mat'];
load(name1_t)
load(name2_t)
[X1_t, X2_t, X_test_t, y_test_t] = extract_data(mrk, cnt, test_idx, true_y);

num_tmp = size(X_test_t,1);
data_target = cell(1,1);
for idx = 1:num_tmp
    
    data_target{1}{idx} = squeeze(X_test_t(idx,:,:));
    
end
%% training CSP using new covariance matrix. data_source is useless
[ csp_coeff,all_coeff] = csp_analysis(data_source,9,csp_per_class, 0,gm);
[ data_source_filter ] = csp_filtering(data_source, csp_coeff);

data_source_log{1} = log_norm_BP(data_source_filter{1});
data_source_log{2} = log_norm_BP(data_source_filter{2});


%% Apply CSP in target subject
[ data_target_filter ] = csp_filtering(data_target, csp_coeff);
data_target_log{1} = log_norm_BP(data_target_filter{1});

%% Subapace alignment
X = zeros(0,6);
Y = zeros(0,1);
maLabeled =false(0,1);
for idx = 1:2
    for idx2 = 1:size(data_source_log{idx},2)
        X = [X;data_source_log{idx}{idx2}'];
        if idx == 1
            Y = [Y;1];
        else
            Y = [Y;2];
        end
        maLabeled = [maLabeled;true];
    end
end

for idx = 1
    for idx2 = 1:size(data_target_log{1},2)
        X = [X;data_target_log{idx}{idx2}'];
        Y = [Y;idx];
        maLabeled = [maLabeled;false];
    end
end
param = []; param.pcaCoef = 2;
[Xproj,transMdl] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);

size_tr = size(data_source_log{1},2)+size(data_source_log{2},2);
X1_sa = Xproj(1:size(data_source_log{1},2),:);
X2_sa = Xproj(size(data_source_log{1},2)+1:size_tr,:);
X_target_sa = Xproj(size_tr+1:end,:);
X1_t_idx = find(y_test_t == 1);
X2_t_idx = find(y_test_t == 2);
X1_t_sa = X_target_sa(X1_t_idx,:);
X2_t_sa = X_target_sa(X2_t_idx,:);


%% scatter plot
scatter(X1_sa(:,1),X1_sa(:,2),'o','red');
hold
scatter(X2_sa(:,1),X2_sa(:,2),'o','blue');
scatter(X1_t_sa(:,1),X1_t_sa(:,2),'+','red');
scatter(X2_t_sa(:,1),X2_t_sa(:,2),'+','blue');



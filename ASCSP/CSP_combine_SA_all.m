% CSP_SA.m
sub_names = {'a','l','v','w','y'};
acc_for_all = zeros(5,1);
csp_per_class = 3;
acc_csp = zeros(5,1);

gms = cell(1,5);
data_sources = cell(1,5);

for i = 1:5
    name1 = ['../data/data_set_IVa_a',sub_names{i},'.mat'];
    name2 = ['../data/true_labels_a',sub_names{i},'.mat'];
    load(name1)
    load(name2)
    [X1_s, X2_s, X_test_s, y_test_s] = extract_data(mrk, cnt, test_idx, true_y);
    % get the CSP filters and the covariance matrix for left hand and right
    % hand data
    % X1_s and X2_s are matrices with trials * channels * time points
    channels = size(X1_s, 2);
    % number of trials in one class
    num_tmp1 = size(X1_s,1);
    num_tmp2 = size(X2_s,1);
    x1_index = find(y_test_s == 1);
    x2_index = find(y_test_s == 2);
    sigma1 = zeros(channels, channels);
    sigma2 = zeros(channels, channels);
    
    data_source = cell(1,2); %1 left 2 right
    for idx = 1:num_tmp1
        % left data
        data_source{1}{idx} = squeeze(X1_s(idx,:,:));
        x1_tmp = squeeze(X1_s(idx,:,:));
        x1_cov = cov(x1_tmp.');
        sigma1 = sigma1 + x1_cov/trace(x1_cov);
    end
    
    for idx = 1:num_tmp2
        % left data
        data_source{2}{idx} = squeeze(X2_s(idx,:,:));
        x2_tmp = squeeze(X2_s(idx,:,:));
        x2_cov = cov(x2_tmp.');
        sigma2 = sigma2 + x2_cov/trace(x2_cov);
    end
    
    for idx = 1:length(x1_index)
        % left data
        data_source{1}{idx+num_tmp1} = squeeze(X_test_s(x1_index(idx),:,:));
        x1_tmp = squeeze(X_test_s(x1_index(idx),:,:));
        x1_cov = cov(x1_tmp.');
        sigma1 = sigma1 + x1_cov/trace(x1_cov);
    end
    
    for idx = 1:length(x2_index)
        % left data
        data_source{2}{idx+num_tmp2} = squeeze(X_test_s(x2_index(idx),:,:));
        x2_tmp = squeeze(X_test_s(x2_index(idx),:,:));
        x2_cov = cov(x2_tmp.');
        sigma2 = sigma2 + x2_cov/trace(x2_cov);
    end
    
    cov_s1 = sigma1/(num_tmp1 + length(x1_index));
    cov_s2 = sigma2/(num_tmp2 + length(x2_index));
       
    gm = cell(1,2); %cm 1 left 2 right 
    gm{1} = cov_s1;
    gm{2} = cov_s2;
    gms{i} = gm;
    data_sources{i} = data_source;
end

for j = 1:5
%     if j == i
%         continue; 
%     end
    name1_t = ['../data/data_set_IVa_a',sub_names{j},'.mat'];
    name2_t = ['../data/true_labels_a',sub_names{j},'.mat'];
    load(name1_t)
    load(name2_t)
    [X1_t, X2_t, X_test_t, y_test_t] = extract_data(mrk, cnt, test_idx, true_y);

    num_tmp = size(X_test_t,1);
    data_target = cell(1,1);
    for idx = 1:num_tmp
            
        data_target{1}{idx} = squeeze(X_test_t(idx,:,:));
            
    end
    
    new_source = cell(1,2);
    new_gm = cell(1,2);
    sigma1 = zeros(channels, channels);
    sigma2 = zeros(channels, channels);
    new_gm{1} = sigma1;
    new_gm{2} = sigma2;
    for i = [1:j-1 j+1:5]
        tmp_source = data_sources{i};
        tmp_gm = gms{i};
        for ii = 1:2
            new_gm{ii} = new_gm{ii}*size(new_source{ii},2)+tmp_gm{ii}*size(tmp_source{ii},2);
            new_source{ii} = [new_source{ii},tmp_source{ii}];
            new_gm{ii} = new_gm{ii}/size(new_source{ii},2);
        end
    end
        
    %% training CSP using new covariance matrix. data_source is useless
    [ csp_coeff,all_coeff] = csp_analysis(new_source,9,csp_per_class, 0,new_gm);
    [ data_source_filter ] = csp_filtering(new_source, csp_coeff);
    
    data_source_log{1} = log_norm_BP(data_source_filter{1});
    data_source_log{2} = log_norm_BP(data_source_filter{2});
    
    
    %% Apply CSP in target subject
    [ data_target_filter ] = csp_filtering(data_target, csp_coeff);
    data_target_log{1} = log_norm_BP(data_target_filter{1});
%     data_target_log{2} = log_norm_BP(data_target_filter{2});
        
    %% Train LDA. ASCSP without subspace alignment
    trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
    [W, B, ~] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);
    
    [X_LDA1, predicted_y_class] = lda_apply(cell2mat(data_target_log{1})', W, B);
    predicted_y_class(predicted_y_class == 1) = 2;   % incorrect choice
    predicted_y_class(predicted_y_class == -1) = 1;
    test_err = sum(abs(y_test_t - predicted_y_class))/size(y_test_t, 1);
    acc2 = 1 - test_err;
    text = ['s', int2str(j)];
    fprintf(text)
    fprintf(': %f\n', acc2)
    acc_csp(j) = acc2;
    
    %% Subapace alignment
    X = zeros(0,6);
    Y = zeros(0,1);
    maLabeled =false(0,1);
    for idx = 1:2
        for idx2 = 1:size(data_source_log{idx},2)
            X = [X;data_source_log{idx}{idx2}'];
            Y = [Y;idx];
            maLabeled = [maLabeled;true];
        end
    end
    
    for idx = 1
        for idx2 = 1:size(data_target_log{idx},2)
            X = [X;data_target_log{idx}{idx2}'];
            Y = [Y;idx];
            maLabeled = [maLabeled;false];
        end
    end
    param = []; param.pcaCoef = 2;
    [Xproj,transMdl] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);
    
    %% Train LDA. ASCSP with subspace alignment
    trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
    size_xproj = size(data_source_log{1},2) + size(data_source_log{2},2);
    [W,B,~] = lda_train_reg(Xproj(1:size_xproj,:), trainY, 0);
    
    [X_LDA, predicted_y_class] = lda_apply(Xproj(size_xproj+1:end,:), W, B);
    predicted_y_class(predicted_y_class == 1) = 2;   % incorrect choice
    predicted_y_class(predicted_y_class == -1) = 1;
    
    test_err = sum(abs(y_test_t - predicted_y_class))/size(y_test_t, 1);
    acc2 = 1 - test_err;
    text = ['s', int2str(j)];
    fprintf(text)
    fprintf(': %f\n', acc2)
    acc_for_all(j) = acc2;
end

% CSP_SA.m
acc_for_all = zeros(9,1);
csp_per_class = 3;

gms = cell(1,9);
data_sources = cell(1,9);

for i = 1:9
    name1 = ['../BCICIV_2a_gdf/A0',int2str(i),'T.gdf'];
    [s_train,h_train] = sload(name1);
    [X1_s, X2_s] = extract_bciIV2a_train(s_train, h_train);
    % get the CSP filters and the covariance matrix for left hand and right
    % hand data
    % X1_s and X2_s are matrices with trials * channels * time points
    channels = size(X1_s, 2);
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
    gms{i} = gm;
    data_sources{i} = data_source;
end

for j = 1:9
%     if j == i
%         continue; 
%     end
    name2 = ['../BCICIV_2a_gdf/true_labels/A0',int2str(j),'E.mat'];
    name3 = ['../BCICIV_2a_gdf/A0',int2str(j),'E.gdf'];
    test_label = load(name2).classlabel;
    [s_test,h_test] = sload(name3);
    [X_test_t, y_test_t] = extract_bciIV2a_test(s_test, h_test, test_label);
    
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
        
%     %% Train LDA. ASCSP without subspace alignment
%     trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
%     size_xproj = size(data_source_log{1},2) + size(data_source_log{2},2);
%     [W, B, ~] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);
%     
%     [X_LDA1, predicted_y_class] = lda_apply(cell2mat(data_target_log{1})', W, B);
%     predicted_y_class(predicted_y_class == 1) = 2;   % incorrect choice
%     predicted_y_class(predicted_y_class == -1) = 1;
%     test_err = sum(abs(y_test_t - predicted_y_class))/size(y_test_t, 1);
%     acc2 = 1 - test_err;
%     text = ['s', int2str(j)];
%     fprintf(text)
%     fprintf(': %f\n', acc2)
%     acc_for_all(j) = acc2;
    
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

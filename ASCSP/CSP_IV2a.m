% CSP for bci III 4a
acc_csp = zeros(1,9);
csp_per_class = 3;
for i = 1:9 %loop through the 5 subjects
    name1 = ['../BCICIV_2a_gdf/A0',int2str(i),'T.gdf'];
    [s_train,h_train] = sload(name1);
    [X1_s, X2_s] = extract_bciIV2a_train(s_train, h_train);
    name2 = ['../BCICIV_2a_gdf/true_labels/A0',int2str(i),'E.mat'];
    name3 = ['../BCICIV_2a_gdf/A0',int2str(i),'E.gdf'];
    test_label = load(name2).classlabel;
    [s_test,h_test] = sload(name3);
    [X_test_t, y_test_t] = extract_bciIV2a_test(s_test, h_test, test_label);
        
    % number of trials in one class
    num_tmp = size(X1_s,1);
    
    data_source = cell(1,2); %1 left 2 right
    for idx = 1:num_tmp
        % left data
        data_source{1}{idx} = squeeze(X1_s(idx,:,:));
    end
    
    for idx = 1:num_tmp
        % left data
        data_source{2}{idx} = squeeze(X2_s(idx,:,:));
    end
    
    [~, cov_s1, cov_s2] = CSP(X1_s, X2_s);
    
    gm = cell(1,2); %cm 1 left 2 right
    gm{1} = cov_s1;
    gm{2} = cov_s2;
    
    num_tmp2 = size(X_test_t,1);
    data_target = cell(1,1);
    for idx = 1:num_tmp2
        
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
    %         data_target_log{2} = log_norm_BP(data_target_filter{2});
    
    %% train LDA. This is ASCSP without subspace alignment
    trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
    [W B class_means] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);
    [X_LDA, predicted_y_class1] = lda_apply(cell2mat(data_target_log{1})', W, B);
    predicted_y_class1(predicted_y_class1 == 1) = 2;   % incorrect choice
    predicted_y_class1(predicted_y_class1 == -1) = 1;
    
    test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1);
    
    %         [X_LDA predicted_y_class2] = lda_apply(cell2mat(data_target_log{2})', W, B);    % there is a vector output! should all be -1!
    %         predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
    %         predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
    %         temp = [predicted_y_class1; predicted_y_class2];
    %         acc1 = sum(temp)/length(temp);   % this is the percent correct classification
    acc1 = 1 - test_err;
    disp(['Acc: ' num2str(acc1)])
    acc_csp(i) = acc1;
    
end
% CSP for bci III 4a
acc_csp = zeros(5,1);
csp_per_class = 3;
sub_names = {'a','l','v','w','y'};
for i = 1:5 %loop through the 5 subjects
    name1 = ['../data/data_set_IVa_a',sub_names{i},'.mat'];
    name2 = ['../data/true_labels_a',sub_names{i},'.mat'];
    load(name1)
    load(name2)
    [X1_s, X2_s, X_test_s, y_test_s] = extract_data(mrk, cnt, test_idx, true_y);
    
    disp(size(find(y_test_s==1), 1))
    disp(size(find(y_test_s==2), 1))
    disp(size(X1_s,1))
    disp(size(X2_s,1))
    
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
    lambda1 = cal_shrinkage_modified(X1_s);
    lambda2 = cal_shrinkage_modified(X2_s);
    channels = size(X1_s, 2);
    gm = cell(1,2); %cm 1 left 2 right 
    alpha1 = mean(diag(cov_s1));
    alpha2 = mean(diag(cov_s2));
    gm{1} = lambda1*alpha1*eye(channels) + (1-lambda1)*cov_s1;
    gm{2} = lambda2*alpha2*eye(channels) + (1-lambda2)*cov_s2;
%     gm{1} = cov_s1;
%     gm{2} = cov_s2;
    
    num_tmp2 = size(X_test_s,1);
    data_target = cell(1,1);
    for idx = 1:num_tmp2
        
        data_target{1}{idx} = squeeze(X_test_s(idx,:,:));
        
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
    
    test_err = sum(abs(y_test_s - predicted_y_class1))/size(y_test_s, 1);
    
    %         [X_LDA predicted_y_class2] = lda_apply(cell2mat(data_target_log{2})', W, B);    % there is a vector output! should all be -1!
    %         predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
    %         predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
    %         temp = [predicted_y_class1; predicted_y_class2];
    %         acc1 = sum(temp)/length(temp);   % this is the percent correct classification
    acc1 = 1 - test_err;
    disp(['Acc: ' num2str(acc1)])
    acc_csp(i) = acc1;
    
end
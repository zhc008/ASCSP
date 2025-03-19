% CSP_SA.m
sub_names = {'a','l','v','w','y'};
acc_for_all = zeros(5,1);

csp_per_class = 3;

gms = cell(1,15);
data_sources = cell(1,15);
gms2 = cell(1,5);
data_sources2 = cell(1,5);

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
    
    num_split1 = floor(num_tmp1/3);
    num_split2 = floor(num_tmp2/3);
    for l = 1:3
        if l == 3
            X1_tmp = X1_s((l-1)*num_split1+1:end,:,:);
            X2_tmp = X2_s((l-1)*num_split2+1:end,:,:);
        else
            X1_tmp = X1_s((l-1)*num_split1+1:l*num_split1,:,:);
            X2_tmp = X2_s((l-1)*num_split2+1:l*num_split2,:,:);
        end
        data_source = cell(1,2); %1 left 2 right
        for idx = 1:size(X1_tmp,1)
            % left data
            data_source{1}{idx} = squeeze(X1_tmp(idx,:,:));
        end
        
        for idx = 1:size(X2_tmp,1)
            % left data
            data_source{2}{idx} = squeeze(X2_tmp(idx,:,:));
        end
        [~, cov_s1, cov_s2] = CSP(X1_tmp, X2_tmp);
        
        gm = cell(1,2); %cm 1 left 2 right
        gm{1} = cov_s1;
        gm{2} = cov_s2;
        gms{3*(i-1)+l} = gm;
        data_sources{3*(i-1)+l} = data_source;
    end
end

% for i = 1:5
%     name1 = ['../data/data_set_IVa_a',sub_names{i},'.mat'];
%     name2 = ['../data/true_labels_a',sub_names{i},'.mat'];
%     load(name1)
%     load(name2)
%     [X1_s, X2_s, X_test_s, y_test_s] = extract_data(mrk, cnt, test_idx, true_y);
%     % get the CSP filters and the covariance matrix for left hand and right
%     % hand data
%     % X1_s and X2_s are matrices with trials * channels * time points
%     channels = size(X1_s, 2);
%     % number of trials in one class
%     num_tmp1 = size(X1_s,1);
%     num_tmp2 = size(X2_s,1);
%     
%     data_source = cell(1,2); %1 left 2 right
%     for idx = 1:num_tmp1
%         % left data
%         data_source{1}{idx} = squeeze(X1_s(idx,:,:));
%     end
%     
%     for idx = 1:num_tmp2
%         % left data
%         data_source{2}{idx} = squeeze(X2_s(idx,:,:));
%     end
%     [~, cov_s1, cov_s2] = CSP(X1_s, X2_s);
%        
%     gm = cell(1,2); %cm 1 left 2 right 
%     gm{1} = cov_s1;
%     gm{2} = cov_s2;
%     gms2{i} = gm;
%     data_sources2{i} = data_source;
% end

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
        
    predicted_ys = zeros(num_tmp,1);
    denominator = 0;
    
    new_source = cell(1,2);
    new_gm = cell(1,2);
    sigma1 = zeros(channels, channels);
    sigma2 = zeros(channels, channels);
    new_gm{1} = sigma1;
    new_gm{2} = sigma2;
    for i = [1:(j-1)*3 j*3+1:15]
%         %% training CSP using new covariance matrix. data_source is useless
%         [ csp_coeff,all_coeff] = csp_analysis(data_sources{i},9,csp_per_class, 0,gms{i});
%         [ data_source_filter ] = csp_filtering(data_sources{i}, csp_coeff);
% 
%         data_source_log{1} = log_norm_BP(data_source_filter{1}); 
%         data_source_log{2} = log_norm_BP(data_source_filter{2});
% 
% 
%         %% Apply CSP in target subject
%         [ data_target_filter ] = csp_filtering(data_target, csp_coeff);
%         data_target_log{1} = log_norm_BP(data_target_filter{1});
% %         data_target_log{2} = log_norm_BP(data_target_filter{2});
%         
%         %% Train LDA. ASCSP with subspace alignment
%         trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
%         size_xproj = size(data_source_log{1},2) + size(data_source_log{2},2);
%         [W, B, ~] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);
%         
%         [X_LDA1, predicted_y_class] = lda_apply([cell2mat(data_target_log{1})'], W, B);
%         X_LDAs = cell(1,5);
%         
%         estimate_performance = zeros(size(y_test_t,1),1);
%         for k = 1:5
%             if (k == i||k == j)
%                 continue;
%             end
%             %% training CSP using new covariance matrix. data_source is useless
%             [ csp_coeff2,all_coeff2] = csp_analysis(data_sources2{k},9,csp_per_class, 0,gms2{k});
%             [ data_source_filter2 ] = csp_filtering(data_sources2{k}, csp_coeff2);
%             
%             data_source_log2{1} = log_norm_BP(data_source_filter2{1});
%             data_source_log2{2} = log_norm_BP(data_source_filter2{2});
%             
%             
%             %% Apply CSP in target subject
%             [ data_target_filter2 ] = csp_filtering(data_target, csp_coeff2);
%             data_target_log2{1} = log_norm_BP(data_target_filter2{1});
% %             data_target_log2{2} = log_norm_BP(data_target_filter2{2});
%             
%             %% Train LDA. ASCSP with subspace alignment
%             trainY2 = [(-1)*ones(size(data_source_log2{1},2),1); ones(size(data_source_log2{2},2),1)];
%             size_xproj2 = size(data_source_log2{1},2) + size(data_source_log2{2},2);
%             [W2, B2, ~] = lda_train_reg([cell2mat(data_source_log2{1})';cell2mat(data_source_log2{2})'], trainY2, 0);
% 
%             [X_LDA, predicted_y_class2] = lda_apply([cell2mat(data_target_log2{1})'], W2, B2);
%             X_LDAs{k} = X_LDA;
%             estimate_performance = estimate_performance + predicted_y_class2;
%         end
%         for k = 1:size(y_test_t,1)
%             if estimate_performance(k) == 0
%                 for kk = 1:5
%                     XLDAk = X_LDAs{kk};
%                     if size(XLDAk) == 0
%                         continue;
%                     end
%                     estimate_performance(k) = estimate_performance(k) + XLDAk(k);
%                 end
%             end
%         end
%         estimate_performance = sign(estimate_performance);
%         
%         err2 = sum(abs(estimate_performance - predicted_y_class))/2/size(data_target_log{1},2);
%         tmp_acc = 1-err2/2;
%         if tmp_acc <= 0.5
%             disp('smaller than 0.5')
%             continue;
%         end
        tmp_source = data_sources{i};
        tmp_gm = gms{i};
        for ii = 1:2
            new_gm{ii} = new_gm{ii}*size(new_source{ii},2)+tmp_gm{ii}*size(tmp_source{ii},2);
            new_source{ii} = [new_source{ii},tmp_source{ii}];
            new_gm{ii} = new_gm{ii}/size(new_source{ii},2);
        end
    end
        
    %% training CSP using new covariance matrix. data_source is useless
    [ csp_coeff3,all_coeff3] = csp_analysis(new_source,9,csp_per_class, 0,new_gm);
    [ data_source_filter3 ] = csp_filtering(new_source, csp_coeff3);
    
    data_source_log3{1} = log_norm_BP(data_source_filter3{1});
    data_source_log3{2} = log_norm_BP(data_source_filter3{2});
    
    
    %% Apply CSP in target subject
    [ data_target_filter3 ] = csp_filtering(data_target, csp_coeff3);
    data_target_log3{1} = log_norm_BP(data_target_filter3{1});
%     data_target_log3{2} = log_norm_BP(data_target_filter3{2});
        
    %% Train LDA. ASCSP with subspace alignment
    trainY3 = [(-1)*ones(size(data_source_log3{1},2),1); ones(size(data_source_log3{2},2),1)];
    size_xproj = size(data_source_log3{1},2) + size(data_source_log3{2},2);
    [W3, B3, ~] = lda_train_reg([cell2mat(data_source_log3{1})';cell2mat(data_source_log3{2})'], trainY3, 0);
    
    [~, predicted_y_class3] = lda_apply(cell2mat(data_target_log3{1})', W3, B3);
    predicted_y_class3(predicted_y_class3 == 1) = 2;   % incorrect choice
    predicted_y_class3(predicted_y_class3 == -1) = 1;
    test_err3 = sum(abs(y_test_t - predicted_y_class3))/size(y_test_t, 1);
    acc3 = 1 - test_err3;
    text = ['s', int2str(j)];
    fprintf(text)
    fprintf(': %f\n', acc3)
    acc_for_all(j) = acc3;
end

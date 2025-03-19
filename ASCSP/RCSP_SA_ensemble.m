% CSP_SA.m
clear
sub_names = {'a','l','v','w','y'};
acc_for_all = zeros(5,1);
acc_csp = zeros(5,5);

csp_per_class = 3;

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
    
%     X2_s = X2_s(1:26,:,:);
%     X1_s = X1_s(1:26,:,:);

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
    gms{i} = gm;
    data_sources{i} = data_source;
end
%     c = alpha(best_j); 
%     gm{1} = cov_s1 + c*eye(size(X1_s,2));
%     gm{2} = cov_s2 + c*eye(size(X1_s,2));
for j = 1:5
%     if j == i
%         continue; 
%     end
    name1_t = ['../data/data_set_IVa_a',sub_names{j},'.mat'];
    name2_t = ['../data/true_labels_a',sub_names{j},'.mat'];
    load(name1_t)
    load(name2_t)
    [X1_t, X2_t, X_test_t, y_test_t] = extract_data(mrk, cnt, test_idx, true_y);
%         if size(X1_t,1) < size(X2_t,1)
%             X2_t = X2_t(1:size(X1_t,1),:,:);
%         end
%         if size(X1_t,1) > size(X2_t,1)
%             X1_t = X1_t(1:size(X2_t,1),:,:);
%         end
        
        % number of trials in one class
%         num_tmp = size(X1_t,1);
    num_tmp = size(X_test_t,1);
    data_target = cell(1,1);
    for idx = 1:num_tmp
            
        data_target{1}{idx} = squeeze(X_test_t(idx,:,:));
            
    end
        
%         for idx = 1:num_tmp
%             
%             data_target{2}{idx} = squeeze(X2_t(idx,:,:));
%             
%         end
    
    predicted_ys = zeros(num_tmp,1);
    denominator = 0;
    
    for i = [1:j-1 j+1:5]
        %% training CSP using new covariance matrix. data_source is useless
        [ csp_coeff,all_coeff] = csp_analysis(data_sources{i},9,csp_per_class, 0,gms{i});
        [ data_source_filter ] = csp_filtering(data_sources{i}, csp_coeff);

        data_source_log{1} = log_norm_BP(data_source_filter{1}); 
        data_source_log{2} = log_norm_BP(data_source_filter{2});


        %% Apply CSP in target subject
        [ data_target_filter ] = csp_filtering(data_target, csp_coeff);
        data_target_log{1} = log_norm_BP(data_target_filter{1});
        %           data_target_log{2} = log_norm_BP(data_target_filter{2});
        
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
            for idx2 = 1:size(data_target_log{1},2)
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
        
        
        left_size = size(Xproj,1)-size_xproj;
        assert(mod(left_size,2) == 0, "Size mismatches");
        [X_LDA, predicted_y_class] = lda_apply(Xproj(size_xproj+1:end,:), W, B);
        X_LDAs = cell(1,5);
        
        estimate_performance = zeros(num_tmp,1);
        for k = 1:5
            if (k == i||k == j)
                continue;
            end
            %% training CSP using new covariance matrix. data_source is useless
            [ csp_coeff2,all_coeff2] = csp_analysis(data_sources{k},9,csp_per_class, 0,gms{i});
            [ data_source_filter2 ] = csp_filtering(data_sources{k}, csp_coeff2);
            
            data_source_log2{1} = log_norm_BP(data_source_filter2{1});
            data_source_log2{2} = log_norm_BP(data_source_filter2{2});
            
            
            %% Apply CSP in target subject
            [ data_target_filter2 ] = csp_filtering(data_target, csp_coeff2);
            data_target_log2{1} = log_norm_BP(data_target_filter2{1});
%             data_target_log2{2} = log_norm_BP(data_target_filter2{2});
            
            %% Subapace alignment
            X2 = zeros(0,6);
            Y2 = zeros(0,1);
            maLabeled2 =false(0,1);
            for idx = 1:2
                for idx2 = 1:size(data_source_log2{idx},2)
                    X2 = [X2;data_source_log2{idx}{idx2}'];
                    Y2 = [Y2;idx];
                    maLabeled2 = [maLabeled2;true];
                end
            end
            
            for idx = 1
                for idx2 = 1:size(data_target_log2{idx},2)
                    X2 = [X2;data_target_log2{idx}{idx2}'];
                    Y2 = [Y2;idx];
                    maLabeled2 = [maLabeled2;false];
                end
            end
            param = []; param.pcaCoef = 2;
            [Xproj2,transMdl] = ftTrans_sa(X2,maLabeled2,Y2(maLabeled2),maLabeled2,param);
            
            %% Train LDA. ASCSP with subspace alignment
            trainY2 = [(-1)*ones(size(data_source_log2{1},2),1); ones(size(data_source_log2{2},2),1)];
            size_xproj2 = size(data_source_log2{1},2) + size(data_source_log2{2},2);
            [W2,B2,~] = lda_train_reg(Xproj2(1:size_xproj2,:), trainY2, 0);
            
            
            left_size2 = size(Xproj2,1)-size_xproj2;
            assert(mod(left_size2,2) == 0, "Size mismatches");
            [X_LDA, predicted_y_class2] = lda_apply(Xproj2(size_xproj2+1:end,:), W2, B2);
            X_LDAs{k} = X_LDA;
            estimate_performance = estimate_performance + predicted_y_class2;
        end
        for k = 1:num_tmp
            if estimate_performance(k) == 0
                for kk = 1:5
                    XLDAk = X_LDAs{kk};
                    if size(XLDAk) == 0
                        continue;
                    end
                    estimate_performance(k) = estimate_performance(k) + XLDAk(k);
                end
            end
        end
        estimate_performance = sign(estimate_performance);
        err2 = sum(abs(estimate_performance - predicted_y_class))/2/size(data_target_log{1},2);
        tmp_acc = 1-err2/2;
%         if tmp_acc <= 0.5
%             disp('smaller than 0.5')
%             continue;
%         end
        predicted_ys = predicted_ys + tmp_acc * predicted_y_class;
        denominator = denominator + tmp_acc;
    end
    predicted_ys = predicted_ys/denominator;
    predicted_y_class1 = sign(predicted_ys);
    predicted_y_class1(predicted_y_class1 == 1) = 2;   % incorrect choice
    predicted_y_class1(predicted_y_class1 == -1) = 1;
    %       predicted_y_class1(predicted_y_class1 == 0) = randi([1,2]);
    test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1);
    %           [X_LDA predicted_y_class2] = lda_apply(Xproj(size_xproj+left_size/2+1:end,:), W, B);    % there is a vector output! should all be -1!
    %           predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
    %           predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
    %           temp = [predicted_y_class1; predicted_y_class2];
    %           acc2 = sum(temp)/length(temp);   % this is the percent correct classification
    acc2 = 1 - test_err;
    text = [sub_names{j}];
    fprintf(text)
    fprintf(': %f\n', acc2)
    acc_for_all(j) = acc2;
end

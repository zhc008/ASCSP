% CSP_SA.m
sub_names = {'a','l','v','w','y'};
acc_for_all = zeros(5,5);
acc_csp = zeros(5,5);

csp_per_class = 3;

alpha = [0.01, 0.005, 0.001, 0.0005];

for i = 1:5
    name1 = ['../data/data_set_IVa_a',sub_names{i},'.mat'];
    name2 = ['../data/true_labels_a',sub_names{i},'.mat'];
    load(name1)
    load(name2)
    [X1_s, X2_s, X_test_s, y_test_s] = extract_data(mrk, cnt, test_idx, true_y);
    % get the CSP filters and the covariance matrix for left hand and right
    % hand data
    if size(X1_s,1) < size(X2_s,1)
        X2_s = X2_s(1:size(X1_s,1),:,:);
    end
    if size(X1_s,1) > size(X2_s,1)
        X1_s = X1_s(1:size(X2_s,1),:,:);
    end
%     X2_s = X2_s(1:26,:,:);
%     X1_s = X1_s(1:26,:,:);
    best_j = 1;
    min_error = 1;
    for j = 1:4
        fold = 5;
        split_1 = floor(size(X1_s,1)/fold);
        split_2 = floor(size(X2_s,1)/fold);
        X1_cv = X1_s(1:fold*split_1,:,:);
        X2_cv = X2_s(1:fold*split_2,:,:);
        rng('default')
        shuffle_idx1 = randperm(size(X1_cv,1));
        shuffle_idx2 = randperm(size(X2_cv,1));
        X1_cv = X1_cv(shuffle_idx1,:,:);
        X2_cv = X2_cv(shuffle_idx2,:,:);
        X1_idx = 1:size(X1_cv,1);
        X2_idx = 1:size(X2_cv,1);

        X1_idx = reshape(X1_idx, fold, split_1);
        X2_idx = reshape(X2_idx, fold, split_2);
        valid_error = zeros(fold, 1);
        for k = 1:fold
            valid_1 = X1_idx(k, :);
            valid_2 = X2_idx(k, :);
            buffer = 1:fold;
            trainIdx_1 = reshape(X1_idx(buffer~=k,:).',1,[]);
            trainIdx_2 = reshape(X2_idx(buffer~=k,:).',1,[]);
            X1_t = X1_cv(trainIdx_1,:,:);
            X1_v = X1_cv(valid_1,:,:);
            X2_t = X2_cv(trainIdx_2,:,:);
            X2_v = X2_cv(valid_2,:,:);
            [W_tr,~,~] = TRCSP(alpha(j), X1_t, X2_t);
%            [W_tr,~,~] = CSP(X1_t, X2_t);
            f_X1_t = features(W_tr, X1_t);
            f_X2_t = features(W_tr, X2_t);
            f_X1_v = features(W_tr, X1_v);
            f_X2_v = features(W_tr, X2_v);
            f_X_v = [f_X1_v;f_X2_v];
            y_v = ones(size(f_X1_v, 1)+size(f_X2_v, 1),1);
            y_v((size(f_X1_v,1)+1):end) = y_v((size(f_X1_v,1)+1):end)*2;
            classified_y_test_cv = simpleLDA(f_X1_t, f_X2_t, f_X_v);
            valid_error(k) = sum(abs(y_v - classified_y_test_cv))/size(y_v, 1);
        end
        cv_error = min(valid_error);
%        disp(cv_error)
        if cv_error < min_error
            min_error = cv_error;
            best_j = j;
        end
    end


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
%     gm{1} = cov_s1;
%     gm{2} = cov_s2;
    c = alpha(best_j); 
    gm{1} = cov_s1 + c*eye(size(X1_s,2));
    gm{2} = cov_s2 + c*eye(size(X1_s,2));
    for j = 1:5
        if j == i
           continue; 
        end
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
        acc_csp(i,j) = acc1;
        
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
        [W,B,class_means] = lda_train_reg(Xproj(1:size_xproj,:), trainY, 0);
        
        
        left_size = size(Xproj,1)-size_xproj;
        assert(mod(left_size,2) == 0, "Size mismatches");
        
        [X_LDA predicted_y_class1] = lda_apply(Xproj(size_xproj+1:end,:), W, B);
        predicted_y_class1(predicted_y_class1 == 1) = 2;   % incorrect choice
        predicted_y_class1(predicted_y_class1 == -1) = 1;
        test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1);
%         [X_LDA predicted_y_class2] = lda_apply(Xproj(size_xproj+left_size/2+1:end,:), W, B);    % there is a vector output! should all be -1! 
%         predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
%         predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
%         temp = [predicted_y_class1; predicted_y_class2];
%         acc2 = sum(temp)/length(temp);   % this is the percent correct classification 
        acc2 = 1 - test_err;
        text = [sub_names{i},' to ',sub_names{j}];
        fprintf(text)
        fprintf(': %f\n', acc2)
        acc_for_all(i,j) = acc2;
    end
end

% CSP_SA.m
sub_names = {'a','l','v','w','y'};
acc_for_all = zeros(5,5);
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
    if size(X1_s,1) < size(X2_s,1)
        X2_s = X2_s(1:size(X1_s,1),:,:);
    end
    if size(X1_s,1) > size(X2_s,1)
        X1_s = X1_s(1:size(X2_s,1),:,:);
    end
%     X2_s = X2_s(1:26,:,:);
%     X1_s = X1_s(1:26,:,:);

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
    
    
    predicted_y_classes = cell(1,5);
    X_LDAs = cell(1,5);
    for k = 1:12
        X_LDAs{k} = zeros(num_tmp,1);
    end
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
        X_LDAs{i} = X_LDA;
        predicted_y_classes{i} = predicted_y_class;
    end
    
    denominator = 0;
    predicted_ys = zeros(num_tmp,1);
    for i = [1:j-1 j+1:5]
        estimate_performance = zeros(num_tmp,1);
        for k = 1:5
            if (k == i||k == j)
                continue;
            end
            estimate_performance = estimate_performance + predicted_y_classes{k};
        end
        for k = 1:num_tmp
            if estimate_performance(k) == 0
                for kk = 1:5
                    if kk == i
                        continue;
                    end
                    XLDAk = XLDAs{kk};
                    estimate_performance(k) = estimate_performance(k) + XLDAk(k);
                end
            end
        end
        estimate_performance = sign(estimate_performance);
        err2 = sum(abs(estimate_performance - predicted_y_classes{i}))/2/num_tmp;
        tmp_acc = 1-err2/2;
        predicted_ys = predicted_ys + tmp_acc * predicted_y_classes{i};
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
    acc_for_all(i,j) = acc2;
end

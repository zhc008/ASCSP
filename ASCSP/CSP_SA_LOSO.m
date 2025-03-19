% CSP_SA_LOSO.m
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
    
    predicted_ys = zeros(num_tmp,1);
        
    for i = [1:j-1 j+1:5]
        main_source = data_sources{i};
        main_size = size(main_source{1},2);
        for k = 1:5
            if (k == i||k == j)
                continue;
            end
            tmp_source = data_sources{k};
            tmp_size = size(tmp_source{1},2);
            combine_size = main_size + tmp_size;
            for g = 1:tmp_size
                for gg = 1:2
                    main_source{gg}{g+main_size} = tmp_source{gg}{g};
                end
            end
            main_size = size(main_source{1},2);
        end
        combine_cm = cell(1,2);
        channels = size(main_source{1}{1},1);
        sigma1 = zeros(channels,channels);
        sigma2 = zeros(channels,channels);
        combine_cm{1} = sigma1;
        combine_cm{2} = sigma2;
        for gg = 1:combine_size
            for ggg = 1:2
                tmp_train = main_source{ggg}{gg};
                cov_tmp = cov(tmp_train.');
                combine_cm{ggg} = combine_cm{ggg}+cov_tmp/trace(cov_tmp);
            end
        end
        combine_cm{1} = combine_cm{1}/combine_size;
        combine_cm{2} = combine_cm{2}/combine_size;
        %% training CSP using new covariance matrix. data_source is useless
        [ csp_coeff,all_coeff] = csp_analysis(main_source,9,csp_per_class, 0,combine_cm);
        [ data_source_filter ] = csp_filtering(main_source, csp_coeff);

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
        
        predicted_y_class(predicted_y_class == 1) = 2;   % incorrect choice
        predicted_y_class(predicted_y_class == -1) = 1;
%       predicted_y_class1(predicted_y_class1 == 0) = randi([1,2]);
        test_err = sum(abs(y_test_t - predicted_y_class))/size(y_test_t, 1);
%           [X_LDA predicted_y_class2] = lda_apply(Xproj(size_xproj+left_size/2+1:end,:), W, B);    % there is a vector output! should all be -1! 
%           predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
%           predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
%           temp = [predicted_y_class1; predicted_y_class2];
%           acc2 = sum(temp)/length(temp);   % this is the percent correct classification 
        acc2 = 1 - test_err;
        text = [sub_names{i},' to ',sub_names{j}];
        fprintf(text)
        fprintf(': %f\n', acc2)
        acc_for_all(i,j) = acc2;
    end
end

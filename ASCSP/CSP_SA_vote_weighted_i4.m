
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
        
        estimate_performance = zeros(num_tmp,1);
        
        idx_list = [1,2,3,4,5];
        idx_list = idx_list(idx_list~=i);
        idx_list = idx_list(idx_list~=j);
        for iidx = 1:3
            k = idx_list(iidx);
            iiidx = iidx;
            if iiidx == 3
                iiidx = 0;
            end
            l = idx_list(iiidx+1);
            
            tmp_source = data_sources{l};
            tmp_size = size(tmp_source{1},2);
            main_source = data_sources{k};
            main_size = size(main_source{1},2);
            combine_size = main_size + tmp_size;
            for g = 1:tmp_size
                for gg = 1:2
                    main_source{gg}{g+main_size} = tmp_source{gg}{g};
                end
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
            %% cross validation
            rng('default')
            shuffle1 = randperm(combine_size);
            shuffle2 = randperm(combine_size);
            shuffled = cell(1,2);
            shuffled{1} = main_source{1}(shuffle1);
            shuffled{2} = main_source{2}(shuffle2);
            split_size = floor(combine_size/5);
            cv_acc = zeros(5,1);
            % 5-fold cross validation
            for g = 1:5
                X_valid = cell(1,2);
                if g == 5
                    X_valid{1} = shuffled{1}((g-1)*split_size+1:end);
                    X_valid{2} = shuffled{2}((g-1)*split_size+1:end);
                else
                    X_valid{1} = shuffled{1}((g-1)*split_size+1:g*split_size);
                    X_valid{2} = shuffled{2}((g-1)*split_size+1:g*split_size);
                end
                X_train = cell(1,2);
                if g == 1
                    X_train{1} = shuffled{1}(g*split_size+1:end);
                    X_train{2} = shuffled{2}(g*split_size+1:end);
                elseif g == 5
                    X_train{1} = shuffled{1}((g-2)*split_size+1:(g-1)*split_size);
                    X_train{2} = shuffled{2}((g-2)*split_size+1:(g-1)*split_size);
                else
                    X_train{1} = shuffled{1}([(g-2)*split_size+1:(g-1)*split_size g*split_size+1:end]);
                    X_train{2} = shuffled{2}([(g-2)*split_size+1:(g-1)*split_size g*split_size+1:end]);
                end
                % calculate training covariance matrix
                train_size = size(X_train{1},2);
                train_cm = cell(1,2);
                channels = size(X_train{1}{1},1);
                sigma1 = zeros(channels,channels);
                sigma2 = zeros(channels,channels);
                train_cm{1} = sigma1;
                train_cm{2} = sigma2;
                for gg = 1:train_size
                    for ggg = 1:2
                        tmp_train = X_train{ggg}{gg};
                        cov_tmp = cov(tmp_train.');
                        train_cm{ggg} = train_cm{ggg}+cov_tmp/trace(cov_tmp);
                    end
                end
                train_cm{1} = train_cm{1}/train_size;
                train_cm{2} = train_cm{2}/train_size;
                %% csp
                [ csp_coeff4,~] = csp_analysis(X_train,9,csp_per_class, 0,train_cm);
                [ data_source_filter4 ] = csp_filtering(X_train, csp_coeff4);
                
                data_source_log4{1} = log_norm_BP(data_source_filter4{1});
                data_source_log4{2} = log_norm_BP(data_source_filter4{2});
                
                [ data_target_filter4 ] = csp_filtering(X_valid, csp_coeff4);
                data_target_log4{1} = log_norm_BP(data_target_filter4{1});
                data_target_log4{2} = log_norm_BP(data_target_filter4{2});
                %% Subapace alignment
                X4 = zeros(0,6);
                Y4 = zeros(0,1);
                maLabeled4 =false(0,1);
                for idx = 1:2
                    for idx2 = 1:size(data_source_log4{idx},2)
                        X4 = [X4;data_source_log4{idx}{idx2}'];
                        Y4 = [Y4;idx];
                        maLabeled4 = [maLabeled4;true];
                    end
                end
                
                for idx = 1:2
                    for idx2 = 1:size(data_target_log4{idx},2)
                        X4 = [X4;data_target_log4{idx}{idx2}'];
                        Y4 = [Y4;idx];
                        maLabeled4 = [maLabeled4;false];
                    end
                end
                param = []; param.pcaCoef = 2;
                [Xproj4,~] = ftTrans_sa(X4,maLabeled4,Y4(maLabeled4),maLabeled4,param);
                
                %% Train LDA. ASCSP with subspace alignment
                trainY4 = [(-1)*ones(size(data_source_log4{1},2),1); ones(size(data_source_log4{2},2),1)];
                size_xproj4 = size(data_source_log4{1},2) + size(data_source_log4{2},2);
                [W4,B4,~] = lda_train_reg(Xproj4(1:size_xproj4,:), trainY4, 0);
                
                
                left_size4 = size(Xproj4,1)-size_xproj4;
                assert(mod(left_size4,2) == 0, "Size mismatches");
                [~, predicted_y_class4] = lda_apply(Xproj4(size_xproj4+1:end,:), W4, B4);
                testY4 = [(-1)*ones(size(data_target_log4{1},2),1); ones(size(data_target_log4{2},2),1)];
                err4 = sum(abs(testY4 - predicted_y_class4))/2/size(data_target_log4{1},2);
                tmp_acc4 = 1-err4/2;
                cv_acc(g) = tmp_acc4;
            end
            valid_acc = mean(cv_acc);
            
            %% training CSP using new covariance matrix. data_source is useless
            [ csp_coeff3,all_coeff3] = csp_analysis(main_source,9,csp_per_class, 0,combine_cm);
            [ data_source_filter3 ] = csp_filtering(main_source, csp_coeff3);
            
            data_source_log3{1} = log_norm_BP(data_source_filter3{1});
            data_source_log3{2} = log_norm_BP(data_source_filter3{2});
            
            
            %% Apply CSP in target subject
            [ data_target_filter3 ] = csp_filtering(data_target, csp_coeff3);
            data_target_log3{1} = log_norm_BP(data_target_filter3{1});
            %           data_target_log{2} = log_norm_BP(data_target_filter{2});
            
            %% Subapace alignment
            X3 = zeros(0,6);
            Y3 = zeros(0,1);
            maLabeled3 =false(0,1);
            for idx = 1:2
                for idx2 = 1:size(data_source_log3{idx},2)
                    X3 = [X3;data_source_log3{idx}{idx2}'];
                    Y3 = [Y3;idx];
                    maLabeled3 = [maLabeled3;true];
                end
            end
            
            for idx = 1
                for idx2 = 1:size(data_target_log3{1},2)
                    X3 = [X3;data_target_log3{idx}{idx2}'];
                    Y3 = [Y3;idx];
                    maLabeled3 = [maLabeled3;false];
                end
            end
            param = []; param.pcaCoef = 2;
            [Xproj3,transMdl] = ftTrans_sa(X3,maLabeled3,Y3(maLabeled3),maLabeled3,param);
            
            %% Train LDA. ASCSP with subspace alignment
            trainY3 = [(-1)*ones(size(data_source_log3{1},2),1); ones(size(data_source_log3{2},2),1)];
            size_xproj3 = size(data_source_log3{1},2) + size(data_source_log3{2},2);
            [W3,B3,~] = lda_train_reg(Xproj3(1:size_xproj3,:), trainY3, 0);
            
            
            left_size3 = size(Xproj3,1)-size_xproj3;
            assert(mod(left_size3,2) == 0, "Size mismatches");
            [~, predicted_y_class3] = lda_apply(Xproj3(size_xproj3+1:end,:), W3, B3);
            estimate_performance = estimate_performance + predicted_y_class3*valid_acc;
        end
        estimate_performance = sign(estimate_performance);
        err2 = sum(abs(estimate_performance - predicted_y_class))/2/size(data_target_log{1},2);
        tmp_acc = 1-err2/2;
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
    acc_for_all(i,j) = acc2;
end

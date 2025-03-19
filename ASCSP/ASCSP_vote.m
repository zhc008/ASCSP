% ASCSP_vote.m
sub_names = {'a','l','v','w','y'};
acc_for_all = zeros(5,5);
acc_csp = zeros(5,5);
left_accs = zeros(5,5);
right_accs = zeros(5,5);

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
  
        denominator = 1;
        for k = 1:5
            if (k == i||k == j)
                continue;
            end
            tmp_source = data_sources{k};
            tmp_size = size(tmp_source{1},2);
            main_source = data_sources{i};
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
                [ csp_coeff2,~] = csp_analysis(X_train,9,csp_per_class, 0,train_cm);
                [ data_source_filter2 ] = csp_filtering(X_train, csp_coeff2);

                data_source_log2{1} = log_norm_BP(data_source_filter2{1}); 
                data_source_log2{2} = log_norm_BP(data_source_filter2{2});
                
                [ data_target_filter2 ] = csp_filtering(X_valid, csp_coeff2);
                data_target_log2{1} = log_norm_BP(data_target_filter2{1}); 
                data_target_log2{2} = log_norm_BP(data_target_filter2{2});
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
                
                for idx = 1:2
                    for idx2 = 1:size(data_target_log2{idx},2)
                        X2 = [X2;data_target_log2{idx}{idx2}'];
                        Y2 = [Y2;idx];
                        maLabeled2 = [maLabeled2;false];
                    end
                end
                param = []; param.pcaCoef = 2;
                [Xproj2,~] = ftTrans_sa(X2,maLabeled2,Y2(maLabeled2),maLabeled2,param);
                
                %% Train LDA. ASCSP with subspace alignment
                trainY2 = [(-1)*ones(size(data_source_log2{1},2),1); ones(size(data_source_log2{2},2),1)];
                size_xproj2 = size(data_source_log2{1},2) + size(data_source_log2{2},2);
                [W2,B2,~] = lda_train_reg(Xproj2(1:size_xproj2,:), trainY2, 0);
                
                
                left_size2 = size(Xproj2,1)-size_xproj2;
                assert(mod(left_size2,2) == 0, "Size mismatches");
                [~, predicted_y_class2] = lda_apply(Xproj2(size_xproj2+1:end,:), W2, B2);
                testY2 = [(-1)*ones(size(data_target_log2{1},2),1); ones(size(data_target_log2{2},2),1)];
                err2 = sum(abs(testY2 - predicted_y_class2))/2/size(data_target_log2{1},2);
                tmp_acc = 1-err2/2;
                cv_acc(g) = tmp_acc;
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
            [Xproj3,~] = ftTrans_sa(X3,maLabeled3,Y3(maLabeled3),maLabeled3,param);

            %% Train LDA. ASCSP with subspace alignment
            trainY3 = [(-1)*ones(size(data_source_log3{1},2),1); ones(size(data_source_log3{2},2),1)];
            size_xproj3 = size(data_source_log3{1},2) + size(data_source_log3{2},2);
            [W3,B3,~] = lda_train_reg(Xproj3(1:size_xproj3,:), trainY3, 0);
        
        
            left_size3 = size(Xproj3,1)-size_xproj3;
            assert(mod(left_size3,2) == 0, "Size mismatches");
            [~, predicted_y_class3] = lda_apply(Xproj3(size_xproj3+1:end,:), W3, B3);
            predicted_ys = predicted_ys + predicted_y_class3*valid_acc;
            denominator = denominator + valid_acc;
        end
        
        % call update function
        [gm1,store_idx] = update_vote4(gms{i},data_sources{i},data_target,predicted_ys,denominator);
        y_test1 = y_test_t(store_idx(:,1));
        y_test2 = y_test_t(store_idx(:,2));
        %     num_err = (sum(abs(y_test1 - 1)) + sum(abs(y_test2 - 2)))/total/2;
        left_err = sum(abs(y_test1 - 1))/size(store_idx,1);
        right_err = sum(abs(y_test2 - 2))/size(store_idx,1);
        left_accs(i,j) = 1-left_err;
        right_accs(i,j) = 1-right_err;
        
        %% training CSP using new covariance matrix. data_source is useless
        [ csp_coeff,all_coeff] = csp_analysis(data_sources{i},9,csp_per_class, 0,gm1);
        [ data_source_filter ] = csp_filtering(data_sources{i}, csp_coeff);

        data_source_log{1} = log_norm_BP(data_source_filter{1}); 
        data_source_log{2} = log_norm_BP(data_source_filter{2});


        %% Apply CSP in target subject
        [ data_target_filter ] = csp_filtering(data_target, csp_coeff);
        data_target_log{1} = log_norm_BP(data_target_filter{1}); 

        %% train LDA. This is ASCSP without subspace alignment
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        [W4, B4, ~] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);
        [~, predicted_y_class4] = lda_apply(cell2mat(data_target_log{1})', W4, B4);
        predicted_y_class4(predicted_y_class4 == 1) = 2;   % incorrect choice
        predicted_y_class4(predicted_y_class4 == -1) = 1;
        
        test_err = sum(abs(y_test_t - predicted_y_class4))/size(y_test_t, 1);
        
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
        [Xproj,~] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);

        %% Train LDA. ASCSP with subspace alignment
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        size_xproj = size(data_source_log{1},2) + size(data_source_log{2},2);
        [W,B,class_means] = lda_train_reg(Xproj(1:size_xproj,:), trainY, 0);
        
        
        left_size = size(Xproj,1)-size_xproj;
        assert(mod(left_size,2) == 0, "Size mismatches");
        
%         y_test_t = [(-1)*ones(size(X1_t,1),1); (-1)*ones(size(X1_t,1),1)];
%         for idx = 1:size(X1_t,1)
%             y_test_t(idx*2) = 1;
%         end
        
        [X_LDA, predicted_y_class1] = lda_apply(Xproj(size_xproj+1:end,:), W, B);
        predicted_y_class1(predicted_y_class1 == 1) = 2;   % incorrect choice
        predicted_y_class1(predicted_y_class1 == -1) = 1;
        test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1);
%         [X_LDA predicted_y_class2] = lda_apply(Xproj(size_xproj+left_size/2+1:end,:), W, B);    % there is a vector output! should all be -1! 
%         predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
%         predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
%         temp = [predicted_y_class1; predicted_y_class2];
        acc2 = 1-test_err;   % this is the percent correct classification 
        text = [sub_names{i},' to ',sub_names{j}];
        fprintf(text)
        fprintf(': %f\n', acc2)
        acc_for_all(i,j) = acc2;
    end
end

% CSP_SA_vote.m
clear

% acc_for_all = zeros(12,10);
acc_csp = zeros(12,10);

csp_per_class = 3;

gms = cell(10,12);
data_sources = cell(10,12);

for i = 1:12
for instance = 1:10
    name1 = ['C:/Users/CZN/Documents/Balanced_mahta_data/S',num2str(i),'_I',num2str(instance),'_calib.mat'];
    load(name1)
    % get the CSP filters and the covariance matrix for left hand and right
    % hand data
    % number of trials * number of channels * number of time samples

    % number of trials in one class
    num_aL = size(aL_train,1);
    num_tL = size(tL_train,1);
    num_aR = size(aR_train,1);
    num_tR = size(tR_train,1);
    
    data_source = cell(1,2); %1 left 2 right
    channels = size(aL_train, 2);
    sigma1 = zeros(channels, channels);
    sigma2 = zeros(channels, channels);

    for idx = 1:num_aL
        % left data
        data_source{1}{idx} = squeeze(aL_train(idx,:,:));
        aL_tmp = squeeze(aL_train(idx,:,:));
        aL_cov = cov(aL_tmp.');
        sigma1 = sigma1 + aL_cov/trace(aL_cov);
        % right data
        data_source{2}{idx} = squeeze(aR_train(idx,:,:));
        aR_tmp = squeeze(aR_train(idx,:,:));
        aR_cov = cov(aR_tmp.');
        sigma2 = sigma2 + aR_cov/trace(aR_cov);
    end
    for idx = 1:num_tL
        % left data
        data_source{1}{num_aL+idx} = squeeze(tL_train(idx,:,:));
        tL_tmp = squeeze(tL_train(idx,:,:));
        tL_cov = cov(tL_tmp.');
        sigma1 = sigma1 + tL_cov/trace(tL_cov);
        % right data
        data_source{2}{num_aR+idx} = squeeze(tR_train(idx,:,:));
        tR_tmp = squeeze(tR_train(idx,:,:));
        tR_cov = cov(tR_tmp.');
        sigma2 = sigma2 + tR_cov/trace(tR_cov);
    end
    lambda1 = cal_shrinkage_modified([aL_train; tL_train]);
    lambda2 = cal_shrinkage_modified([aR_train; tR_train]);
    
    sigma1 = sigma1/(num_aL+num_tL);
    sigma2 = sigma2/(num_aR+num_tR);
    gm = cell(1,2); %cm 1 left 2 right 
%     gm{1} = lambda1*eye(channels) + (1-lambda1)*sigma1;
%     gm{2} = lambda2*eye(channels) + (1-lambda2)*sigma2;
    gm{1} = sigma1;
    gm{2} = sigma2;
    gms{instance,i} = gm;
    data_sources{instance,i} = data_source;
end
end

for j = 1:12
for instance = 1:10
    name2 = ['C:/Users/CZN/Documents/Balanced_mahta_data/S',num2str(j),'_I',num2str(instance),'_onln.mat'];
    load(name2)
    
    num_aL_ol = size(aL_test,1);
    num_tL_ol = size(tL_test,1);
    num_aR_ol = size(aR_test,1);
    num_tR_ol = size(tR_test,1);
    
    % store target data in data_target
    data_target = cell(1,2);
    
    for target_trial = 1:num_aL_ol
        % left trials
        data_target{1}{target_trial} = squeeze(aL_test(target_trial,:,:));
        % right trials
        data_target{2}{target_trial} = squeeze(aR_test(target_trial,:,:));
    end
    for target_trial = 1:num_tL_ol
        % left trials
        data_target{1}{target_trial+num_aL_ol} = squeeze(tL_test(target_trial,:,:));
        % right trials
        data_target{2}{target_trial+num_aR_ol} = squeeze(tR_test(target_trial,:,:));
    end
    y_test_t = [(-1)*ones(size(data_target{1},2),1); ones(size(data_target{2},2),1)];
    
    predicted_ys = zeros(size(y_test_t,1),1);
    denominator = 0;
    
    for i = [1:j-1 j+1:12]
%         if (i == 6||i == 10||i == 12)
%             continue;
%         end
        %% training CSP using new covariance matrix. data_source is useless
        [ csp_coeff,all_coeff] = csp_analysis(data_sources{instance,i},9,csp_per_class, 0,gms{instance,i});
        [ data_source_filter ] = csp_filtering(data_sources{instance,i}, csp_coeff);

        data_source_log{1} = log_norm_BP(data_source_filter{1}); 
        data_source_log{2} = log_norm_BP(data_source_filter{2});


        %% Apply CSP in target subject
        [ data_target_filter ] = csp_filtering(data_target, csp_coeff);
        data_target_log{1} = log_norm_BP(data_target_filter{1});
        data_target_log{2} = log_norm_BP(data_target_filter{2});
        
        %% Train LDA. ASCSP with subspace alignment
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        [W, B, class_means] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);
        [~, predicted_y_class] = lda_apply([cell2mat(data_target_log{1})';cell2mat(data_target_log{2})'], W, B);
        X_LDAs = cell(1,12);
        
        estimate_performance = zeros(size(y_test_t,1),1);
        for k = 1:12
            if (k == i||k == j)
                continue;
            end
%             if (k == 6||k == 10||k == 12)
%                 continue;
%             end
            %% training CSP using new covariance matrix. data_source is useless
            [ csp_coeff2,all_coeff2] = csp_analysis(data_sources{instance,k},9,csp_per_class, 0,gms{instance,i});
            [ data_source_filter2 ] = csp_filtering(data_sources{instance,k}, csp_coeff2);
            
            data_source_log2{1} = log_norm_BP(data_source_filter2{1});
            data_source_log2{2} = log_norm_BP(data_source_filter2{2});
            
            
            %% Apply CSP in target subject
            [ data_target_filter2 ] = csp_filtering(data_target, csp_coeff2);
            data_target_log2{1} = log_norm_BP(data_target_filter2{1});
            data_target_log2{2} = log_norm_BP(data_target_filter2{2});
            
            %% Train LDA. ASCSP with subspace alignment
            trainY2 = [(-1)*ones(size(data_source_log2{1},2),1); ones(size(data_source_log2{2},2),1)];
            size_xproj2 = size(data_source_log2{1},2) + size(data_source_log2{2},2);
            [W2,B2,~] = lda_train_reg([cell2mat(data_source_log2{1})';cell2mat(data_source_log2{2})'], trainY2, 0);

            [X_LDA, predicted_y_class2] = lda_apply([cell2mat(data_target_log2{1})';cell2mat(data_target_log2{2})'], W2, B2);
            X_LDAs{k} = X_LDA;
            estimate_performance = estimate_performance + predicted_y_class2;
            
        end
        for k = 1:size(y_test_t,1)
            if estimate_performance(k) == 0
                for kk = 1:12
                    XLDAk = X_LDAs{kk};
                    if size(XLDAk) == 0
                        continue;
                    end
                    estimate_performance(k) = estimate_performance(k) + XLDAk(k);
                end
            end
        end
        estimate_performance = sign(estimate_performance);
        err2 = sum(abs(estimate_performance - predicted_y_class))/size(predicted_y_class,1);
        tmp_acc = 1-err2/2;
%         if tmp_acc <= 0.5
%             disp('smaller than 0.5')
%             continue;
%         end
%         tmp_acc_all(i,j) = tmp_acc;
        predicted_ys = predicted_ys + (tmp_acc-0.5) * predicted_y_class;
        denominator = denominator + tmp_acc;
    end
    predicted_ys = predicted_ys/denominator;
    predicted_y_class1 = sign(predicted_ys);
    test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1)/2;

    acc2 = 1 - test_err;
    text = ['s', int2str(j)];
    fprintf(text)
    fprintf(': %f\n', acc2)
    acc_csp(j,instance) = acc2;
end
end
mean_csp = mean(acc_csp,2);
% mean_csp_sa = mean(acc_for_all,2);



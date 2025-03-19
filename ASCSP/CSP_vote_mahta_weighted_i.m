% CSP_SA.m
acc_for_all = zeros(12,1);

csp_per_class = 3;

gms = cell(1,12);
data_sources = cell(1,12);

for i = 1:12
    name1 = ['../Mahta_data/S',int2str(i),'_calib.mat'];
    load(name1)
    % get the CSP filters and the covariance matrix for left hand and right
    % hand data
    % number of channels * number of time samples * number of epochs

    % number of trials in one class
    num_aL = size(aL_calib,3);
    num_tL = size(tL_calib,3);
    num_aR = size(aR_calib,3);
    num_tR = size(tR_calib,3);
    
    data_source = cell(1,2); %1 left 2 right
    channels = size(aL_calib, 1);
    sigma1 = zeros(channels, channels);
    sigma2 = zeros(channels, channels);
    
    % left data
    for idx = 1:num_aL
        data_source{1}{idx} = squeeze(aL_calib(:,:,idx));
        aL_tmp = squeeze(aL_calib(:,:,idx));
        aL_cov = cov(aL_tmp.');
        sigma1 = sigma1 + aL_cov/trace(aL_cov);
    end
    for idx = 1:num_tL
        data_source{1}{num_aL+idx} = squeeze(tL_calib(:,:,idx));
        tL_tmp = squeeze(tL_calib(:,:,idx));
        tL_cov = cov(tL_tmp.');
        sigma1 = sigma1 + tL_cov/trace(tL_cov);
    end
    
    % right data
    for idx = 1:num_aR
        data_source{2}{idx} = squeeze(aR_calib(:,:,idx));
        aR_tmp = squeeze(aR_calib(:,:,idx));
        aR_cov = cov(aR_tmp.');
        sigma2 = sigma2 + aR_cov/trace(aR_cov);
    end
    for idx = 1:num_tR
        data_source{2}{num_aR+idx} = squeeze(tR_calib(:,:,idx));
        tR_tmp = squeeze(tR_calib(:,:,idx));
        tR_cov = cov(tR_tmp.');
        sigma2 = sigma2 + tR_cov/trace(tR_cov);
    end
    
    sigma1 = sigma1/(num_aL+num_tL);
    sigma2 = sigma2/(num_aR+num_tR);
    gm = cell(1,2); %cm 1 left 2 right 
    gm{2} = sigma2;
    gm{1} = sigma1;
    gms{i} = gm;
    data_sources{i} = data_source;
end

for j = 1:12
    name2 = ['../Mahta_data/S',int2str(j),'_online.mat'];
    load(name2)
    
    num_aL_ol = size(aL_onln,3);
    num_tL_ol = size(tL_onln,3);
    num_aR_ol = size(aR_onln,3);
    num_tR_ol = size(tR_onln,3);
    
    % store target data in data_target
    data_target = cell(1,2);
    
    % left trials
    for target_trial = 1:num_aL_ol
        data_target{1}{target_trial} = squeeze(aL_onln(:,:,target_trial));
    end
    for target_trial = 1:num_tL_ol
        data_target{1}{target_trial+num_aL_ol} = squeeze(tL_onln(:,:,target_trial));
    end
    
    % right trials
    for target_trial = 1:num_aR_ol
        data_target{2}{target_trial} = squeeze(aR_onln(:,:,target_trial));
    end
    for target_trial = 1:num_tR_ol
        data_target{2}{target_trial+num_aR_ol} = squeeze(tR_onln(:,:,target_trial));
    end
    y_test_t = [(-1)*ones(size(data_target{1},2),1); ones(size(data_target{2},2),1)];
    
    predicted_ys = zeros(size(y_test_t,1),1);
    denominator = 0;
    
    for i = [1:j-1 j+1:12]
        %% training CSP using new covariance matrix. data_source is useless
        [ csp_coeff,all_coeff] = csp_analysis(data_sources{i},9,csp_per_class, 0,gms{i});
        [ data_source_filter ] = csp_filtering(data_sources{i}, csp_coeff);

        data_source_log{1} = log_norm_BP(data_source_filter{1}); 
        data_source_log{2} = log_norm_BP(data_source_filter{2});


        %% Apply CSP in target subject
        [ data_target_filter ] = csp_filtering(data_target, csp_coeff);
        data_target_log{1} = log_norm_BP(data_target_filter{1});
        data_target_log{2} = log_norm_BP(data_target_filter{2});
        
        %% Train LDA. ASCSP with subspace alignment
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        size_xproj = size(data_source_log{1},2) + size(data_source_log{2},2);
        [W, B, ~] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);
        
        [X_LDA1, predicted_y_class] = lda_apply([cell2mat(data_target_log{1})';cell2mat(data_target_log{2})'], W, B);
        X_LDAs = cell(1,12);
        
        estimate_performance = zeros(size(y_test_t,1),1);
        for k = 1:12
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
            data_target_log2{2} = log_norm_BP(data_target_filter2{2});
            
            %% Train LDA. ASCSP with subspace alignment
            trainY2 = [(-1)*ones(size(data_source_log2{1},2),1); ones(size(data_source_log2{2},2),1)];
            size_xproj2 = size(data_source_log2{1},2) + size(data_source_log2{2},2);
            [W2, B2, ~] = lda_train_reg([cell2mat(data_source_log2{1})';cell2mat(data_source_log2{2})'], trainY2, 0);

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
        predicted_ys = predicted_ys + tmp_acc * predicted_y_class;
        denominator = denominator + tmp_acc;
    end
    predicted_ys = predicted_ys/denominator;
    predicted_y_class1 = sign(predicted_ys);
    test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1)/2;
    %           [X_LDA predicted_y_class2] = lda_apply(Xproj(size_xproj+left_size/2+1:end,:), W, B);    % there is a vector output! should all be -1!
    %           predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
    %           predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
    %           temp = [predicted_y_class1; predicted_y_class2];
    %           acc2 = sum(temp)/length(temp);   % this is the percent correct classification
    acc2 = 1 - test_err;
    text = ['s', int2str(j)];
    fprintf(text)
    fprintf(': %f\n', acc2)
    acc_for_all(j) = acc2;
end

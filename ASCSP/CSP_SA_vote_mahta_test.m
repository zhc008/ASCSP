% CSP_SA_vote using after-the-fact weighting for i
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
        
        for idx = 1:2
            for idx2 = 1:size(data_target_log{idx},2)
                X = [X;data_target_log{idx}{idx2}'];
                Y = [Y;idx];
                maLabeled = [maLabeled;false];
            end
        end
        param = []; param.pcaCoef = 6;
        [Xproj,transMdl] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);
        
        %% Train LDA. ASCSP with subspace alignment
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        size_xproj = size(data_source_log{1},2) + size(data_source_log{2},2);
        [W,B,~] = lda_train_reg(Xproj(1:size_xproj,:), trainY, 0);
        
        
%         left_size = size(Xproj,1)-size_xproj;
%         assert(mod(left_size,2) == 0, "Size mismatches");
        [X_LDA, predicted_y_class] = lda_apply(Xproj(size_xproj+1:end,:), W, B);
        
        predicted_y_class2 = predicted_y_class;
        test_err2 = sum(abs(y_test_t - predicted_y_class2))/size(y_test_t, 1)/2;
        weight = 1-test_err2;
%         if weight <= 0.5
%             disp('smaller than 0.5')
%             continue;
%         end
        predicted_ys = predicted_ys + weight * predicted_y_class;
        denominator = denominator + weight;
    end
    predicted_ys = predicted_ys/denominator;
    predicted_y_class1 = sign(predicted_ys);
    %       predicted_y_class1(predicted_y_class1 == 0) = randi([1,2]);
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

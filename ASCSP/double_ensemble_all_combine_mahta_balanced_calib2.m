% CSP_combine for balanced mahta's data
clear

acc_for_all = zeros(12,10);
% acc_csp = zeros(12,10);

csp_per_class = 3;

gms = cell(10,12);
data_sources = cell(10,12);
gms_r = cell(10,12);

for i = 1:12
for instance = 1:10
    name1 = ['C:/Users/CZN/Documents/Balanced_mahta_data/Calib/S',num2str(i),'_I',num2str(instance),'_calib_split.mat'];
    load(name1)
    % get the CSP filters and the covariance matrix for left hand and right
    % hand data
    % number of trials * number of channels * number of time samples

    % number of trials in one class
    num_aL = size(aL_tr,1);
    num_tL = size(tL_tr,1);
    num_aR = size(aR_tr,1);
    num_tR = size(tR_tr,1);
    
    data_source = cell(1,2); %1 left 2 right
    channels = size(aL_tr, 2);
    sigma1 = zeros(channels, channels);
    sigma2 = zeros(channels, channels);

    for idx = 1:num_aL
        % left data
        data_source{1}{idx} = squeeze(aL_tr(idx,:,:));
        aL_tmp = squeeze(aL_tr(idx,:,:));
        aL_cov = cov(aL_tmp.');
        sigma1 = sigma1 + aL_cov/trace(aL_cov);
        % right data
        data_source{2}{idx} = squeeze(aR_tr(idx,:,:));
        aR_tmp = squeeze(aR_tr(idx,:,:));
        aR_cov = cov(aR_tmp.');
        sigma2 = sigma2 + aR_cov/trace(aR_cov);
    end
    for idx = 1:num_tL
        % left data
        data_source{1}{num_aL+idx} = squeeze(tL_tr(idx,:,:));
        tL_tmp = squeeze(tL_tr(idx,:,:));
        tL_cov = cov(tL_tmp.');
        sigma1 = sigma1 + tL_cov/trace(tL_cov);
        % right data
        data_source{2}{num_aR+idx} = squeeze(tR_tr(idx,:,:));
        tR_tmp = squeeze(tR_tr(idx,:,:));
        tR_cov = cov(tR_tmp.');
        sigma2 = sigma2 + tR_cov/trace(tR_cov);
    end
    lambda1 = cal_shrinkage_modified([aL_tr; tL_tr]);
    lambda2 = cal_shrinkage_modified([aR_tr; tR_tr]);
    
    sigma1 = sigma1/(num_aL+num_tL);
    sigma2 = sigma2/(num_aR+num_tR);
    gm = cell(1,2); %cm 1 left 2 right 
    gm{1} = sigma1;
    gm{2} = sigma2;
    gms{instance,i} = gm;
    gm_r = cell(1,2);
    gm_r{1} = lambda1*eye(channels) + (1-lambda1)*sigma1;
    gm_r{2} = lambda2*eye(channels) + (1-lambda2)*sigma2;
    gms_r{instance,i} = gm_r;
    data_sources{instance,i} = data_source;
end
end

for j_instance = 1:10
for j = 1:12
    name1 = ['C:/Users/CZN/Documents/Balanced_mahta_data/Calib/S',num2str(j),'_I',num2str(j_instance),'_calib_split.mat'];
    load(name1)
    
    num_aL_ol = size(aL_te,1);
    num_tL_ol = size(tL_te,1);
    num_aR_ol = size(aR_te,1);
    num_tR_ol = size(tR_te,1);
    
    % store target data in data_target
    data_target = cell(1,2);
    
    for target_trial = 1:num_aL_ol
        % left trials
        data_target{1}{target_trial} = squeeze(aL_te(target_trial,:,:));
        % right trials
        data_target{2}{target_trial} = squeeze(aR_te(target_trial,:,:));
    end
    for target_trial = 1:num_tL_ol
        % left trials
        data_target{1}{target_trial+num_aL_ol} = squeeze(tL_te(target_trial,:,:));
        % right trials
        data_target{2}{target_trial+num_aR_ol} = squeeze(tR_te(target_trial,:,:));
    end
    y_test_t = [(-1)*ones(size(data_target{1},2),1); ones(size(data_target{2},2),1)];
    
%% CSP_combine_SA
    predicted_y_classes = cell(1,10);
    X_LDAs = zeros(size(y_test_t,1),10);
for instance = 1:10
    new_source = cell(1,2);
    new_gm = cell(1,2);
    sigma1 = zeros(channels, channels);
    sigma2 = zeros(channels, channels);
    new_gm{1} = sigma1;
    new_gm{2} = sigma2;
    for i = [1:j-1 j+1:12]
        tmp_source = data_sources{instance,i};
        tmp_gm = gms{instance,i};
        for ii = 1:2
            new_gm{ii} = new_gm{ii}*size(new_source{ii},2)+tmp_gm{ii}*size(tmp_source{ii},2);
            new_source{ii} = [new_source{ii},tmp_source{ii}];
            new_gm{ii} = new_gm{ii}/size(new_source{ii},2);
        end
    end
        
    %% training CSP using new covariance matrix. data_source is useless
    [ csp_coeff,all_coeff] = csp_analysis(new_source,9,csp_per_class, 0,new_gm);
    [ data_source_filter ] = csp_filtering(new_source, csp_coeff);
    
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
    
    [X_LDA, predicted_y_class] = lda_apply(Xproj(size_xproj+1:end,:), W, B);
    predicted_y_classes{instance} = predicted_y_class;
    X_LDAs(:,instance) = X_LDA;
end

    predicted_ys = zeros(size(y_test_t,1),1);

for instance = 1:10
    predicted_ys = predicted_ys + predicted_y_classes{instance};
end
for ii = size(y_test_t,1)
    if predicted_ys(ii) == 0
        predicted_ys(ii) = sum(X_LDAs(ii,:));
    end
end
    predicted_y_class1 = sign(predicted_ys);
    test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1)/2;
    acc1 = 1 - test_err;
    text = ['CSP_combine_SA s', int2str(j)];
    fprintf(text)
    fprintf(': %f\n', acc1);

%% RCSP_combine_SA
    predicted_y_classes = cell(1,10);
    X_LDAs = zeros(size(y_test_t,1),10);
for instance = 1:10
    new_source = cell(1,2);
    new_gm = cell(1,2);
    sigma1 = zeros(channels, channels);
    sigma2 = zeros(channels, channels);
    new_gm{1} = sigma1;
    new_gm{2} = sigma2;
    for i = [1:j-1 j+1:12]
        tmp_source = data_sources{instance,i};
        tmp_gm = gms_r{instance,i};
        for ii = 1:2
            new_gm{ii} = new_gm{ii}*size(new_source{ii},2)+tmp_gm{ii}*size(tmp_source{ii},2);
            new_source{ii} = [new_source{ii},tmp_source{ii}];
            new_gm{ii} = new_gm{ii}/size(new_source{ii},2);
        end
    end
        
    %% training CSP using new covariance matrix. data_source is useless
    [ csp_coeff,all_coeff] = csp_analysis(new_source,9,csp_per_class, 0,new_gm);
    [ data_source_filter ] = csp_filtering(new_source, csp_coeff);
    
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
    
    [X_LDA, predicted_y_class] = lda_apply(Xproj(size_xproj+1:end,:), W, B);
    predicted_y_classes{instance} = predicted_y_class;
    X_LDAs(:,instance) = X_LDA;
end

    predicted_ys = zeros(size(y_test_t,1),1);

for instance = 1:10
    predicted_ys = predicted_ys + predicted_y_classes{instance};
end
for ii = size(y_test_t,1)
    if predicted_ys(ii) == 0
        predicted_ys(ii) = sum(X_LDAs(ii,:));
    end
end
    predicted_y_class2 = sign(predicted_ys);
    test_err = sum(abs(y_test_t - predicted_y_class2))/size(y_test_t, 1)/2;
    acc2 = 1 - test_err;
    text = ['RCSP_combine_SA s', int2str(j)];
    fprintf(text)
    fprintf(': %f\n', acc2);
    
%% RCSP_SA_vote
    predicted_ys = zeros(size(y_test_t,1),1);
    denominator = 0;
    
    for i = [1:j-1 j+1:12]
        %% training CSP using new covariance matrix. data_source is useless
        [ csp_coeff,all_coeff] = csp_analysis(data_sources{j_instance,i},9,csp_per_class, 0,gms_r{j_instance,i});
        [ data_source_filter ] = csp_filtering(data_sources{j_instance,i}, csp_coeff);

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
        
        %% Train LDA. CSP with subspace alignment
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        size_xproj = size(data_source_log{1},2) + size(data_source_log{2},2);
        [W,B,~] = lda_train_reg(Xproj(1:size_xproj,:), trainY, 0);
        
        [X_LDA, predicted_y_class] = lda_apply(Xproj(size_xproj+1:end,:), W, B);
        X_LDAs = cell(1,12);
        
        estimate_performance = zeros(size(y_test_t,1),1);
        for k = 1:12
            if (k == j||k == i)
                continue;
            end
            %% training CSP using new covariance matrix. data_source is useless
            [ csp_coeff2,all_coeff2] = csp_analysis(data_sources{j_instance,k},9,csp_per_class, 0,gms_r{j_instance,i});
            [ data_source_filter2 ] = csp_filtering(data_sources{j_instance,k}, csp_coeff2);
            
            data_source_log2{1} = log_norm_BP(data_source_filter2{1});
            data_source_log2{2} = log_norm_BP(data_source_filter2{2});
            
            %% Apply CSP in target subject
            [ data_target_filter2 ] = csp_filtering(data_target, csp_coeff2);
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
            param = []; param.pcaCoef = 6;
            [Xproj2,transMdl] = ftTrans_sa(X2,maLabeled2,Y2(maLabeled2),maLabeled2,param);
            
            %% Train LDA. ASCSP with subspace alignment
            trainY2 = [(-1)*ones(size(data_source_log2{1},2),1); ones(size(data_source_log2{2},2),1)];
            size_xproj2 = size(data_source_log2{1},2) + size(data_source_log2{2},2);
            [W2,B2,~] = lda_train_reg(Xproj2(1:size_xproj2,:), trainY2, 0);
            
            [X_LDA, predicted_y_class4] = lda_apply(Xproj2(size_xproj2+1:end,:), W2, B2);
            X_LDAs{k} = X_LDA;
            estimate_performance = estimate_performance + predicted_y_class4;
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
        predicted_ys = predicted_ys + tmp_acc * predicted_y_class;
        denominator = denominator + tmp_acc;
    end
    predicted_ys = predicted_ys/denominator;
    predicted_y_class3 = sign(predicted_ys);
    test_err = sum(abs(y_test_t - predicted_y_class3))/size(y_test_t, 1)/2;
    acc3 = 1 - test_err;
    text = ['RCSP_SA_vote s', int2str(j)];
    fprintf(text)
    fprintf(': %f\n', acc3)
    
    predicted_y_vote = predicted_y_class1+predicted_y_class2+predicted_y_class3;
    predicted_y_vote = sign(predicted_y_vote);
    test_err = sum(abs(y_test_t - predicted_y_vote))/size(y_test_t, 1)/2;
    acc4 = 1 - test_err;
    text = ['s', int2str(j)];
    fprintf(text)
    fprintf(': %f\n', acc4)
    acc_for_all(j,j_instance) = acc4;

end
end
% mean_csp = mean(acc_csp,2);
mean_csp_sa = mean(acc_for_all,2);


% CSP_combine for balanced mahta's data
clear

acc_for_all = zeros(12,10);
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

for j_instance = 1:10
for j = 1:12
    name1 = ['C:/Users/CZN/Documents/Balanced_mahta_data/S',num2str(j),'_I',num2str(j_instance),'_onln.mat'];
    load(name1)
    
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
    
    %% train LDA. This is ASCSP without subspace alignment
    trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
    [W, B, class_means] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);
    [~, predicted_y_class1] = lda_apply([cell2mat(data_target_log{1})';cell2mat(data_target_log{2})'], W, B);
    
    test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1)/2;
    acc1 = 1 - test_err;
    acc_csp(j,j_instance) = acc1;
    
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
%     denominator = 0;

for instance = 1:10
%     estimate_p = zeros(size(y_test_t,1),1);
%     for ii = [1:instance-1 instance+1:10]
%         estimate_p = estimate_p + predicted_y_classes{ii};
%     end
%     estimate_p = sign(estimate_p);
%     test_err = sum(abs(estimate_p - predicted_y_classes{instance}))/size(y_test_t, 1)/2;
%     tmp_acc = 1 - test_err;
%     predicted_ys = predicted_ys + tmp_acc * predicted_y_classes{instance};
%     denominator = denominator + tmp_acc;
    predicted_ys = predicted_ys + predicted_y_classes{instance};
end
for ii = 1:size(y_test_t,1)
    if predicted_ys(ii) == 0
        predicted_ys(ii) = sum(X_LDAs(ii,:));
    end
end
%     predicted_y_class1 = sign(predicted_ys/denominator);
    predicted_y_class1 = sign(predicted_ys);
    test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1)/2;
    acc2 = 1 - test_err;
    acc_for_all(j,j_instance) = acc2;
    text = ['s', int2str(j)];
    fprintf(text)
    fprintf(': %f\n', acc2);
end
end
mean_csp = mean(acc_csp,2);
mean_csp_sa = mean(acc_for_all,2);


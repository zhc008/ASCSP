% CSP_combine for balanced mahta's data
clear

acc_for_all = zeros(12,10);
acc_csp = zeros(12,10);

csp_per_class = 3;

gms = cell(10,12);
data_sources = cell(10,12);

combined_train = cell(10,12);
for instance = 1:10
    S = cell(1,12);
    s = cell(1,4);
    for j = 1:12
        name1 = ['C:/Users/CZN/Documents/Balanced_mahta_data/S',num2str(j),'_I',num2str(instance),'_calib.mat'];
        load(name1)
        s{1} = aL_train;
        s{2} = tL_train;
        s{3} = aR_train;
        s{4} = tR_train;
        S{j} = s;
    end
for i = 1:12
    combine_aL=[];
    combine_tL=[];
    combine_aR=[];
    combine_tR=[];
    
    for j = [1:i-1 i+1:12]
        tmp_s = S{j};
        combine_aL = [combine_aL;tmp_s{1}];
        combine_tL = [combine_tL;tmp_s{2}];
        combine_aR = [combine_aR;tmp_s{3}];
        combine_tR = [combine_tR;tmp_s{4}];
    end
    % get the CSP filters and the covariance matrix for left hand and right
    % hand data
    % number of trials * number of channels * number of time samples

    % number of trials in one class
    num_aL = size(combine_aL,1);
    num_tL = size(combine_tL,1);
    num_aR = size(combine_aR,1);
    num_tR = size(combine_tR,1);
    
    data_source = cell(1,2); %1 left 2 right
    channels = size(combine_aL, 2);
    sigma1 = zeros(channels, channels);
    sigma2 = zeros(channels, channels);

    for idx = 1:num_aL
        % left data
        data_source{1}{idx} = squeeze(combine_aL(idx,:,:));
        aL_tmp = squeeze(combine_aL(idx,:,:));
        aL_cov = cov(aL_tmp.');
        sigma1 = sigma1 + aL_cov/trace(aL_cov);
        % right data
        data_source{2}{idx} = squeeze(combine_aR(idx,:,:));
        aR_tmp = squeeze(combine_aR(idx,:,:));
        aR_cov = cov(aR_tmp.');
        sigma2 = sigma2 + aR_cov/trace(aR_cov);
    end
    for idx = 1:num_tL
        % left data
        data_source{1}{num_aL+idx} = squeeze(combine_tL(idx,:,:));
        tL_tmp = squeeze(combine_tL(idx,:,:));
        tL_cov = cov(tL_tmp.');
        sigma1 = sigma1 + tL_cov/trace(tL_cov);
        % right data
        data_source{2}{num_aR+idx} = squeeze(combine_tR(idx,:,:));
        tR_tmp = squeeze(combine_tR(idx,:,:));
        tR_cov = cov(tR_tmp.');
        sigma2 = sigma2 + tR_cov/trace(tR_cov);
    end
    lambda1 = cal_shrinkage_modified([combine_aL; combine_tL]);
    lambda2 = cal_shrinkage_modified([combine_aR; combine_tR]);
    
    sigma1 = sigma1/(num_aL+num_tL);
    sigma2 = sigma2/(num_aR+num_tR);
    gm = cell(1,2); %cm 1 left 2 right 
    gm{1} = lambda1*eye(channels) + (1-lambda1)*sigma1;
    gm{2} = lambda2*eye(channels) + (1-lambda2)*sigma2;
%     gm{1} = sigma1;
%     gm{2} = sigma2;
    gms{instance,i} = gm;
    data_sources{instance,i} = data_source;
end
end


for j = 1:12
    acc_instances = zeros(1,10);
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
        
    %% training CSP using new covariance matrix. data_source is useless
    [ csp_coeff,all_coeff] = csp_analysis(data_sources{instance,j},9,csp_per_class, 0,gms{instance,j});
    [ data_source_filter ] = csp_filtering(data_sources{instance,j}, csp_coeff);
    
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
    
    %         [X_LDA predicted_y_class2] = lda_apply(cell2mat(data_target_log{2})', W, B);    % there is a vector output! should all be -1!
    %         predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
    %         predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
    %         temp = [predicted_y_class1; predicted_y_class2];
    %         acc1 = sum(temp)/length(temp);   % this is the percent correct classification
    acc1 = 1 - test_err;
    acc_csp(j,instance) = acc1;
    
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
    
    test_err = sum(abs(y_test_t - predicted_y_class))/size(y_test_t, 1)/2;
    acc2 = 1 - test_err;
    acc_for_all(j,instance) = acc2;
    acc_instances(instance) = acc2;
end
    text = ['s', int2str(j)];
    fprintf(text)
    fprintf(': %f\n', mean(acc_instances))
end
mean_csp = mean(acc_csp,2);
mean_csp_sa = mean(acc_for_all,2);


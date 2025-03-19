% CSP_SA.m
close all
%extract 3 largest eigen value and 3 smallest.
csp_per_class = 3;

acc_csp = zeros(12,12,10);
acc_for_all = zeros(12,12,10);


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
    
    for j= [1:i-1 i+1:12]
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
        
        %% training CSP using new covariance matrix. data_source is useless
        [ csp_coeff,all_coeff] = csp_analysis(data_source,9,csp_per_class, 0,gm);
        [ data_source_filter ] = csp_filtering(data_source, csp_coeff);

        data_source_log{1} = log_norm_BP(data_source_filter{1}); 
        data_source_log{2} = log_norm_BP(data_source_filter{2});


        %% Apply CSP in target subject
        [ data_target_filter ] = csp_filtering(data_target, csp_coeff);
        data_target_log{1} = log_norm_BP(data_target_filter{1}); 
        data_target_log{2} = log_norm_BP(data_target_filter{2});

        %% train LDA. This is CSP without subspace alignment
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        [W, B, ~] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);
        [~, predicted_y_class1] = lda_apply([cell2mat(data_target_log{1})';cell2mat(data_target_log{2})'], W, B);
        test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1)/2;
        acc1 = 1-test_err;
        disp(['Acc: ' num2str(acc1)])
        acc_csp(i,j,instance) = acc1;

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
        [W,B,class_means] = lda_train_reg(Xproj(1:size_xproj,:), trainY, 0);
        
        [~, predicted_y_class1] = lda_apply(Xproj(size_xproj+1:end,:), W, B);
        test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1)/2;
        acc2 = 1-test_err;
        text = ['s', int2str(i),' to ','s', int2str(j)];
        fprintf(text)
        fprintf(': %f\n', acc2)
        acc_for_all(i,j,instance) = acc2;
    end
end
end
mean_csp = mean(acc_csp,3);
mean_csp_sa = mean(acc_for_all,3);




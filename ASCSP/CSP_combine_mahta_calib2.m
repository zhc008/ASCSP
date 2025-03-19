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
    num_aL = floor(size(aL_calib,3)/2);
    num_tL = floor(size(tL_calib,3)/2);
    num_aR = floor(size(aR_calib,3)/2);
    num_tR = floor(size(tR_calib,3)/2);
    
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
    name1 = ['../Mahta_data/S',int2str(j),'_calib.mat'];
    load(name1)
    % get the CSP filters and the covariance matrix for left hand and right
    % hand data
    % number of channels * number of time samples * number of epochs

    % number of trials in one class
%     num_aL = size(aL_calib,3);
%     num_tL = size(tL_calib,3);
%     num_aR = size(aR_calib,3);
%     num_tR = size(tR_calib,3);
    num_aL = floor(size(aL_calib,3)/2);
    num_tL = floor(size(tL_calib,3)/2);
    num_aR = floor(size(aR_calib,3)/2);
    num_tR = floor(size(tR_calib,3)/2);
    
    aL_calib = aL_calib(:,:,num_aL+1:end);
    tL_calib = tL_calib(:,:,num_tL+1:end);
    aR_calib = aR_calib(:,:,num_aR+1:end);
    tR_calib = tR_calib(:,:,num_tR+1:end);
    
    num_aL = size(aL_calib,3);
    num_tL = size(tL_calib,3);
    num_aR = size(aR_calib,3);
    num_tR = size(tR_calib,3);
    
    data_target = cell(1,2); %1 left 2 right
    
    
    % left data
    for idx = 1:num_aL
        data_target{1}{idx} = squeeze(aL_calib(:,:,idx));
    end
    for idx = 1:num_tL
        data_target{1}{num_aL+idx} = squeeze(tL_calib(:,:,idx));
    end
    
    % right data
    for idx = 1:num_aR
        data_target{2}{idx} = squeeze(aR_calib(:,:,idx));
    end
    for idx = 1:num_tR
        data_target{2}{num_aR+idx} = squeeze(tR_calib(:,:,idx));
    end
    y_test_t = [(-1)*ones(size(data_target{1},2),1); ones(size(data_target{2},2),1)];
    
    predicted_ys = zeros(size(y_test_t,1),1);
    denominator = 0;
    
    new_source = cell(1,2);
    new_gm = cell(1,2);
    sigma1 = zeros(channels, channels);
    sigma2 = zeros(channels, channels);
    new_gm{1} = sigma1;
    new_gm{2} = sigma2;
    for i = [1:j-1 j+1:12]
        tmp_source = data_sources{i};
        tmp_gm = gms{i};
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
    
    test_err = sum(abs(y_test_t - predicted_y_class))/size(y_test_t, 1)/2;
    acc2 = 1 - test_err;
    text = ['s', int2str(j)];
    fprintf(text)
    fprintf(': %f\n', acc2)
    acc_for_all(j) = acc2;
end

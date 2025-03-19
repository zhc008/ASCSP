% Main file for ASCSP with subspace alignment.
close all
% fileID = fopen('result/result_ASACSP_simplify.txt','w');

%extract 3 largest eigen value and 3 smallest.
csp_per_class = 3;

SUBS_NAM = {'S_BP-240416-1','S_BP-240416-2','S_BP-240416-3','S_BP-270416-2','S_BP-130516-1','S_BP-141216'};
freqsRange = [[1, 3]; [2, 5]; [4, 7];[6, 10]; [7, 12]; [10, 15]; [12, 19]; [18, 25]; [19, 30]; [25, 35]; [30, 40]];

%load general matrix to cm
%cm is a 5-D matrix with the size of (6,11,2,47,47)
%where 6 is the subject index, 11 is the 11 frequencies, 2 the two class
%label, 47x47 is the covariance matrix.
%cm stores the averaged normalized covariance in each 6 subject each class.

%load('CSP_covariance_matrix.mat');
acc_for_all = zeros(6,6);
gms = cell(1,6);
data_sources = cell(1,6);

freqs_idx=5;
%except 7, someting wrong
%training subject index from 1 to 6
for i = 1:6
    if i == 7
        continue
    end
%     fprintf(fileID,'Tain ID %d\n',sub_idx);

    % Only use the frequency of 7-12
    
    name = ['../Motor_Imagery_Data/',SUBS_NAM{i}, 'freqs7_12_shams_FP.mat'];
    load(name)
    

    %number of trials in one subject one class
    num_tmp = size(prepData.B_l,2);

    channels = size(prepData.B_l{1},1);
    sigma1 = zeros(channels, channels);
    sigma2 = zeros(channels, channels);
    
    %prepare source data
    % left: Gl and Bl
    data_source = cell(1,2); %1 left 2 right
    for idx = 1:num_tmp
        %trial_num = size(train_Gl, 2);
        
        % left data both good and bad with instance 1
        X1_temp = prepData.G_l{idx};
        data_source{1}{idx} = X1_temp;
        cov1_temp = cov(X1_temp.');
        sigma1 = sigma1 + cov1_temp/trace(cov1_temp);
        
        %data_source{1}{i} = [reshape( train_Gl(freqs_idx,i), [] ) ; train_Bl()];
        X2_temp = prepData.G_r{idx};
        data_source{2}{idx} = X2_temp;
        cov2_temp = cov(X2_temp.');
        sigma2 = sigma2 + cov2_temp/trace(cov2_temp);
        %data_source{2}{i} = prepData.G_r{i};
    end
   
    for idx = 1:num_tmp
        %trial_num = size(train_Gl, 2);
        
        % left data both good and bad with instance 1
        X1_temp = prepData.B_l{idx};
        data_source{1}{idx+num_tmp} = X1_temp;
        cov1_temp = cov(X1_temp.');
        sigma1 = sigma1 + cov1_temp/trace(cov1_temp);
        
        %data_source{1}{i} = [reshape( train_Gl(freqs_idx,i), [] ) ; train_Bl()];
        X2_temp = prepData.B_r{idx};
        data_source{2}{idx+num_tmp} = X2_temp;
        cov2_temp = cov(X2_temp.');
        sigma2 = sigma2 + cov2_temp/trace(cov2_temp);
        %data_source{2}{i} = prepData.G_r{i};
    end
    sigma1 = sigma1/(num_tmp*2);
    sigma2 = sigma2/(num_tmp*2);
    gm = cell(1,2); %cm 1 left 2 right 
    gm{2} = sigma2;
    gm{1} = sigma1;
    gms{i} = gm;
    data_sources{i} = data_source;
end
for j= 1:6
    if j == 7
        continue
    end
    name = ['../Motor_Imagery_Data/',SUBS_NAM{j}, 'freqs7_12_shams_FP.mat'];
    load(name)
    
    num_tmp = size(prepData.B_l,2);
    
    % store target data in data_target
    data_target = cell(1,2);
    for target_trial = 1:num_tmp
        
        %             data_target{1}{target_trial} = squeeze(train_Gl(freqs_idx,target_trial,:,:) );
        %             data_target{2}{target_trial} = squeeze(train_Gr(freqs_idx, target_trial,:,:));
        data_target{1}{target_trial} = prepData.G_l{target_trial};
        data_target{2}{target_trial} = prepData.G_r{target_trial};
    end
    
    for target_trial = 1:num_tmp
        
        %             data_target{1}{target_trial+num_tmp} = squeeze(train_Bl(freqs_idx,target_trial,:,:) );
        %             data_target{2}{target_trial+num_tmp} = squeeze(train_Br(freqs_idx, target_trial,:,:));
        data_target{1}{target_trial+num_tmp} = prepData.B_l{target_trial};
        data_target{2}{target_trial+num_tmp} = prepData.B_r{target_trial};
        %data_target{1}{target_trial} = prepData.G_l{target_trial};
        %data_target{2}{target_trial} = prepData.G_r{target_trial};
    end
    
    predicted_ys = zeros(4*num_tmp,1);
    denominator = 0;
    
    for i = [1:j-1 j+1:6]
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
        param = []; param.pcaCoef = 2;
        [Xproj,transMdl] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);
        
        %% Train LDA. ASCSP with subspace alignment
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        size_xproj = size(data_source_log{1},2) + size(data_source_log{2},2);
        [W,B,~] = lda_train_reg(Xproj(1:size_xproj,:), trainY, 0);
        
        
        left_size = size(Xproj,1)-size_xproj;
        assert(mod(left_size,2) == 0, "Size mismatches");
        [X_LDA, predicted_y_class] = lda_apply(Xproj(size_xproj+1:end,:), W, B);
        X_LDAs = cell(1,6);
        
        estimate_performance = zeros(4*num_tmp,1);
        for k = 1:6
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
            [Xproj2,transMdl] = ftTrans_sa(X2,maLabeled2,Y2(maLabeled2),maLabeled2,param);
            
            %% Train LDA. ASCSP with subspace alignment
            trainY2 = [(-1)*ones(size(data_source_log2{1},2),1); ones(size(data_source_log2{2},2),1)];
            size_xproj2 = size(data_source_log2{1},2) + size(data_source_log2{2},2);
            [W2,B2,~] = lda_train_reg(Xproj2(1:size_xproj2,:), trainY2, 0);
            
            
            left_size2 = size(Xproj2,1)-size_xproj2;
            assert(mod(left_size2,2) == 0, "Size mismatches");
            [X_LDA, predicted_y_class2] = lda_apply(Xproj2(size_xproj2+1:end,:), W2, B2);
            X_LDAs{k} = X_LDA;
            estimate_performance = estimate_performance + predicted_y_class2;
        end
        for k = 1:4*num_tmp
            if estimate_performance(k) == 0
                for kk = 1:6
                    XLDAk = X_LDAs{kk};
                    if size(XLDAk) == 0
                        continue;
                    end
                    estimate_performance(k) = estimate_performance(k) + XLDAk(k);
                end
%                 disp(estimate_performance(k))
            end
        end
        estimate_performance = sign(estimate_performance);
        err2 = sum(abs(estimate_performance - predicted_y_class))/2/size(data_target_log{1},2);
        tmp_acc = 1-err2/2;
        predicted_ys = predicted_ys + tmp_acc * predicted_y_class;
        denominator = denominator + tmp_acc;
    end
    predicted_ys = predicted_ys/denominator;
    predicted_y_class1 = sign(predicted_ys);
    y_test_t = [(-1)*ones(2*num_tmp,1); ones(2*num_tmp,1)];
    %       predicted_y_class1(predicted_y_class1 == 0) = randi([1,2]);
    test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1)/2;
    %           [X_LDA predicted_y_class2] = lda_apply(Xproj(size_xproj+left_size/2+1:end,:), W, B);    % there is a vector output! should all be -1!
    %           predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
    %           predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
    %           temp = [predicted_y_class1; predicted_y_class2];
    %           acc2 = sum(temp)/length(temp);   % this is the percent correct classification
    acc2 = 1 - test_err;
    text = [SUBS_NAM{j}];
    fprintf(text)
    fprintf(': %f\n', acc2)
    acc_for_all(i,j) = acc2;
    
end
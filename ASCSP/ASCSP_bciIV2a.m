% ASCSP for bci IV 2a
acc_csp = zeros(9,9);
acc_for_all = zeros(9,9);
csp_per_class = 3;
for i = 1:9 %loop through the 9 subjects
    name1 = ['../BCICIV_2a_gdf/A0',int2str(i),'T.gdf'];
    [s_train,h_train] = sload(name1);
    [X1_s, X2_s] = extract_bciIV2a_train(s_train, h_train);
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
    for j = 1:9
        if j == i
            continue; 
        end
        name2 = ['../BCICIV_2a_gdf/true_labels/A0',int2str(j),'E.mat'];
        name3 = ['../BCICIV_2a_gdf/A0',int2str(j),'E.gdf'];
        test_label = load(name2).classlabel;
        [s_test,h_test] = sload(name3);
        [X_test_t, y_test_t] = extract_bciIV2a_test(s_test, h_test, test_label);
        
        num_tmp2 = size(X_test_t,1);
        data_target = cell(1,1);
        for idx = 1:num_tmp2
            
            data_target{1}{idx} = squeeze(X_test_t(idx,:,:));
            
        end
        
        [gm1,store_idx] = new_update_test_train(gm,data_source,data_target);
        %% training CSP using new covariance matrix. data_source is useless
        [ csp_coeff,all_coeff] = csp_analysis(data_source,9,csp_per_class, 0,gm1);
        [ data_source_filter ] = csp_filtering(data_source, csp_coeff);

        data_source_log{1} = log_norm_BP(data_source_filter{1}); 
        data_source_log{2} = log_norm_BP(data_source_filter{2});


        %% Apply CSP in target subject
        [ data_target_filter ] = csp_filtering(data_target, csp_coeff);
        data_target_log{1} = log_norm_BP(data_target_filter{1}); 
%         data_target_log{2} = log_norm_BP(data_target_filter{2});

        %% train LDA. This is ASCSP without subspace alignment
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        [W B class_means] = lda_train_reg([cell2mat(data_source_log{1})';cell2mat(data_source_log{2})'], trainY, 0);
        [X_LDA, predicted_y_class1] = lda_apply(cell2mat(data_target_log{1})', W, B);
        predicted_y_class1(predicted_y_class1 == 1) = 2;   % incorrect choice
        predicted_y_class1(predicted_y_class1 == -1) = 1;
        
        test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1);
        
%         [X_LDA predicted_y_class2] = lda_apply(cell2mat(data_target_log{2})', W, B);    % there is a vector output! should all be -1! 
%         predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
%         predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
%         temp = [predicted_y_class1; predicted_y_class2];
%         acc1 = sum(temp)/length(temp);   % this is the percent correct classification 
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
        [Xproj,transMdl] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);

        %% Train LDA. ASCSP with subspace alignment
        trainY = [(-1)*ones(size(data_source_log{1},2),1); ones(size(data_source_log{2},2),1)];
        size_xproj = size(data_source_log{1},2) + size(data_source_log{2},2);
        [W,B,class_means] = lda_train_reg(Xproj(1:size_xproj,:), trainY, 0);
        
        
        left_size = size(Xproj,1)-size_xproj;
        assert(mod(left_size,2) == 0, "Size mismatches");
        
        [X_LDA predicted_y_class1] = lda_apply(Xproj(size_xproj+1:end,:), W, B);
        predicted_y_class1(predicted_y_class1 == 1) = 2;   % incorrect choice
        predicted_y_class1(predicted_y_class1 == -1) = 1;
        test_err = sum(abs(y_test_t - predicted_y_class1))/size(y_test_t, 1);
%         [X_LDA predicted_y_class2] = lda_apply(Xproj(size_xproj+left_size/2+1:end,:), W, B);    % there is a vector output! should all be -1! 
%         predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
%         predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
%         temp = [predicted_y_class1; predicted_y_class2];
%         acc2 = sum(temp)/length(temp);   % this is the percent correct classification 
        acc2 = 1 - test_err;
        text = ['s',int2str(i),' to s',int2str(j)];
        fprintf(text)
        fprintf(': %f\n', acc2)
        acc_for_all(i,j) = acc2;
   end
end
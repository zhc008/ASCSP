% new_ASCSP_test.m
sub_names = {'a','l','v','w','y'};
acc_for_all = zeros(5,5);
for i = 1:4
    name1 = ['../data/data_set_IVa_a',sub_names{i},'.mat'];
    name2 = ['../data/true_labels_a',sub_names{i},'.mat'];
    load(name1)
    load(name2)
    [X1_s, X2_s, X_test_s, y_test_s] = extract_data(mrk, cnt, test_idx, true_y);
    % get the CSP filters and the covariance matrix for left hand and right
    % hand data
    if size(X1_s,1) < size(X2_s,1)
        X2_s = X2_s(1:size(X1_s,1),:,:);
    end
    if size(X1_s,1) > size(X2_s,1)
        X1_s = X1_s(1:size(X2_s,1),:,:);
    end
    X2_s = X2_s(1:25,:,:);
    X1_s = X1_s(1:25,:,:);
    [~, cov_s1, cov_s2] = CSP(X1_s, X2_s);
    gm = cell(1,2); %cm 1 left 2 right 
    gm{1} = cov_s1;
    gm{2} = cov_s2;
    for j = 1:5
        if j == i
           continue; 
        end
        name1_t = ['../data/data_set_IVa_a',sub_names{j},'.mat'];
        name2_t = ['../data/true_labels_a',sub_names{j},'.mat'];
        load(name1_t)
        load(name2_t)
        [X1_t, X2_t, X_test_t, y_test_t] = extract_data(mrk, cnt, test_idx, true_y);
        if size(X1_t,1) < size(X2_t,1)
            X2_t = X2_t(1:size(X1_t,1),:,:);
        end
        if size(X1_t,1) > size(X2_t,1)
            X1_t = X1_t(1:size(X2_t,1),:,:);
        end
        [gm1,store_idx] = new_update_test(gm,X1_s,X2_s,X1_t,X2_t);

        %% train CSP with new covariance matrix
        [EVector,~] = eig(gm1{1}, gm1{1}+gm1{2});
        W = [EVector(:, 1:3), EVector(:, 66:end)];
        f_X1_s = features(W, X1_s);
        f_X2_s = features(W, X2_s);
        f_X1_t = features(W, X1_t);
        f_X2_t = features(W, X2_t);

        %% Subspace
        X = zeros(0,6);
        Y = zeros(0,1);
        maLabeled = false(0,1);
        % Source Domain
        for ii = 1:size(f_X1_s,1)
            Y = [Y;1];
            maLabeled = [maLabeled;true];
        end
        X = [X;f_X1_s];
        for ii = 1:size(f_X2_s,1)
            Y = [Y;2];
            maLabeled = [maLabeled;true];
        end
        X = [X;f_X2_s];
        
        
        % Target Domain
        for ii = 1:size(f_X1_t,1)
            Y = [Y;1];
            maLabeled = [maLabeled;false];
        end
        X = [X;f_X1_t];
        for ii = 1:size(f_X2_t,1)
            Y = [Y;2];
            maLabeled = [maLabeled;false];
        end
        X = [X;f_X2_t];
        param = []; param.pcaCoef = 2;
        [Xproj,transMdl] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);

        %% Train LDA. ASCSP with subspace alignment
        trainY = [(-1)*ones(size(f_X1_s,1),1); ones(size(f_X2_s,1),1)];
        size_xproj = size(f_X1_s,1) + size(f_X2_s,1);
        [W,B,class_means] = lda_train_reg(Xproj(1:size_xproj,:), trainY, 0);
        
        
        left_size = size(Xproj,1)-size_xproj;
        assert(mod(left_size,2) == 0, "Size mismatches");
        
        [X_LDA predicted_y_class1] = lda_apply(Xproj(size_xproj+1:size_xproj+left_size/2,:), W, B);
        predicted_y_class1(predicted_y_class1 == 1) = 0;   % incorrect choice
        predicted_y_class1(predicted_y_class1 == -1) = 1;
        [X_LDA predicted_y_class2] = lda_apply(Xproj(size_xproj+left_size/2+1:end,:), W, B);    % there is a vector output! should all be -1! 
        predicted_y_class2(predicted_y_class2 == -1) = 0;   % wrong choice
        predicted_y_class2(predicted_y_class2 == 1) = 1;   % correct choice
        temp = [predicted_y_class1; predicted_y_class2];
        acc2 = sum(temp)/length(temp);   % this is the percent correct classification 
        text = [sub_names{i},' to ',sub_names{j}];
        fprintf(text)
        fprintf(': %f\n', acc2)
        acc_for_all(i,j) = acc2;
    end
end

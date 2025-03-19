function [result,store_idx] = update_vote4(gm,data_source,data_target,predicted_ys,denominator)
csp_per_class=3;
result=gm;
n_target = size(data_target{1},2);
n_source = size(data_source{1},2);
num_select = size(data_source{1},2);
acc = 0;
total = 0;

store_idx = 0;

isVis = zeros(1,n_target);

flag = false;

% select trial from target subject
%left or right index, random selectly one from subject
leftIdx = 0;
rightIdx = n_target;


% label: (i+1)%2 + 1, sampleIdx: ceil(i/2)
for i = 1:n_target
    
    %size(data_target{1})
    disp(['Processing ' num2str(i)])
    tmp_acc = 0;
    label_idx = 0;
    best_mean1 = 0;
    best_mean2 = 0;
    
    
    %% train a LDA, flag is used to determine whether it is the first time
    %train LDA in source domain
    if flag == true
        trainY = [(-1)*ones(n_source,1); ones(n_source,1)];
        
        % train on source target after subspace alignment
        [W,B,~] = lda_train_reg(Xproj_tmp(1:n_source*2,:), trainY, 0);
        
        mappedIdx = i;
        
        % left label
        
        % 1 is right label, -1 is left label. We need left Label   
        [X_LDA, predicted_y_class1] = lda_apply(Xproj_tmp(n_source*2+mappedIdx,:), W, B);
        % vote
        predicted_y1 = (predicted_y_class1 + predicted_ys(i))/denominator;
        if abs(predicted_y1) < 1/(2*denominator)
            continue;
        end
        predicted_y1 = sign(predicted_y1);
           
        if predicted_y1 == 1
            continue;
        end
    end
    
    
    
    %% update covariance matrix
    result_tmp = result;
    
    %readIdx = 1;
    
    
    labelIdx = 1;
    sampleIdx = i;
    
    % get new convariance matrix
    C_new = cov(data_target{labelIdx }{sampleIdx}')/trace(cov( data_target{labelIdx}{sampleIdx}'));
    
    %left
    result_tmp{1} = result_tmp{1}*num_select/(num_select+1) + C_new/(num_select+1);
    for j = (i+1):n_target
        if isVis(j) == 1
            continue;
        end
        
        labelIdx_next = 1;
        sampleIdx_next = j;

        C_new = cov( data_target{labelIdx_next}{sampleIdx_next}')/trace(cov(data_target{labelIdx_next}{sampleIdx_next}'));
        result_tmp_1 = result_tmp;
        result_tmp_1{2} = result_tmp{2}*num_select/(num_select+1) + C_new/(num_select+1);
        
        %% calculate CSP using new general matrix
        [ csp_coeff,~] = csp_analysis(data_source,9,csp_per_class, 0,result_tmp_1); % actually don't need data_source
        [ data_source_filter ] = csp_filtering(data_source, csp_coeff);
        [data_target_filter] = csp_filtering(data_target,csp_coeff);
        data_source_log{1} = log_norm_BP(data_source_filter{1}); 
        data_source_log{2} = log_norm_BP(data_source_filter{2});
        data_target_log{1} = log_norm_BP(data_target_filter{1}); 

        %% SA
        %CSP has 3 classes, 6 means highes and lowest?
        X = zeros(0,6);
        Y = zeros(0,1);
        maLabeled = false(0,1);
        % Source Domain
        for idx = 1:2
            for ii = 1:size(data_source_log{idx},2)
                X = [X;data_source_log{idx}{ii}'];
                Y = [Y;idx];
                maLabeled = [maLabeled;true];
            end
        end

        % Target Domain
        for idx = 1
            for ii = 1:size(data_target_log{1},2)
                X = [X;data_target_log{idx}{ii}'];
                Y = [Y;idx];
                maLabeled = [maLabeled;false];
            end
        end
        
        % subspace alignment
        param = []; param.pcaCoef = 2;
        [Xproj,~] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);
        mean1 = mean(X(1:n_source*2,:));
        mean2 = mean(X(2*n_source+1:end,:));
        mean_dif = sum(abs(mean1-mean2));
        
        %% 5-fold cross validation
        rng('default')
        shuffle1 = randperm(n_source);
        shuffle2 = randperm(n_source);
        shuffled = cell(1,2);
        shuffled{1} = data_source{1}(shuffle1);
        shuffled{2} = data_source{2}(shuffle2);
        split_size = floor(n_source/5);
        cv_acc = zeros(5,1);
        for g = 1:5
     
            X_train = cell(1,2);
            if g == 1
                X_train{1} = shuffled{1}(g*split_size+1:end);
                X_train{2} = shuffled{2}(g*split_size+1:end);
            elseif g == 5
                X_train{1} = shuffled{1}((g-2)*split_size+1:(g-1)*split_size);
                X_train{2} = shuffled{2}((g-2)*split_size+1:(g-1)*split_size);
            else
                X_train{1} = shuffled{1}([(g-2)*split_size+1:(g-1)*split_size g*split_size+1:end]);
                X_train{2} = shuffled{2}([(g-2)*split_size+1:(g-1)*split_size g*split_size+1:end]);
            end
            % calculate training covariance matrix
            train_size = size(X_train{1},2);
            for gg = 1:total
                for ggg = 1:2
                    X_train{ggg}{train_size+gg} = data_target{1}{store_idx(gg,ggg)};
                end
            end
            X_train{1}{train_size+total+1} = data_target{1}{i};
            X_train{2}{train_size+total+1} = data_target{1}{j};
            train_size = size(X_train{1},2);
            train_cm = cell(1,2);
            channels = size(X_train{1}{1},1);
            sigma1 = zeros(channels,channels);
            sigma2 = zeros(channels,channels);
            train_cm{1} = sigma1;
            train_cm{2} = sigma2;
            for gg = 1:train_size
                for ggg = 1:2
                    tmp_train = X_train{ggg}{gg};
                    cov_tmp = cov(tmp_train.');
                    train_cm{ggg} = train_cm{ggg}+cov_tmp/trace(cov_tmp);
                end
            end
            
            train_cm{1} = train_cm{1}/train_size;
            train_cm{2} = train_cm{2}/train_size;
            %% csp
            [ csp_coeff2,~] = csp_analysis(data_source,9,csp_per_class, 0,train_cm);
            [ data_source_filter2 ] = csp_filtering(data_source, csp_coeff2);
            
            data_source_log2{1} = log_norm_BP(data_source_filter2{1});
            data_source_log2{2} = log_norm_BP(data_source_filter2{2});
            
            [ data_target_filter2 ] = csp_filtering(data_target, csp_coeff2);
            data_target_log2{1} = log_norm_BP(data_target_filter2{1});

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
            
            for idx = 1
                for idx2 = 1:size(data_target_log2{idx},2)
                    X2 = [X2;data_target_log2{idx}{idx2}'];
                    Y2 = [Y2;idx];
                    maLabeled2 = [maLabeled2;false];
                end
            end
            param = []; param.pcaCoef = 2;
            [Xproj2,~] = ftTrans_sa(X2,maLabeled2,Y2(maLabeled2),maLabeled2,param);
            
            shuffled1 = Xproj2(1:n_source,:);
            shuffled2 = Xproj2(n_source+1:2*n_source,:);
            shuffled1 = shuffled1(shuffle1,:);
            shuffled2 = shuffled2(shuffle2,:);
            if g == 5
                X_valid1 = shuffled1((g-1)*split_size+1:end,:);
                X_valid2 = shuffled2((g-1)*split_size+1:end,:);
            else
                X_valid1 = shuffled1((g-1)*split_size+1:g*split_size,:);
                X_valid2 = shuffled2((g-1)*split_size+1:g*split_size,:);
            end
            X_valid = [X_valid1;X_valid2];

            if g == 1
                X_train_new1 = shuffled1(g*split_size+1:end,:);
                X_train_new2 = shuffled2(g*split_size+1:end,:);
            elseif g == 5
                X_train_new1 = shuffled1((g-2)*split_size+1:(g-1)*split_size,:);
                X_train_new2 = shuffled2((g-2)*split_size+1:(g-1)*split_size,:);
            else
                X_train_new1 = shuffled1([(g-2)*split_size+1:(g-1)*split_size g*split_size+1:end],:);
                X_train_new2 = shuffled2([(g-2)*split_size+1:(g-1)*split_size g*split_size+1:end],:);
            end
            X_train_tmp1 = zeros(total+1,2);
            X_train_tmp2 = zeros(total+1,2);
            if total ~= 0
                X_train_tmp1(1:total,:) = Xproj2(2*n_source+store_idx(:,1),:);
        
                X_train_tmp2(1:total,:) = Xproj2(2*n_source+store_idx(:,2),:);
            end
            X_train_tmp1(end,:) = Xproj(2*n_source+i,:);
            X_train_tmp2(end,:) = Xproj(2*n_source+j,:);
            X_train_new = [X_train_new1;X_train_tmp1;X_train_new2;X_train_tmp2];
            
            %% Train LDA. ASCSP with subspace alignment
            trainY2 = [(-1)*ones(train_size,1); ones(train_size,1)];

            [W2,B2,~] = lda_train_reg(X_train_new, trainY2, 0);
           
            [~, predicted_y_class2] = lda_apply(X_valid, W2, B2);
            testY2 = [(-1)*ones(size(X_valid1,1),1); ones(size(X_valid1,1),1)];
            err2 = sum(abs(testY2 - predicted_y_class2))/size(X_valid,1);
            acc1 = 1-err2/2;
            cv_acc(g) = acc1;
        end
        valid_acc = mean(cv_acc);
        
        %% select the largest validation accuracy

        if ((valid_acc > tmp_acc)||(valid_acc == tmp_acc && mean_dif < mean_pre))
            flag = true;
            Xproj_tmp = Xproj;
            best_mean1 = mean1;
            best_mean2 = mean2;
            tmp_acc = valid_acc;
            label_idx = j;
            mean_pre = mean_dif;
        end
        %idx_all(j) = ~idx_all(j);
        
    end
    
    %% already selecting two trials, updating covariance matrix
    if tmp_acc ~=0
        labelIdx_left = 1;
        sampleIdx_left = i;
        
        labelIdx_right = 1;
        sampleIdx_right = label_idx;
        
        C_new = cov(data_target{labelIdx_left}{sampleIdx_left}')/trace(cov(data_target{labelIdx_left}{sampleIdx_left}'));
        result{1} = result{1}*num_select/(1+num_select) + C_new/(1+num_select);
        C_new = cov(data_target{labelIdx_right}{sampleIdx_right}')/trace(cov(data_target{labelIdx_right}{sampleIdx_right}'));
        result{2} = result{2}*num_select/(1+num_select) + C_new/(1+num_select);
        %idx_all(label_idx) = ~idx_all(label_idx);
        %var2 = tmp_var;
        
        num_select = num_select + 1;
        
        isVis(label_idx) = 1;
        disp(['First ' num2str(i) ' Second: ' num2str(label_idx) ' Acc: ' num2str(tmp_acc) ' Mean1: ' num2str(sum(abs(best_mean1))) ' Mean2: ' num2str(sum(abs(best_mean2)))]);
        total = total+1;
        store_idx(total,1) = i;
        store_idx(total,2) = label_idx;
        
        if(label_idx>162)
            acc = acc+1;
        end
    end
    
    %disp(['var2: ' num2str(var2)]);
    

end

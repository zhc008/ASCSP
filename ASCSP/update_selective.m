% update_selective.m
function [result,store_idx] = update_selective(gm,data_source,data_target)
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
        [W,B,class_means] = lda_train_reg(Xproj_tmp(1:n_source*2,:), trainY, 0);
        
        mappedIdx = i;
        
        % left label
        
        % 1 is right label, -1 is left label. We need left Label   
        [X_LDA predicted_y_class1] = lda_apply(Xproj_tmp(n_source*2+mappedIdx,:), W, B);
           
        if predicted_y_class1 == 1 && X_LDA>0.005
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
    if flag == true
        result_tmp{1} = result_tmp{1}*num_select/(num_select+1) + C_new/(num_select+1);
    end
 
    for j = (i+1):n_target
        if isVis(j) == 1
            continue;
        end
        
        labelIdx_next = 1;
        sampleIdx_next = j;

        C_new = cov( data_target{labelIdx_next}{sampleIdx_next}')/trace(cov(data_target{labelIdx_next}{sampleIdx_next}'));
        result_tmp_1 = result_tmp;
        if flag == true
            result_tmp_1{2} = result_tmp{2}*num_select/(num_select+1) + C_new/(num_select+1);
        end
        
        %% calculate CSP using new general matrix
        [ csp_coeff,all_coeff] = csp_analysis(data_source,9,csp_per_class, 0,result_tmp_1); % actually don't need data_source
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
        [Xproj,transMdl] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);
        
        if flag == false
            flag = true;
            continue;
        end
        
        mean1 = mean(X(1:n_source*2,:));
        mean2 = mean(X(2*n_source+1:end,:));
        mean_dif = sum(abs(mean1-mean2));
        
%         %% 6 is the threshold
%         if  mean_dif > 6
%             continue;
%         end
%         
%         %% the mean also should not be too large. (Not present in the algorithm)
%         if sum(abs(mean2)) > 20 || sum(abs(mean1)) > 20
%             continue;
%         end
        
%         %% two fold cross validation on source dataset using current csp_SA model
%         rng('default')
%         shuffle1 = randperm(n_source);
%         shuffle2 = randperm(n_source);
%         X1 = Xproj(1:n_source,:);
%         X2 = Xproj(n_source+1:n_source*2,:);
%         X_split = cell(1,2);
%         y_split = cell(1,2);
%         accs = zeros(2,1);
%         split_size = floor(n_source/2);
%         X1_shuffled = X1(shuffle1,:);
%         X2_shuffled = X2(shuffle2,:);
%         X1_split1 = X1_shuffled(1:split_size,:);
%         X2_split1 = X2_shuffled(1:split_size,:);
%         X_split{1} = [X1_split1;X2_split1];
%         X1_split2 = X1_shuffled(1+split_size:end,:);
%         X2_split2 = X2_shuffled(1+split_size:end,:);
%         X_split{2} = [X1_split2;X2_split2];
%         y_split{1} = [(-1)*ones(split_size,1); ones(split_size,1)];
%         y_split{2} = [(-1)*ones(n_source-split_size,1); ones(n_source-split_size,1)];
%         
%         % train lda and average the validation acc
%         for idx = 1:2
%             [W1,B1,~] = lda_train_reg(X_split{idx}, y_split{idx}, 0);
%             [~, predicted_y1] = lda_apply(X_split{3-idx}, W1, B1);
%             test_err = sum(abs(y_split{3-idx} - predicted_y1))/size(y_split{3-idx}, 1);
%             accs(idx) = 1-test_err/2;
%         end
%         valid_acc = mean(accs);
        %% two fold cross validation on source dataset using current csp_SA model
        rng('default')
        shuffle1 = randperm(n_source*2);
        shuffle2 = randperm(n_target);
        X_source = Xproj(1:n_source*2,:);
        X_target = Xproj(1+n_source*2:end,:);
        X_split = cell(1,2);
        y_split = cell(1,2);
        accs = zeros(2,1);
        split_source = n_source;
        split_target = floor(n_target/2);
        X_source_shuffled = X_source(shuffle1,:);
        X_target_shuffled = X_target(shuffle2,:);
        X_s_split1 = X_source_shuffled(1:split_source,:);
        X_t_split1 = X_target_shuffled(1:split_target,:);
        X_split{1} = [X_s_split1;X_t_split1];
        X_s_split2 = X_source_shuffled(1+split_source:end,:);
        X_t_split2 = X_target_shuffled(1+split_target:end,:);
        X_split{2} = [X_s_split2;X_t_split2];
        y_split{1} = [(-1)*ones(split_source,1); ones(split_target,1)];
        y_split{2} = [(-1)*ones(split_source,1); ones(n_target-split_target,1)];
        
        % train lda and average the validation acc
        for idx = 1:2
            [W1,B1,~] = lda_train_reg(X_split{idx}, y_split{idx}, 0);
            [~, predicted_y1] = lda_apply(X_split{3-idx}, W1, B1);
            test_err = sum(abs(y_split{3-idx} - predicted_y1))/size(y_split{3-idx}, 1);
            accs(idx) = 1-abs(test_err/2-0.5);
        end
        valid_acc = mean(accs);
        
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

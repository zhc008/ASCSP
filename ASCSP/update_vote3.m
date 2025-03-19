function [result,store_idx] = update_vote3(gm,data_source,data_target,predicted_ys,denominator)
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
    right_find = false;
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
%         if abs(predicted_y1) > denominator/4
        if abs(predicted_y1) < 1/2/denominator
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
        
        trainY2 = [(-1)*ones(n_source,1); ones(n_source,1)];
        
        % train on source target after subspace alignment
        [W2,B2,~] = lda_train_reg(Xproj(1:n_source*2,:), trainY2, 0);
        
        mappedIdx2 = j;
        
        % left label
        
        % 1 is right label, -1 is left label. We need right Label   
        [~, predicted_y_class2] = lda_apply(Xproj(n_source*2+mappedIdx2,:), W2, B2);
        % vote
        predicted_y2 = (predicted_y_class2 + predicted_ys(j))/denominator;
%         if abs(predicted_y2) > denominator/4
        if abs(predicted_y2) < 1/2/denominator
            continue;
        end
        
        predicted_y2 = sign(predicted_y2);
        if predicted_y2 == -1
            continue;
        else
            
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
            
            flag = true;
            Xproj_tmp = Xproj;
            best_mean1 = mean1;
            best_mean2 = mean2;
            label_idx = j;
            right_find = true;
            break;
        end
        %idx_all(j) = ~idx_all(j);
        
    end
    
    %% already selecting two trials, updating covariance matrix
    if right_find
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
        disp(['First ' num2str(i) ' Second: ' num2str(label_idx) ' Mean1: ' num2str(sum(abs(best_mean1))) ' Mean2: ' num2str(sum(abs(best_mean2)))]);
        total = total+1;
        store_idx(total,1) = i;
        store_idx(total,2) = label_idx;
        

    end
    
    %disp(['var2: ' num2str(var2)]);
    result{1} = (result{1}*num_select - 0.5*gm{1}*n_source)/(num_select-0.5*n_source);
    result{2} = (result{2}*num_select - 0.5*gm{2}*n_source)/(num_select-0.5*n_source);
    
end

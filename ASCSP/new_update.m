% new_update.m
function [result,store_idx] = new_update(gm,X1_s,X2_s,X1_t,X2_t)
result=gm;
n_target = size(X1_t,1);
n_source = size(X1_s,1);
num_select = size(X1_s,1);
acc = 0;
total = 0;

store_idx = 0;

isVis = zeros(1,n_target*2);

flag = false;

for i = 1:2*n_target
    
    %size(data_target{1})
    disp(['Processing ' num2str(i)])
    tmp_var = 0;
    label_idx = 0;
    best_mean1 = 0;
    best_mean2 = 0;
    
    
    %% train a LDA, flag is used to determine whether it is the first time
    %train LDA in source domain
    if flag == true
        trainY = [(-1)*ones(n_source,1); ones(n_source,1)];
        
        % train on source target after subspace alignment
        [W,B,class_means] = lda_train_reg(Xproj_tmp(1:n_source*2,:), trainY, 0);
        
        mappedIdx = ceil(i/2) + mod(i+1,2)*n_target;
        
        % left label
        
        % 1 is right label, -1 is left label. We need left Label   
        [X_LDA ,predicted_y_class1] = lda_apply(Xproj_tmp(n_source*2+mappedIdx,:), W, B);
           
        if predicted_y_class1 == 1 && X_LDA>0.005
            continue;
        end
    end
    
    
    
    %% update covariance matrix
    result_tmp = result;
    
    %readIdx = 1;
    
    
    labelIdx = mod(i+1,2)+1;
    sampleIdx = ceil(i/2);
    
    % get new convariance matrix
    if labelIdx == 1
        C_new = cov(squeeze(X1_t(sampleIdx,:,:))')/trace(cov(squeeze(X1_t(sampleIdx,:,:))'));
    else
        C_new = cov(squeeze(X2_t(sampleIdx,:,:))')/trace(cov(squeeze(X2_t(sampleIdx,:,:))'));
    end
    
    %left
    result_tmp{1} = result_tmp{1}*num_select/(num_select+1) + C_new/(num_select+1);
    for j = (i+1):n_target*2
        if isVis(j) == 1
            continue;
        end
        
        labelIdx_next = mod(j+1,2)+1;
        sampleIdx_next = ceil(j/2);

        if labelIdx_next == 1
            C_new = cov(squeeze(X1_t(sampleIdx_next,:,:))')/trace(cov(squeeze(X1_t(sampleIdx_next,:,:))'));
        else
            C_new = cov(squeeze(X2_t(sampleIdx_next,:,:))')/trace(cov(squeeze(X2_t(sampleIdx_next,:,:))'));
        end
        result_tmp_1 = result_tmp;
        result_tmp_1{2} = result_tmp{2}*num_select/(num_select+1) + C_new/(num_select+1);
        
        %% calculate CSP using new general matrix
        w = CSP_cov(result_tmp_1);
        f_X1_s = features(w, X1_s);
        f_X2_s = features(w, X2_s);
        f_X1_t = features(w, X1_t);
        f_X2_t = features(w, X2_t);

        %% SA
        %CSP has 3 classes, 6 means highes and lowest?
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
        
        % subspace alignment
        param = []; param.pcaCoef = 2;
        [Xproj,transMdl] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);
        mean1 = mean(X(1:n_source*2,:));
        mean2 = mean(X(2*n_source+1:end,:));
        mean_dif = sum(abs(mean1-mean2));
        
        %% 6 is the threshold
        if  mean_dif > 6
            continue;
        end
        
        %% the mean also should not be too large. (Not present in the algorithm)
        if sum(abs(mean2)) > 20 || sum(abs(mean1)) > 20
            continue;
        end
        
        %% select the largest variance
        var_t1 = mean(var(Xproj(n_source*2+1:end,:)));
        if(var_t1>tmp_var )
            flag = true;
            Xproj_tmp = Xproj;
            best_mean1 = mean1;
            best_mean2 = mean2;
            tmp_var = var_t1;
            label_idx = j;
            mean_pre = mean_dif;
        end
        %idx_all(j) = ~idx_all(j);
        
    end
    
    %% already selecting two trials, updating covariance matrix
    if tmp_var ~=0
        labelIdx_left = mod(i+1,2)+1;
        sampleIdx_left = ceil(i/2);
        
        labelIdx_right = mod(label_idx+1,2)+1;
        sampleIdx_right = ceil(label_idx/2);
        
        if labelIdx_left == 1
            C_new = cov(squeeze(X1_t(sampleIdx_left,:,:))')/trace(cov(squeeze(X1_t(sampleIdx_left,:,:))'));
        else
            C_new = cov(squeeze(X2_t(sampleIdx_left,:,:))')/trace(cov(squeeze(X2_t(sampleIdx_left,:,:))'));
        end
        result{1} = result{1}*num_select/(1+num_select) + C_new/(1+num_select);
        if labelIdx_right == 1
            C_new = cov(squeeze(X1_t(sampleIdx_right,:,:))')/trace(cov(squeeze(X1_t(sampleIdx_right,:,:))'));
        else
            C_new = cov(squeeze(X2_t(sampleIdx_right,:,:))')/trace(cov(squeeze(X2_t(sampleIdx_right,:,:))'));
        end
        result{2} = result{2}*num_select/(1+num_select) + C_new/(1+num_select);
        
%         C_new = cov(squeeze(X1_t(sampleIdx_left,:,:))')/trace(cov(squeeze(X1_t(sampleIdx_left,:,:))'));
%         result{1} = result{1}*num_select/(1+num_select) + C_new/(1+num_select);
%         C_new = cov(squeeze(X2_t(sampleIdx_right,:,:))')/trace(cov(squeeze(X2_t(sampleIdx_right,:,:))'));
%         result{2} = result{2}*num_select/(1+num_select) + C_new/(1+num_select);
        %idx_all(label_idx) = ~idx_all(label_idx);
        %var2 = tmp_var;
        
%         num_select = num_select + 1;
        
        isVis(label_idx) = 1;
        disp(['First ' num2str(i) ' Second: ' num2str(label_idx) ' Var: ' num2str(tmp_var) ' Mean1: ' num2str(sum(abs(best_mean1))) ' Mean2: ' num2str(sum(abs(best_mean2)))]);
        total = total+1;
        store_idx(total,1) = i;
        store_idx(total,2) = label_idx;
        if(label_idx>162)
            acc = acc+1;
        end
    end
    
    %disp(['var2: ' num2str(var2)]);
    
end

% %% show result for selecting accuracy
% [EVector,~] = eig(result{1}, result{1}+result{2});
% W = [EVector(:, 1:3), EVector(:, 66:end)];
% f_X1_s = features(W, X1_s);
% f_X2_s = features(W, X2_s);
% f_X1_t = features(W, X1_t);
% f_X2_t = features(W, X2_t);
% %% SA
% X = zeros(0,6);
% Y = zeros(0,1);
% maLabeled = false(0,1);
% % Source Domain
% for ii = 1:size(f_X1_s,1)
%     Y = [Y;1];
%     maLabeled = [maLabeled;true];
% end
% X = [X;f_X1_s];
% for ii = 1:size(f_X2_s,1)
%     Y = [Y;2];
%     maLabeled = [maLabeled;true];
% end
% X = [X;f_X2_s];
%         
%         
% % Target Domain
% for ii = 1:size(f_X1_t,1)
%     Y = [Y;1];
%     maLabeled = [maLabeled;false];
% end
% X = [X;f_X1_t];
% for ii = 1:size(f_X2_t,1)
%     Y = [Y;2];
%     maLabeled = [maLabeled;false];
% end
% X = [X;f_X2_t];
% param = []; param.pcaCoef = 2;
% [Xproj,transMdl] = ftTrans_sa(X,maLabeled,Y(maLabeled),maLabeled,param);
% 
% n = size(f_X1_s,1);
% mean1 = mean(X(1:n*2,:));
% mean2 = mean(X(2*n+1:end,:));
% disp(['Mean1: ' num2str(sum(abs(mean1))) ' Mean2: ' num2str(sum(abs(mean2)))]);
% disp(['Acc: ' num2str(acc/total) ' Total: ' num2str(total)]);

end
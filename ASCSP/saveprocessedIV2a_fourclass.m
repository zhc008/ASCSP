% preprocessing for bci IV 2a

%% save the preprocessed training data
for i = 1:9 %loop through the 9 subjects
    name1 = ['../BCICIV_2a_gdf/A0',int2str(i),'T.gdf'];
    [s_train,h_train] = sload(name1);
    [X1_s, X2_s] = extract_bciIV2a_train(s_train, h_train);
    
    
    save_name = ['D:\My Computer\SACSP codes\IV2a_preprocessed\S',int2str(i),'_train.mat'];
    save(save_name, 'X1_s', 'X2_s')
    
    
%     % number of trials in one class
%     num_tmp = size(X1_s,1);
%     left_train = zeros(size(X1_s,2), size(X1_s,3), num_tmp);
%     right_train = zeros(size(X2_s,2), size(X2_s,3), num_tmp);
%     for idx = 1:num_tmp
%         left_train(:,:,idx) = X1_s(idx,:,:);
%         right_train(:,:,idx) = X2_s(idx,:,:);
%     end
%     save_name = ['../BCICIV_2a_preprocessed/S',int2str(i),'_train.mat'];
%     save(save_name, 'left_train', 'right_train')
end

%% save the preprocessed test data
for j = 1:9
    name2 = ['../BCICIV_2a_gdf/true_labels/A0',int2str(j),'E.mat'];
    name3 = ['../BCICIV_2a_gdf/A0',int2str(j),'E.gdf'];
    test_label = load(name2).classlabel;
    [s_test,h_test] = sload(name3);
    [X_test_t, y_test_t] = extract_bciIV2a_test(s_test, h_test, test_label);
    
    X1_t = X_test_t(y_test_t == 1,:,:);
    X2_t = X_test_t(y_test_t == 2,:,:);
    
    save_name = ['D:\My Computer\SACSP codes\IV2a_preprocessed\S',int2str(j),'_test.mat'];
    save(save_name, 'X1_t', 'X2_t')
    
%     num_tmp2 = size(X1_t,1);
%     left_test = zeros(size(X1_t,2), size(X1_t,3), num_tmp2);
%     right_test = zeros(size(X2_t,2), size(X2_t,3), num_tmp2);
%     for idx = 1:num_tmp2
%         left_test(:,:,idx) = X1_t(idx,:,:);
%         right_test(:,:,idx) = X2_t(idx,:,:);
%     end
%     save_name = ['../BCICIV_2a_preprocessed/S',int2str(j),'_test.mat'];
%     save(save_name, 'left_test', 'right_test')
end
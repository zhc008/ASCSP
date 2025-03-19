% TRCSP_main.m
sub_names = {'a','l','v','w','y'};
alpha = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7];
for i = 1:5 %loop through the 5 subjects
   name1 = ['../data/data_set_IVa_a',sub_names{i},'.mat'];
   name2 = ['../data/true_labels_a',sub_names{i},'.mat'];
   load(name1)
   load(name2)
   [X1, X2, X_test, y_test] = extract_data(mrk, cnt, test_idx, true_y);
   [W,~,~] = CSP(X1, X2);
   best_j = 1;
   min_error = 1;
   for j = 1:7
       fold = 10;
       split_1 = floor(size(X1,1)/fold);
       split_2 = floor(size(X2,1)/fold);
       X1_cv = X1(1:fold*split_1,:,:);
       X2_cv = X2(1:fold*split_2,:,:);
       rng('default')
       shuffle_idx1 = randperm(size(X1_cv,1));
       shuffle_idx2 = randperm(size(X2_cv,1));
       X1_cv = X1_cv(shuffle_idx1,:,:);
       X2_cv = X2_cv(shuffle_idx2,:,:);
       X1_idx = 1:size(X1_cv,1);
       X2_idx = 1:size(X2_cv,1);

       X1_idx = reshape(X1_idx, fold, split_1);
       X2_idx = reshape(X2_idx, fold, split_2);
       valid_error = zeros(fold, 1);
       for k = 1:fold
           valid_1 = X1_idx(k, :);
           valid_2 = X2_idx(k, :);
           buffer = 1:fold;
           trainIdx_1 = reshape(X1_idx(buffer~=k,:).',1,[]);
           trainIdx_2 = reshape(X2_idx(buffer~=k,:).',1,[]);
           X1_t = X1_cv(trainIdx_1,:,:);
           X1_v = X1_cv(valid_1,:,:);
           X2_t = X2_cv(trainIdx_2,:,:);
           X2_v = X2_cv(valid_2,:,:);
           [W_tr,~,~] = TRCSP(alpha(j), X1_t, X2_t);
%            [W_tr,~,~] = CSP(X1_t, X2_t);
           f_X1_t = features(W_tr, X1_t);
           f_X2_t = features(W_tr, X2_t);
           f_X1_v = features(W_tr, X1_v);
           f_X2_v = features(W_tr, X2_v);
           f_X_v = [f_X1_v;f_X2_v];
           y_v = ones(size(f_X1_v, 1)+size(f_X2_v, 1),1);
           y_v((size(f_X1_v,1)+1):end) = y_v((size(f_X1_v,1)+1):end)*2;
           classified_y_test_cv = simpleLDA(f_X1_t, f_X2_t, f_X_v);
           valid_error(k) = sum(abs(y_v - classified_y_test_cv))/size(y_v, 1);
       end
       cv_error = min(valid_error);
%        disp(cv_error)
       if cv_error < min_error
           min_error = cv_error;
           best_j = j;
       end
   end
   disp(best_j)
   [W_tr,sigma1,sigma2] = TRCSP(alpha(best_j), X1, X2);
   f_X1 = features(W, X1);
   f_X2 = features(W, X2);
   f_X_test = features(W, X_test);
   f_X1_tr = features(W_tr, X1);
   f_X2_tr = features(W_tr, X2);
   f_X_test_tr = features(W_tr, X_test);
   classified_y_test = simpleLDA(f_X1, f_X2, f_X_test);
   test_err = sum(abs(y_test - classified_y_test))/size(y_test, 1);
   classified_y_test_tr = simpleLDA(f_X1_tr, f_X2_tr, f_X_test_tr);
   test_err_tr = sum(abs(y_test - classified_y_test_tr))/size(y_test, 1);
   fprintf(sub_names{i})
   fprintf(': %f\n', 1-test_err)
   fprintf(sub_names{i})
   fprintf('_tr: %f\n', 1-test_err_tr)
end
sub_names = {'a','l','v','w','y'};
alpha = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7];
for i = 1 %loop through the 5 subjects
   name1 = ['../data/data_set_IVa_a',sub_names{i},'.mat'];
   name2 = ['../data/true_labels_a',sub_names{i},'.mat'];
   load(name1)
   load(name2)
   [X1, X2, X_test, y_test] = extract_data(mrk, cnt, test_idx, true_y);
   [W,~,~] = CSP(X1, X2);
   [W_tr,sigma1,sigma2] = TRCSP(alpha(best_j), X1, X2);
   f_X1 = features(W, X1);
   f_X2 = features(W, X2);
   for j = 1:5
       name1_t = ['../data/data_set_IVa_a',sub_names{j},'.mat'];
       name2_t = ['../data/true_labels_a',sub_names{j},'.mat'];
       load(name1_t)
       load(name2_t)
       [~, ~, X_test_t, y_test_t] = extract_data(mrk, cnt, test_idx, true_y);
       f_X_test_t = features(W, X_test_t);
       classified_y_test = simpleLDA(f_X1, f_X2, f_X_test_t);
       test_err = sum(abs(y_test_t - classified_y_test))/size(y_test_t, 1);
       fprintf(sub_names{j})
       fprintf(': %f\n', 1-test_err)
       fprintf(sub_names{j})
       fprintf('_tr: %f\n', 1-test_err_tr)
   end
   name1 = ['../data/data_set_IVa_a',sub_names{i},'.mat'];
   name2 = ['../data/true_labels_a',sub_names{i},'.mat'];
   load(name1)
   load(name2)
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
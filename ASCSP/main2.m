% main2.m
sub_names = {'a','l','v','w','y'};
for i = 1:5 %loop through the 5 subjects
   name1 = ['../data/data_set_IVa_a',sub_names{i},'.mat'];
   name2 = ['../data/true_labels_a',sub_names{i},'.mat'];
   load(name1)
   load(name2)
   [X1_s, X2_s, X_test_s, y_test_s] = extract_data(mrk, cnt, test_idx, true_y);
   [W,~,~] = CSP(X1_s, X2_s);
   f_X1 = features(W, X1_s);
   f_X2 = features(W, X2_s);
   for j = 1:5
       name1_t = ['../data/data_set_IVa_a',sub_names{j},'.mat'];
       name2_t = ['../data/true_labels_a',sub_names{j},'.mat'];
       load(name1_t)
       load(name2_t)
       [X1_t, X2_t, X_test_t, y_test_t] = extract_data(mrk, cnt, test_idx, true_y);
       f_X_t = features(W, X_test_t);
       classified_y_test = simpleLDA(f_X1, f_X2, f_X_t);
       test_err = sum(abs(y_test_t - classified_y_test))/size(y_test_t, 1);
       text = [sub_names{i},' to ',sub_names{j}];
       fprintf(text)
       fprintf(': %f\n', 1-test_err)
   end
end
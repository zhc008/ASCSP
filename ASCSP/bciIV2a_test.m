% CSP for bci IV 2a
acc_csp = zeros(9,9);
csp_per_class = 3;
for i = 1:9 %loop through the 9 subjects
    name1 = ['../BCICIV_2a_gdf/A0',int2str(i),'T.gdf'];
    [s_train,h_train] = sload(name1);
    [X1_s, X2_s] = extract_bciIV2a_train(s_train, h_train);
    [W,~,~] = CSP(X1_s, X2_s);
    f_X1 = features(W, X1_s);
    f_X2 = features(W, X2_s);
    name2 = ['../BCICIV_2a_gdf/true_labels/A0',int2str(i),'E.mat'];
    name3 = ['../BCICIV_2a_gdf/A0',int2str(i),'E.gdf'];
    test_label = load(name2).classlabel;
    [s_test,h_test] = sload(name3);
    [X_test_t, y_test_t] = extract_bciIV2a_test(s_test, h_test, test_label);
    f_test_t = features(W, X_test_t);
    classified_y_test = simpleLDA(f_X1, f_X2, f_test_t);
    test_err = sum(abs(y_test_t - classified_y_test))/size(y_test_t, 1);
    fprintf('s')
    fprintf(int2str(i))
    fprintf(': %f\n', 1-test_err)
    acc_csp(i,i) = 1-test_err;
end
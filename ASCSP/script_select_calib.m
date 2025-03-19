% this file chops train-val-test for all frequency bands  

clc
clear 

close all


%% extract data 

% freqsCell = {[0.5 10],[1,3],[2,5],[4 7],[6 10], [7 12], [10 15], [12 19], [18 25], [19 30], [25 35], [30 40]};


for i = 1:12 % for each subject
    rng('default');
for instance = 1:10
    
%     freqs = freqsCell{1} 
        
    load(['../Mahta_data/S',int2str(i),'_calib.mat'])
    % find the number of trials in the class with fewest trials 
    % number of channels * number of time samples * number of epochs
    NUM_a = min([size(aL_calib,3),size(aR_calib,3)]); 
    NUM_t = min([size(tL_calib,3),size(tR_calib,3)]);
    
    % randomly permute the classes 
    temp = randperm(size(aL_calib,3));
    vect_aL = temp(1:NUM_a);
    
    temp = randperm(size(aR_calib,3));
    vect_aR = temp(1:NUM_a);
    
    temp = randperm(size(tL_calib,3));
    vect_tL = temp(1:NUM_t);
    
    temp = randperm(size(tR_calib,3));
    vect_tR = temp(1:NUM_t);
    
    [m, n, ~] = size(aL_calib);
    aL_train = zeros(NUM_a, m, n);
    aR_train = zeros(NUM_a, m, n);
    tL_train = zeros(NUM_t, m, n);
    tR_train = zeros(NUM_t, m, n);
    
    for kl = 1:length(vect_aL)
        aL_train(kl,:,:) = aL_calib(:,:,vect_aL(kl));
        aR_train(kl,:,:) = aR_calib(:,:,vect_aR(kl));
    end
    for kl = 1:length(vect_tL)
        tL_train(kl,:,:) = tL_calib(:,:,vect_tL(kl));
        tR_train(kl,:,:) = tR_calib(:,:,vect_tR(kl));
    end
    
    train_NUM_a = floor(0.6 * NUM_a);
    train_NUM_t = floor(0.6 * NUM_t);
    
    aL_tr = aL_train(1:train_NUM_a,:,:);
    tL_tr = tL_train(1:train_NUM_t,:,:);
    aR_tr = aR_train(1:train_NUM_a,:,:);
    tR_tr = tR_train(1:train_NUM_t,:,:);
    aL_te = aL_train(1+train_NUM_a:end,:,:);
    tL_te = tL_train(1+train_NUM_t:end,:,:);
    aR_te = aR_train(1+train_NUM_a:end,:,:);
    tR_te = tR_train(1+train_NUM_t:end,:,:);
    
    % save the data 
    disp('saving data')
    save(['C:/Users/CZN/Documents/Balanced_mahta_data/Calib/S',num2str(i),'_I',num2str(instance),'_calib_split.mat'],...     
            'aL_tr','aR_tr','tL_tr','tR_tr',...
            'aL_te','aR_te','tL_te','tR_te')
    disp('Done!')
end

end

    
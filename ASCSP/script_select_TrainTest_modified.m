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
    
    % save the data 
    disp('saving data')
    save(['C:/Users/CZN/Documents/Balanced_mahta_data/S',num2str(i),'_I',num2str(instance),'_calib.mat'],...     
            'aL_train','aR_train','tL_train','tR_train')
    disp('Calib Done!')
        
end
for instance = 1:10
    
%     freqs = freqsCell{1} 
        
    load(['../Mahta_data/S',int2str(i),'_online.mat'])
    % find the number of trials in the class with fewest trials 
    % number of channels * number of time samples * number of epochs
    NUM_a = min([size(aL_onln,3),size(aR_onln,3)]); 
    NUM_t = min([size(tL_onln,3),size(tR_onln,3)]);
    
    % randomly permute the classes 
    temp = randperm(size(aL_onln,3));
    vect_aL = temp(1:NUM_a);
    
    temp = randperm(size(aR_onln,3));
    vect_aR = temp(1:NUM_a);
    
    temp = randperm(size(tL_onln,3));
    vect_tL = temp(1:NUM_t);
    
    temp = randperm(size(tR_onln,3));
    vect_tR = temp(1:NUM_t);
    
    [m, n, ~] = size(aL_onln);
    aL_test = zeros(NUM_a, m, n);
    aR_test = zeros(NUM_a, m, n);
    tL_test = zeros(NUM_t, m, n);
    tR_test = zeros(NUM_t, m, n);
    
    for kl = 1:length(vect_aL)
        aL_test(kl,:,:) = aL_onln(:,:,vect_aL(kl));
        aR_test(kl,:,:) = aR_onln(:,:,vect_aR(kl));
    end
    for kl = 1:length(vect_tL)
        tL_test(kl,:,:) = tL_onln(:,:,vect_tL(kl));
        tR_test(kl,:,:) = tR_onln(:,:,vect_tR(kl));
    end
    
    % save the data 
    disp('saving data')
    save(['C:/Users/CZN/Documents/Balanced_mahta_data/S',num2str(i),'_I',num2str(instance),'_onln.mat'],...     
            'aL_test','aR_test','tL_test','tR_test')
    disp('Online Done!')
   
        
end


 

end


    
% this file chops train-val-test for all frequency bands  

clc
clear 

close all


%% extract data 

freqsCell = {[0.5 10],[1,3],[2,5],[4 7],[6 10], [7 12], [10 15], [12 19], [18 25], [19 30], [25 35], [30 40]};


for i = 1:10 % for each subject
for instance = 1:10
    
    freqs = freqsCell{1} 
        
    load(['~/Documents/MATLAB/epoched-data/S',num2str(i),'freqs',num2str(freqs(1)),'_',num2str(freqs(2)),'.mat'],'prepData')
    % find the number of trials in the class with fewest trials 
    NUM = min([size(prepData.B_r,3),size(prepData.B_l,3), size(prepData.G_r,3),size(prepData.G_l,3)]); 
    
    train_NUM = floor(0.7 * NUM); 
    val_NUM = floor(0.1 * NUM); 
    test_NUM = floor(0.2 * NUM); 

%     size(prepData.G_l,3)
%     size(prepData.B_l,3)
%     size(prepData.G_r,3)
%     size(prepData.B_r,3)
    
    % randomly permute the classes 
    temp = randperm(size(prepData.B_r,3)) ;
    vect_train_Br = temp(1:train_NUM); 
    vect_val_Br = temp(train_NUM+1:train_NUM+val_NUM);
    vect_test_Br = temp(train_NUM+val_NUM+1:train_NUM+val_NUM+test_NUM);
    
    temp = randperm(size(prepData.G_r,3)) ;
    vect_train_Gr = temp(1:train_NUM); 
    vect_val_Gr = temp(train_NUM+1:train_NUM+val_NUM);
    vect_test_Gr = temp(train_NUM+val_NUM+1:train_NUM+val_NUM+test_NUM);
    
    temp = randperm(size(prepData.B_l,3));
    vect_train_Bl = temp(1:train_NUM); 
    vect_val_Bl = temp(train_NUM+1:train_NUM+val_NUM);
    vect_test_Bl = temp(train_NUM+val_NUM+1:train_NUM+val_NUM+test_NUM);
    
    temp = randperm(size(prepData.G_l,3)) ;
    vect_train_Gl = temp(1:train_NUM); 
    vect_val_Gl = temp(train_NUM+1:train_NUM+val_NUM);
    vect_test_Gl = temp(train_NUM+val_NUM+1:train_NUM+val_NUM+test_NUM);
    
    [m, n, p] = size(prepData.G_r)
    
    train_Gr = zeros(length(freqsCell), train_NUM, m, n);
    train_Br = zeros(length(freqsCell), train_NUM, m, n);
    train_Gl = zeros(length(freqsCell), train_NUM, m, n);
    train_Bl = zeros(length(freqsCell), train_NUM, m, n);
    
    test_Gr = zeros(length(freqsCell), test_NUM, m, n);
    test_Br = zeros(length(freqsCell), test_NUM, m, n);
    test_Gl = zeros(length(freqsCell), test_NUM, m, n);
    test_Bl = zeros(length(freqsCell), test_NUM, m, n);
    
    val_Gr = zeros(length(freqsCell), val_NUM, m, n);
    val_Br = zeros(length(freqsCell), val_NUM, m, n);
    val_Gl = zeros(length(freqsCell), val_NUM, m, n);
    val_Bl = zeros(length(freqsCell), val_NUM, m, n);
    
    % for each frequency band 
    
    for jk = 1:length(freqsCell) 
        
        freqs = freqsCell{jk}
        
        load(['~/Documents/MATLAB/epoched-data/S',num2str(i),'freqs',num2str(freqs(1)),'_',num2str(freqs(2)),'.mat'],'prepData')
    
        for kl = 1:length(vect_train_Br)
            train_Br(jk,kl,:,:) = prepData.B_r(:,:,vect_train_Br(kl));  
            train_Gr(jk,kl,:,:) = prepData.G_r(:,:,vect_train_Gr(kl)); 
            train_Bl(jk,kl,:,:) = prepData.B_l(:,:,vect_train_Bl(kl));  
            train_Gl(jk,kl,:,:) = prepData.G_l(:,:,vect_train_Gl(kl)); 
        end
        
        for kl = 1:length(vect_test_Br)
            test_Br(jk,kl,:,:) = prepData.B_r(:,:,vect_test_Br(kl));  
            test_Gr(jk,kl,:,:) = prepData.G_r(:,:,vect_test_Gr(kl)); 
            test_Bl(jk,kl,:,:) = prepData.B_l(:,:,vect_test_Bl(kl));  
            test_Gl(jk,kl,:,:) = prepData.G_l(:,:,vect_test_Gl(kl)); 
        end
        
        for kl = 1:length(vect_val_Br)
            val_Br(jk,kl,:,:) = prepData.B_r(:,:,vect_val_Br(kl));  
            val_Gr(jk,kl,:,:) = prepData.G_r(:,:,vect_val_Gr(kl)); 
            val_Bl(jk,kl,:,:) = prepData.B_l(:,:,vect_val_Bl(kl));  
            val_Gl(jk,kl,:,:) = prepData.G_l(:,:,vect_val_Gl(kl)); 
        end

    end
    
    % save the data 
    disp('saving data')
    save(['~/Documents/MATLAB/data-TT/S',num2str(i),'_I',num2str(instance),'.mat'],...     
            'train_Br','train_Gr','train_Bl','train_Gl',...
            'test_Br','test_Gr','test_Bl','test_Gl',...
            'val_Br','val_Gr','val_Bl','val_Gl')
   disp('Done!')
   
        
end


 

end


    
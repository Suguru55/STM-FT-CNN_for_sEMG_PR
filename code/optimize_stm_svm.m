function optimize_stm_svm(config)

%%%%%%%%%%%%%%%%%%%%%%%%%
% set search parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%
C_candidate = [10^-3, 10^-2, 10^-1, 10^0, 10^1, 10^2, 10^3];
kernel_para_candidate = [10^-3, 10^-2, 10^-1, 10^0, 10^1, 10^2, 10^3];
beta_candidate = 0:0.2:3;
gamma_candidate = 0:0.2:3;

for dataset_ind = 1:length(config.dataset_names)
    %%%%%%%%%%%%%%%%%%%%
    % set local config %
    %%%%%%%%%%%%%%%%%%%%
    data_dir = [config.data_dir, '\', config.dataset_names{dataset_ind}];
    code_dir = config.code_dir;
    sub_num = config.sub_num(dataset_ind);
    mov_num = config.mov_num(dataset_ind);
    trial_num = config.trial_num(dataset_ind);
    k = 5;        % 5-fold CV
    nb_init = 15;  % find center of cluster for each class
    
    %%%%%%%%%%%%%
    % load data %
    %%%%%%%%%%%%%
    cd(data_dir);
    load('F_c.mat');
    cd(code_dir);
    feat_dim = size(F{1,1,1},2);
    
    %%%%%%%%%%
    % buffer %
    %%%%%%%%%%
    svm_acc_lib = zeros(sub_num, length(C_candidate), length(kernel_para_candidate));
    acc_transfered = zeros(sub_num, length(beta_candidate), length(gamma_candidate));
    local_z_mu = zeros(sub_num, feat_dim);
    local_z_sigma = zeros(sub_num, feat_dim);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % train optimized individual SVMs %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for sub_ind = 1:sub_num
        % preparation
        data = []; label = [];
        for trial_ind = 1:trial_num
            for mov_ind = 1:mov_num
                data = [data; F{sub_ind, trial_ind, mov_ind}];
                label = [label; c{sub_ind, trial_ind, mov_ind}];
            end
        end
        
        random_ind = randperm(length(data));
    
        % grid search
        for C_ind = 1:length(C_candidate)
            for kernel_para_ind = 1:length(kernel_para_candidate)
                
                cmd = ['-q -s 0 -t 2 -c ', num2str(C_candidate(C_ind)), '-g ', num2str(kernel_para_candidate(kernel_para_ind))]; % RBF kernel

                % 5-fold cv
                Acc = zeros(1,k);
            
                parfor cv_ind = 1:k
                    % separation
                    part_ind = random_ind(1+ceil(length(random_ind)/k)*(cv_ind-1):ceil((length(random_ind)/k)+(length(random_ind)/k)*(cv_ind-1)));
                    data_dammy = data;
                    val_data = data_dammy(part_ind,:);
                    data_dammy(part_ind,:) = [];
                    train_data = data_dammy;
        
                    label_dammy = label;
                    val_label = label_dammy(part_ind,:);
                    label_dammy(part_ind,:) = [];
                    train_label = label_dammy;
        
                    % normalization
                    [ZF, z_mu_temp, z_sigma_temp] = zscore(train_data);
                    val_data = (val_data - z_mu_temp) ./ z_sigma_temp;
                        
                    % SVM
                    model = svmtrain(train_label, ZF, cmd);
                    [pred, ~, ~] = svmpredict(val_label, val_data, model, '-q');
                    Acc(cv_ind) = sum(pred==val_label)/length(val_label);
                end
            
                svm_acc_lib(sub_ind, C_ind, kernel_para_ind) = mean(Acc);
            end
        end
    
        disp(['svm acc dataset', num2str(dataset_ind), ': grid search (svm) sub ', num2str(sub_ind), ' done'])
    end

    % find best parameters (global)
    mean_svm_acc = squeeze(mean(svm_acc_lib,1));
    [~, ind] = max(mean_svm_acc(:));
    [ii, jj] = ind2sub(size(mean_svm_acc),ind);
    best_C = C_candidate(ii);
    best_kernel_para = kernel_para_candidate(jj);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % train individual SVMs %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    SVMs = [];
    
    for sub_ind = 1:sub_num
        % preparation
        data = []; label = [];
        for trial_ind = 1:trial_num
            for mov_ind = 1:mov_num
                data = [data; F{sub_ind, trial_ind, mov_ind}];
                label = [label; c{sub_ind, trial_ind, mov_ind}];
            end
        end
        
        cmd = ['-q -s 0 -t 2 -b 1 -c ', num2str(best_C), '-g ', num2str(best_kernel_para)]; % RBF kernel (with probability estimates)
    
        % normalization
        [ZF, local_z_mu(sub_ind,:), local_z_sigma(sub_ind,:)] = zscore(data);
        
        % store trained models
        SVMs = [SVMs; svmtrain(label, ZF, cmd)];
    end

    disp(['svm dataset', num2str(dataset_ind), ': train ', num2str(sub_num), ' SVMs done'])
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % train optimized STM parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for sub_ind = 1:sub_num
        % preparation
        sub_ind_seq = 1:1:sub_num;
        sub_ind_seq(sub_ind) = [];       
        S_cal = []; L_cal = [];
        S_val = []; L_val = [];
        
        if dataset_ind == 1
            for mov_ind = 1:mov_num
                S_cal = [S_cal; F{sub_ind, 1, mov_ind}]; % 1st trial
                L_cal = [L_cal; c{sub_ind, 1, mov_ind}];
            end
        
            for mov_ind = 1:mov_num
                S_val = [S_val; F{sub_ind, 2, mov_ind}]; % 2nd trial
                L_val = [L_val; c{sub_ind, 2, mov_ind}];
            end
        else
            for trial_ind = 1:2
                for mov_ind = 1:mov_num
                    S_cal = [S_cal; F{sub_ind, trial_ind, mov_ind}]; % 1st and 2nd trials
                    L_cal = [L_cal; c{sub_ind, trial_ind, mov_ind}];
                end
            end
        
            for trial_ind = 3:4
                for mov_ind = 1:mov_num
                    S_val = [S_val; F{sub_ind, trial_ind, mov_ind}]; % 3rd and 4th trials
                    L_val = [L_val; c{sub_ind, trial_ind, mov_ind}];
                end
            end 
        end
    
        [nb_cal, ~] = size(S_cal);
        [nb_val, ~] = size(S_val);

        %%%%%%%%%%%%%%%%%%%%%%
        % weight calculation %
        %%%%%%%%%%%%%%%%%%%%%%
        % calculate performance of each SVM classifier for calibration data
        weights = zeros(sub_num-1, 1);
    
        for i = 1:sub_num-1
            % normalization
            S_cal_dammy = (S_cal - local_z_mu(sub_ind_seq(i),:)) ./ local_z_sigma(sub_ind_seq(i),:);
        
            % SVM
            [pred, ~, ~] = svmpredict(L_cal, S_cal_dammy, SVMs(sub_ind_seq(i)), '-q');
            weights(i) = sum(pred==L_cal)/length(L_cal);
        end

        % style transfer mapping
        for beta_ind = 1:length(beta_candidate)
            beta = beta_candidate(beta_ind);
                
            for gamma_ind = 1:length(gamma_candidate)
                gamma = gamma_candidate(gamma_ind);
               
                transfered_prob = zeros(length(L_val), mov_num);
                
                for nb_ind = 1:sub_num-1
                    % nomalization
                    S_cal_dammy = (S_cal - local_z_mu(sub_ind_seq(nb_ind), :)) ./ local_z_sigma(sub_ind_seq(nb_ind), :);
                    S_val_dammy = (S_val - local_z_mu(sub_ind_seq(nb_ind), :)) ./ local_z_sigma(sub_ind_seq(nb_ind), :);
                    
                    % collect source data
                    source = []; source_L = [];
                    for trial_ind = 1:trial_num
                        for mov_ind = 1:mov_num
                            source = [source; F{sub_ind_seq(nb_ind), trial_ind, mov_ind}];
                            source_L = [source_L; c{sub_ind_seq(nb_ind), trial_ind, mov_ind}];
                        end
                    end
                    
                    % transfer to make S_val more familiar to source SVM
                    train_x = zscore(source);
                    train_y = source_L;
                    
                    [S_transfered,~,~] = supervised_STM(train_x, train_y, S_cal_dammy, L_cal, S_val_dammy, nb_init, mov_num, beta, gamma);
                    [~, ~, temp_prob] = svmpredict([L_cal; L_val], S_transfered, SVMs(sub_ind_seq(nb_ind)), '-q -b 1');
                    temp_prob = temp_prob(nb_cal+1:nb_cal+nb_val,:);
                    % weighted ensemble strategy
                    transfered_prob = transfered_prob + temp_prob*weights(nb_ind);
                end
                
                [~, pred_transfered_temp] = max(transfered_prob');
                acc_transfered(sub_ind, beta_ind, gamma_ind) = sum(pred_transfered_temp'==L_val)/length(L_val); 
            end
        end    
        
        disp(['svm dataset', num2str(dataset_ind), ': grid search (STM) sub ', num2str(sub_ind), ' done'])      
    end

    % find best parameters (global)
    mean_acc_transfered = squeeze(mean(acc_transfered,1));
    [~, ind] = max(mean_acc_transfered(:));
    [ii, jj] = ind2sub(size(mean_acc_transfered),ind);
    best_beta = beta_candidate(ii);
    best_gamma = gamma_candidate(jj);
    
    disp(['Best acc: ', num2str(mean_acc_transfered(ii,jj))]);

    % save optimization results
    cd(data_dir);
    filename = ['best_parameters_stm_svm_ds', num2str(dataset_ind), '.mat'];
    save(filename, 'best_C','best_kernel_para','best_beta','best_gamma','SVMs','acc_transfered');
    cd(code_dir);
end
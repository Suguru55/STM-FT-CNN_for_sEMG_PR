function evaluate_stm_svm(config)

for dataset_ind = 1:length(config.dataset_names)
    
    %%%%%%%%%%%%%%%%%%%%
    % set local config %
    %%%%%%%%%%%%%%%%%%%%
    data_dir = [config.data_dir, '\', config.dataset_names{dataset_ind}];
    code_dir = config.code_dir;
    save_dir = config.save_dir;
    sub_num = config.sub_num(dataset_ind);
    mov_num = config.mov_num(dataset_ind);
    trial_num = config.trial_num(dataset_ind);
    nb_init = 15;  % find center of cluster for each class

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load features, labels, and optimized parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    filename = ['best_parameters_stm_svm_ds', num2str(dataset_ind), '.mat'];
    cd(data_dir);
    load(['F_c.mat']);
    load(filename);
    cd(code_dir);
    feat_dim = size(F{1,1,1},2);
    C = best_C; kernel_para = best_kernel_para;
    beta = best_beta; gamma = best_gamma;
    
    %%%%%%%%%%
    % buffer %
    %%%%%%%%%%
    acc_svm = zeros(1, sub_num);
    acc_stm_svm = zeros(1, sub_num);
    pred_svm = cell(1, sub_num);
    pred_stm_svm = cell(1, sub_num);
    local_z_mu = zeros(sub_num, feat_dim);
    local_z_sigma = zeros(sub_num, feat_dim);

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
        
        cmd = ['-q -s 0 -t 2 -b 1 -c ', num2str(C), '-g ', num2str(kernel_para)]; % RBF kernel (with probability estimates)
    
        % normalization
        [ZF, local_z_mu(sub_ind,:), local_z_sigma(sub_ind,:)] = zscore(data);
        
        % store trained models
        SVMs = [SVMs; svmtrain(label, ZF, cmd)];
    end
    
    disp(['svm acc dataset', num2str(dataset_ind), ': train ', num2str(sub_num), ' SVMs done'])
    
    %%%%%%%%%%%%%%
    % evaluation %
    %%%%%%%%%%%%%%
    for sub_ind = 1:sub_num
        % preparation
        sub_ind_seq = 1:1:sub_num;
        sub_ind_seq(sub_ind) = [];       
        S_cal = []; L_cal = [];
        S_tes = []; L_tes = [];      
        
        if dataset_ind == 1
            for mov_ind = 1:mov_num
                S_cal = [S_cal; F{sub_ind, 1, mov_ind}]; % 1st trial
                L_cal = [L_cal; c{sub_ind, 1, mov_ind}];
            end
        
            for trial_ind = 3:5
               for mov_ind = 1:mov_num
                    S_tes = [S_tes; F{sub_ind, trial_ind, mov_ind}]; % 3rd to 5th trial
                    L_tes = [L_tes; c{sub_ind, trial_ind, mov_ind}];
                end
            end
        else
            for trial_ind = 1:2
                for mov_ind = 1:mov_num
                    S_cal = [S_cal; F{sub_ind, trial_ind, mov_ind}]; % 1st and 2nd trials
                    L_cal = [L_cal; c{sub_ind, trial_ind, mov_ind}];
                end
            end
        
            for trial_ind = 5:6
                for mov_ind = 1:mov_num
                    S_tes = [S_tes; F{sub_ind, trial_ind, mov_ind}]; % 5th and 6th trials
                    L_tes = [L_tes; c{sub_ind, trial_ind, mov_ind}];
                end
            end 
        end
               
        [nb_cal, ~] = size(S_cal);
        [nb_tes, ~] = size(S_tes);
        
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
        naive_prob = zeros(length(L_tes), mov_num);
        transfered_prob = zeros(length(L_tes), mov_num);

        for nb_ind = 1:sub_num-1
            % nomalization
            S_cal_dammy = (S_cal - local_z_mu(sub_ind_seq(nb_ind), :)) ./ local_z_sigma(sub_ind_seq(nb_ind), :);
            S_tes_dammy = (S_tes - local_z_mu(sub_ind_seq(nb_ind), :)) ./ local_z_sigma(sub_ind_seq(nb_ind), :);
                    
            % naive performance
            [~, ~, temp_prob] = svmpredict(L_tes, S_tes_dammy, SVMs(sub_ind_seq(nb_ind)), '-q -b 1');
            naive_prob = naive_prob + temp_prob*weights(nb_ind);
            
            % style transfer mapping
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
                    
            [S_transfered,~,~] = supervised_STM(train_x, train_y, S_cal_dammy, L_cal, S_tes_dammy, nb_init, mov_num, beta, gamma);
            [~, ~, temp_prob] = svmpredict([L_cal; L_tes], S_transfered, SVMs(sub_ind_seq(nb_ind)), '-q -b 1');
            temp_prob = temp_prob(nb_cal+1:nb_cal+nb_tes,:);
            transfered_prob = transfered_prob + temp_prob*weights(nb_ind);
        end
        
        % recognition
        [~, pred_naive_temp] = max(naive_prob');
        pred_svm{sub_ind} = pred_naive_temp;
        acc_svm(sub_ind) = sum(pred_naive_temp'==L_tes)/length(L_tes);
        
        [~, pred_transfered_temp] = max(transfered_prob');
        pred_stm_svm{sub_ind} = pred_transfered_temp;
        acc_stm_svm(sub_ind) = sum(pred_transfered_temp'==L_tes)/length(L_tes); 
        
        disp(['dataset ', num2str(dataset_ind), ', sub ', num2str(sub_ind)])
        disp(['SVM acc = ', num2str(acc_svm(sub_ind)), ', STM-SVM acc = ', num2str(acc_stm_svm(sub_ind))]);
    end

    %%%%%%%%%%%%%%%%
    % save results %
    %%%%%%%%%%%%%%%%
    cd(save_dir);
    filename = ['results_stm_svm_ds', num2str(dataset_ind)];
    save(filename,'acc_svm', 'acc_stm_svm',...
                  'pred_svm', 'pred_stm_svm','best_beta','best_gamma',...
                  'best_C', 'best_kernel_para');
    cd(code_dir);
end
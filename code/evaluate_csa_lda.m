function evaluate_csa_lda(config)

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
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load features, labels, and optimized parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    filename = ['best_parameters_csa_lda_ds', num2str(dataset_ind), '.mat'];
    cd(data_dir);
    load(['F_c.mat']);
    load(filename);
    cd(code_dir);
    feat_dim = size(F{1,1,1},2);
    tau = best_tau;
    lambda = best_lambda;
    
    %%%%%%%%%%
    % buffer %
    %%%%%%%%%%
    acc_lda = zeros(1, sub_num);
    acc_csa_lda = zeros(1, sub_num);
    pred_lda = cell(1, sub_num);
    pred_csa_lda = cell(1, sub_num);
    local_z_mu = zeros(sub_num, feat_dim);
    local_z_sigma = zeros(sub_num, feat_dim);

    %%%%%%%%%%%%%%%%%%%%%%%%%
    % train individual LDAs %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    LDAs = cell(1,sub_num);
    
    for sub_ind = 1:sub_num
        % preparation
        data = []; label = [];
        for trial_ind = 1:trial_num
            for mov_ind = 1:mov_num
                data = [data; F{sub_ind, trial_ind, mov_ind}];
                label = [label; c{sub_ind, trial_ind, mov_ind}];
            end
        end
        
        % normalization
        [ZF, local_z_mu(sub_ind,:), local_z_sigma(sub_ind,:)] = zscore(data);
        
        % store trained models
        LDAs{sub_ind} =  fitcdiscr(ZF, label, 'DiscrimType', 'pseudoLinear');
    end
    
    disp(['lda dataset', num2str(dataset_ind), ': train ', num2str(sub_num), ' LDAs done'])
    
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
        
        [ZF, ~, ~]= zscore(S_cal);
        model_targ =  fitcdiscr(ZF, L_cal, 'DiscrimType', 'pseudoLinear');
        
        %%%%%%%%%%%%%%%%%%%%%%
        % weight calculation %
        %%%%%%%%%%%%%%%%%%%%%%
        % calculate performance of each LDA classifier for calibration data
        weights = zeros(sub_num-1, 1);
        
        for i = 1:sub_num-1
            % normalization
            S_cal_dammy = (S_cal - local_z_mu(sub_ind_seq(i), :)) ./ local_z_sigma(sub_ind_seq(i), :);
             
            % LDA
            pred = predict(LDAs{sub_ind_seq(i)}, S_cal_dammy);        
            weights(i) = sum(pred==L_cal)/length(L_cal); 
        end
        
        % use selected sources (we already know which data is similar to the target data by optimization process)
        naive_prob = zeros(length(L_tes), mov_num);
        transfered_prob = zeros(length(L_tes), mov_num);
    
        for nb_ind = 1:sub_num-1
            % normalization
            S_tes_dammy = (S_tes - local_z_mu(sub_ind_seq(nb_ind),:)) ./ local_z_sigma(sub_ind_seq(nb_ind),:);
  
            % naive performance
            naive_mu = LDAs{sub_ind_seq(nb_ind)}.Mu;
            naive_sigma = LDAs{sub_ind_seq(nb_ind)}.Sigma;
        
            for j = 1:mov_num
                temp_prob = S_tes_dammy*pinv(naive_sigma')*naive_mu(j,:)' - (1/2)*naive_mu(j,:)*pinv(naive_sigma')*naive_mu(j,:)' - log(2);
                naive_prob(:, j) = naive_prob(:, j) + temp_prob*weights(nb_ind);
            end
            
            % covariate shift adaptation    
            transfered_mu = (1-tau).*naive_mu + tau.*model_targ.Mu;
            transfered_sigma = (1-lambda).*naive_sigma + lambda.*model_targ.Sigma;
        
            for j = 1:mov_num
                temp_prob = transfered_prob(:, j) + S_tes_dammy*pinv(transfered_sigma')*transfered_mu(j,:)' - (1/2)*transfered_mu(j,:)*pinv(transfered_sigma')*transfered_mu(j,:)' - log(2);
                transfered_prob(:, j) = transfered_prob(:, j) + temp_prob*weights(nb_ind);
            end 
        end
    
        % recognition
        [~, pred_naive_temp] = max(naive_prob');
        pred_lda{sub_ind} = pred_naive_temp;
        acc_lda(sub_ind) = sum(pred_naive_temp'==L_tes)/length(L_tes);      

        [~, pred_transfered_temp] = max(transfered_prob');
        pred_csa_lda{sub_ind} = pred_transfered_temp;
        acc_csa_lda(sub_ind) = sum(pred_transfered_temp'==L_tes)/length(L_tes); 
        
        disp(['dataset ', num2str(dataset_ind), ', sub ', num2str(sub_ind)])
        disp(['LDA acc = ', num2str(acc_lda(sub_ind)), ', CSA-LDA acc = ', num2str(acc_csa_lda(sub_ind))]);
    end

    %%%%%%%%%%%%%%%%
    % save results %
    %%%%%%%%%%%%%%%%
    cd(save_dir);
    filename = ['results_csa_lda_ds', num2str(dataset_ind)];
    save(filename,'acc_lda', 'acc_csa_lda',...
                  'pred_lda', 'pred_csa_lda', 'best_lambda', 'best_tau');
    cd(code_dir);
end
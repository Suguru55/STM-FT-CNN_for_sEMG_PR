function optimize_csa_lda(config)

%%%%%%%%%%%%%%%%%%%%%%%%%
% set search parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%
tau_candidate = 0:0.1:1;
lambda_candidate = 0:0.1:1;

for dataset_ind = 1:length(config.dataset_names)
    %%%%%%%%%%%%%%%%%%%%
    % set local config %
    %%%%%%%%%%%%%%%%%%%%
    data_dir = [config.data_dir, '\', config.dataset_names{dataset_ind}];
    code_dir = config.code_dir;
    sub_num = config.sub_num(dataset_ind);
    mov_num = config.mov_num(dataset_ind);
    trial_num = config.trial_num(dataset_ind);

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
    acc_transfered = zeros(sub_num, length(tau_candidate), length(lambda_candidate));
    LDAs = cell(1,sub_num);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % train individual LDAs %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % LDA does not need CV procedure
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
        ZF = zscore(data);
        
        % store trained classifiers
        LDAs{sub_ind} =  fitcdiscr(ZF, label, 'DiscrimType', 'pseudoLinear');
    end
    
    disp(['lda dataset', num2str(dataset_ind), ': train ', num2str(sub_num), ' LDAs done'])

    %%%%%%%%%%%%%%%
    % grid search %
    %%%%%%%%%%%%%%%
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
        
        [ZF, mu, sigma]= zscore(S_cal);
        model_targ =  fitcdiscr(ZF, L_cal, 'DiscrimType', 'pseudoLinear');
        S_val_dammy = (S_val - mu) ./ sigma;           
        
        %%%%%%%%%%%%%%%%%%%%%%
        % weight calculation %
        %%%%%%%%%%%%%%%%%%%%%%
        % calculate performance of each LDA classifier for calibration data
        weights = zeros(sub_num-1, 1);
        
        for i = 1:sub_num-1
            % LDA
            pred = predict(LDAs{sub_ind_seq(i)}, ZF);        
            weights(i) = sum(pred==L_cal)/length(L_cal); 
        end
        
        % covariate shift adaptation
        for tau_ind = 1:length(tau_candidate)        
            for lambda_ind = 1:length(lambda_candidate)
                transfered_prob = zeros(length(L_val), mov_num);
            
                for nb_ind = 1:sub_num-1
                    transfered_mu = (1-tau_candidate(tau_ind)).*LDAs{sub_ind_seq(nb_ind)}.Mu + tau_candidate(tau_ind).*model_targ.Mu;
                    transfered_sigma = (1-lambda_candidate(lambda_ind)).*LDAs{sub_ind_seq(nb_ind)}.Sigma + lambda_candidate(lambda_ind).*model_targ.Sigma;

                    for j = 1:mov_num
                        temp_prob = S_val_dammy*pinv(transfered_sigma')*transfered_mu(j,:)' - (1/2)*transfered_mu(j,:)*pinv(transfered_sigma')*transfered_mu(j,:)' - log(2);
                        % weighted ensemble strategy
                        transfered_prob(:, j) = transfered_prob(:, j) + temp_prob*weights(nb_ind);
                    end 
                end
                
                [~, pred_transfered_temp] = max(transfered_prob');
                acc_transfered(sub_ind, tau_ind, lambda_ind) = sum(pred_transfered_temp'==L_val)/length(L_val); 
            end
        end
    
        disp(['lda dataset', num2str(dataset_ind), ': grid search (CSA) sub ', num2str(sub_ind), ' done'])
    end

    % find best parameters (global)
    mean_acc_transfered = squeeze(mean(acc_transfered,1));
    [~, ind] = max(mean_acc_transfered(:));
    [ii, jj] = ind2sub(size(mean_acc_transfered),ind);
    best_tau = tau_candidate(ii);
    best_lambda = lambda_candidate(jj);

    disp(['Best acc: ', num2str(mean_acc_transfered(ii,jj))]);
    
    % save optimization results
    cd(data_dir);
    filename = ['best_parameters_csa_lda_ds', num2str(dataset_ind), '.mat'];
    save(filename,'best_tau','best_lambda','LDAs','acc_transfered');
    cd(code_dir);
end
function optimize_stm_cnn(config)

%%%%%%%%%%%%%%%%%%%%%%%%%
% set search parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%
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
    nb_init = 15;  % find center of cluster for each class
    miniBatchSize = 64;
    
    %%%%%%%%%%%%%
    % load data %
    %%%%%%%%%%%%%
    cd(data_dir);
    if exist(['trained_cnn_ds', num2str(dataset_ind), '.mat']) ~= 0
        disp(['dataset', num2str(dataset_ind), ': already trained'])
    else
        load('F_c.mat');
        cd(code_dir);
        feat_dim = size(F_map{1,1,1},1);
        map_len = size(F_map{1,1,1},2);
    
        %%%%%%%%%%%%%%%%%%%%
        % CNN architecture %
        %%%%%%%%%%%%%%%%%%%%
        layers = [
            imageInputLayer([feat_dim, map_len, 1], 'Name', 'input')
      
            % Block 1
            convolution2dLayer([3 3], 32, 'Stride', 1, 'Name', 'conv_1', 'WeightsInitializer','he')
            batchNormalizationLayer('Name', 'bn_1')
            reluLayer('Name', 'relu_1')
      
            convolution2dLayer([3 3], 32, 'Stride', 1, 'Name', 'conv_2', 'WeightsInitializer','he')
            batchNormalizationLayer('Name', 'bn_2')
            reluLayer('Name', 'relu_2')
      
            averagePooling2dLayer([3 3], 'Name', 'avepool_1');
      
            % Block 2
            convolution2dLayer([5 5], 64, 'Stride', 1, 'Name', 'conv_3', 'WeightsInitializer','he')
            batchNormalizationLayer('Name', 'bn_3')
            reluLayer('Name', 'relu_3')
      
            convolution2dLayer([5 5], 64, 'Stride', 1, 'Name', 'conv_4', 'WeightsInitializer','he')
            batchNormalizationLayer('Name', 'bn_4')
            reluLayer('Name', 'relu_4')
      
            averagePooling2dLayer([3 3], 'Name', 'avepool_2');      
      
            % Block 3
            convolution2dLayer([1 1], 32, 'Stride', 1, 'Name', 'conv_5', 'WeightsInitializer','he')
            batchNormalizationLayer('Name', 'bn_5')
            reluLayer('Name', 'relu_5')
      
            fullyConnectedLayer(mov_num, 'Name','fc')
            dropoutLayer(0.5, 'Name','drop')
            softmaxLayer('Name','softmax')
            classificationLayer('Name','output')
        ];

        %analyzeNetwork(layers);
    
        %%%%%%%%%%
        % buffer %
        %%%%%%%%%%
        local_z_mu = zeros(sub_num, feat_dim);
        local_z_sigma = zeros(sub_num, feat_dim);
        CNNs = cell(1, sub_num);
        training_order_lib = cell(1, sub_num);
        validation_order_lib = cell(1, sub_num);
        info = cell(1, sub_num);
    
        %%%%%%%%%%%%%%%%%%%%%%%%%
        % train individual CNNs %
        %%%%%%%%%%%%%%%%%%%%%%%%%
        for sub_ind = 1:sub_num
            % preparation
            data = []; label = [];
            for trial_ind = 1:trial_num
                for mov_ind = 1:mov_num
                    if isempty(data)
                        data = F_map{sub_ind, trial_ind, mov_ind};
                        label = c_map{sub_ind, trial_ind, mov_ind};
                    else
                        data = cat(3, data, F_map{sub_ind, trial_ind, mov_ind});
                        label = [label; c_map{sub_ind, trial_ind, mov_ind}];
                    end
                end
            end
        
            % separate training and validation data (80% training, 20% validation)
            order = randperm(length(label));
            split_size = fix(length(order)/5);
        
            validation_order = sort(order(1:split_size));
            order(1:split_size) = [];
            training_order = sort(order);
        
            train_data = data(:,:,training_order);
            val_data = data(:,:,validation_order);
        
            train_label = label(training_order,:);
            val_label = label(validation_order,:); % Check if all classes have data.
        
            training_order_lib{sub_ind} = training_order;
            validation_order_lib{sub_ind} = validation_order;
        
            % normalization (by feature dimension)
            [ZF, local_z_mu(sub_ind,:), local_z_sigma(sub_ind,:)] = zscore(train_data,0,[2,3]); 
            val_data = (val_data - local_z_mu(sub_ind,:)') ./ local_z_sigma(sub_ind,:)';
        
            % reshape to 4D matrix (height, width, channel, data size)
            XTrain = zeros(feat_dim, map_len, 1, size(ZF,3));
            XVal = zeros(feat_dim, map_len, 1, size(val_data,3));
        
            XTrain(:,:,1,:) = ZF;
            XVal(:,:,1,:) = val_data;
        
            % CNN
            options = trainingOptions('sgdm', ...
                'MiniBatchSize',64,...
                'MaxEpoch',50,...
                'InitialLearnRate',0.0001,...
                'Shuffle','every-epoch',...
                'ValidationData',{XVal, val_label},...
                'Verbose',false,...
                'ExecutionEnvironment','auto');
        
            [CNNs{sub_ind}, info{sub_ind}] = trainNetwork(XTrain, train_label, layers, options);
            sub_ind
            info{sub_ind}
        end
    
        disp(['cnn dataset', num2str(dataset_ind), ': train ', num2str(sub_num), ' CNNs done'])
    
        % save invidual CNN models
        cd(data_dir);
        filename = ['trained_cnn_ds', num2str(dataset_ind), '.mat'];
        save(filename, 'CNNs','info','training_order_lib','validation_order_lib','layers','local_z_mu','local_z_sigma','layers');
        cd(code_dir);
    end
end

for dataset_ind = 1:length(config.dataset_names)
    %%%%%%%%%%%%%%%%%%%%
    % set local config %
    %%%%%%%%%%%%%%%%%%%%
    data_dir = [config.data_dir, '\', config.dataset_names{dataset_ind}];
    code_dir = config.code_dir;
    sub_num = config.sub_num(dataset_ind);
    mov_num = config.mov_num(dataset_ind);
    trial_num = config.trial_num(dataset_ind);
    nb_init = 15;  % find center of cluster for each class
    miniBatchSize = 64;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load features, labels, and optimized parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cd(data_dir);
    load(['F_c.mat']);
    load(['trained_cnn_ds', num2str(dataset_ind), '.mat']);
    cd(code_dir);
    feat_dim = size(F_map{1,1,1},1);
    map_len = size(F_map{1,1,1},2); 
    
    %%%%%%%%%%
    % buffer %
    %%%%%%%%%%
    acc_transfered = zeros(sub_num, length(beta_candidate), length(gamma_candidate));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % train optimized STM parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for sub_ind = 1:sub_num
        % preparation
        sub_ind_seq = 1:1:sub_num;
        sub_ind_seq(sub_ind) = [];       
        S_map_cal = []; L_map_cal = [];
        S_map_val = []; L_map_val = [];
        S_vec_cal = []; L_vec_cal = [];
        S_vec_val = []; L_vec_val = [];
        
        if dataset_ind == 1
            for mov_ind = 1:mov_num
                if isempty(S_map_cal)
                    S_map_cal = F_map{sub_ind, 1, mov_ind}; % 1st trial
                    L_map_cal = c_map{sub_ind, 1, mov_ind};
                else
                    S_map_cal = cat(3, S_map_cal, F_map{sub_ind, 1, mov_ind});
                    L_map_cal = [L_map_cal; c_map{sub_ind, 1, mov_ind}];
                end
                
                S_vec_cal = [S_vec_cal; F{sub_ind, 1, mov_ind}];
                L_vec_cal = [L_vec_cal; c{sub_ind, 1, mov_ind}];
            end
        
            for mov_ind = 1:mov_num
                if isempty(S_map_val)
                    S_map_val = F_map{sub_ind, 2, mov_ind}; % 2nd trial
                    L_map_val = c_map{sub_ind, 2, mov_ind};
                else
                    S_map_val = cat(3, S_map_val, F_map{sub_ind, 2, mov_ind});
                    L_map_val = [L_map_val; c_map{sub_ind, 2, mov_ind}];
                end
                
                S_vec_val = [S_vec_val; F{sub_ind, 2, mov_ind}];
                L_vec_val = [L_vec_val; c{sub_ind, 2, mov_ind}];
            end
        else
            for trial_ind = 1:2
                for mov_ind = 1:mov_num
                    if isempty(S_map_cal)
                        S_map_cal = F_map{sub_ind, trial_ind, mov_ind}; % 1st and 2nd trials
                        L_map_cal = c_map{sub_ind, trial_ind, mov_ind};
                    else
                        S_map_cal = cat(3, S_map_cal, F_map{sub_ind, trial_ind, mov_ind});
                        L_map_cal = [L_map_cal; c_map{sub_ind, trial_ind, mov_ind}];
                    end
                    
                    S_vec_cal = [S_vec_cal; F{sub_ind, trial_ind, mov_ind}];
                    L_vec_cal = [L_vec_cal; c{sub_ind, trial_ind, mov_ind}];
                end
            end
        
            for trial_ind = 3:4
                for mov_ind = 1:mov_num
                    if isempty(S_map_val)
                        S_map_val = F_map{sub_ind, trial_ind, mov_ind}; % 3rd and 4th trials
                        L_map_val = c_map{sub_ind, trial_ind, mov_ind};
                    else
                        S_map_val = cat(3, S_map_val, F_map{sub_ind, trial_ind, mov_ind});
                        L_map_val = [L_map_val; c_map{sub_ind, trial_ind, mov_ind}];
                    end
                    
                    S_vec_val = [S_vec_val; F{sub_ind, trial_ind, mov_ind}];
                    L_vec_val = [L_vec_val; c{sub_ind, trial_ind, mov_ind}];
                end
            end 
        end
    
        nb_val = size(S_map_val,3);
        
        %%%%%%%%%%%%%%%%%%%%%%
        % weight calculation %
        %%%%%%%%%%%%%%%%%%%%%%
        % calculate performance of each CNN classifier for calibration data
        weights = zeros(sub_num-1, 1);
    
        for i = 1:sub_num-1
            % normalization
            S_cal_dammy = (S_map_cal - local_z_mu(sub_ind_seq(i),:)') ./ local_z_sigma(sub_ind_seq(i),:)';
            
            % reshape to 4D matrix (height, width, channel, data size)
            XCal = zeros(feat_dim, map_len, 1, size(S_cal_dammy,3));
            XCal(:,:,1,:) = S_cal_dammy;
        
            % CNN
            pred = classify(CNNs{sub_ind_seq(i)}, XCal, 'ExecutionEnvironment','auto', 'MiniBatchSize',miniBatchSize);
            weights(i) = sum(pred==L_map_cal)/length(L_map_cal);
        end

        % style transfer mapping
        for beta_ind = 1:length(beta_candidate)
            beta = beta_candidate(beta_ind);
                
            for gamma_ind = 1:length(gamma_candidate)
                gamma = gamma_candidate(gamma_ind);
               
                transfered_prob = zeros(length(L_map_val), mov_num);
                
                for nb_ind = 1:sub_num-1
                    % collect source data
                    source = []; source_L = [];
                    for trial_ind = 1:trial_num
                        for mov_ind = 1:mov_num
                            source = [source; F{sub_ind_seq(nb_ind), trial_ind, mov_ind}];
                            source_L = [source_L; c{sub_ind_seq(nb_ind), trial_ind, mov_ind}];
                        end
                    end
                    
                    % transfer to make S_val more familiar to source CNN
                    [train_x, mu, sigma] = zscore(source);
                    train_y = source_L;
                    
                    % nomalization
                    S_vec_cal_dammy = (S_vec_cal - mu) ./ sigma;
                    S_vec_val_dammy = (S_vec_val - mu) ./ sigma;
                    
                    S_map_val_dammy = (S_map_val - local_z_mu(sub_ind_seq(nb_ind),:)') ./ local_z_sigma(sub_ind_seq(nb_ind),:)';
                  
                    [~,A0,b0] = supervised_STM(train_x, train_y, S_vec_cal_dammy, L_vec_cal, S_vec_val_dammy, nb_init, mov_num, beta, gamma);
                    
                    % transform S_map by A0 and b0
                    S_map_val_trans = zeros(feat_dim, map_len, nb_val); 
                    
                    for i = 1:nb_val
                        for j = 1:map_len
                            S_map_val_trans(:,j,i) = A0*S_map_val_dammy(:,j,i) + b0;
                        end
                    end
                    
                    % reshape to 4D matrix (height, width, channel, data size)
                    XTransfer = zeros(feat_dim, map_len, 1, size(S_map_val_trans, 3));
                    XTransfer(:,:,1,:) = S_map_val_trans;
                    
                    temp_prob = activations(CNNs{sub_ind_seq(nb_ind)}, XTransfer, 'softmax','OutputAs','rows');
                    % weighted ensemble strategy
                    transfered_prob = transfered_prob + temp_prob*weights(nb_ind);
                end
                
                [~, pred_transfered_temp] = max(transfered_prob');
                acc_transfered(sub_ind, beta_ind, gamma_ind) = sum(categorical(pred_transfered_temp)'==L_map_val)/length(L_map_val); 
            end
        end    
        
        disp(['cnn dataset', num2str(dataset_ind), ': grid search (STM) sub ', num2str(sub_ind), ' done'])      
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
    filename = ['best_parameters_stm_cnn_ds', num2str(dataset_ind), '.mat'];
    save(filename, 'best_beta','best_gamma','acc_transfered');
    cd(code_dir);
end
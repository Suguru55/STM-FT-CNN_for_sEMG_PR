function evaluate_stm_ft_cnn(config)

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
    miniBatchSize = 64;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load features, labels, and optimized parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cd(data_dir);
    load(['F_c.mat']);
    load(['trained_cnn_ds', num2str(dataset_ind), '.mat']);
    load(['best_parameters_stm_ft_cnn_ds', num2str(dataset_ind), '.mat']);
    cd(code_dir);
    feat_dim = size(F_map{1,1,1},1);
    map_len = size(F_map{1,1,1},2);
    beta = best_beta; gamma = best_gamma;
    
    %%%%%%%%%%
    % buffer %
    %%%%%%%%%%
    acc_stm_ft_cnn = zeros(1, sub_num);
    pred_stm_ft_cnn = cell(1, sub_num);

    %%%%%%%%%%%%%%
    % evaluation %
    %%%%%%%%%%%%%%
    for sub_ind = 1:sub_num
        % preparation
        sub_ind_seq = 1:1:sub_num;
        sub_ind_seq(sub_ind) = [];       
        S_map_cal = []; L_map_cal = [];
        S_map_tes = []; L_map_tes = [];   
        S_vec_cal = []; L_vec_cal = [];

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
        
            for trial_ind = 3:5
               for mov_ind = 1:mov_num
                   if isempty(S_map_tes)
                        S_map_tes = F_map{sub_ind, 2, mov_ind}; % 3rd to 5th trial
                        L_map_tes = c_map{sub_ind, 2, mov_ind};
                   else
                        S_map_tes = cat(3, S_map_tes, F_map{sub_ind, 2, mov_ind});
                        L_map_tes = [L_map_tes; c_map{sub_ind, 2, mov_ind}];
                   end
                end
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
        
            for trial_ind = 5:6
                for mov_ind = 1:mov_num
                    if isempty(S_map_tes)
                        S_map_tes = F_map{sub_ind, trial_ind, mov_ind}; % 5th and 6th trials
                        L_map_tes = c_map{sub_ind, trial_ind, mov_ind};
                    else
                        S_map_tes = cat(3, S_map_tes, F_map{sub_ind, trial_ind, mov_ind});
                        L_map_tes = [L_map_tes; c_map{sub_ind, trial_ind, mov_ind}];
                    end
                end
            end 
        end
        
        nb_cal = size(S_map_cal,3);
        nb_tes = size(S_map_tes,3);

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
        
        % fine-tuning
        transfered_prob = zeros(length(L_map_tes), mov_num);

        for nb_ind = 1:sub_num-1
            % collect source data
            source = []; source_L = [];
            for trial_ind = 1:trial_num
                for mov_ind = 1:mov_num
                    source = [source; F{sub_ind_seq(nb_ind), trial_ind, mov_ind}];
                    source_L = [source_L; c{sub_ind_seq(nb_ind), trial_ind, mov_ind}];
                end
            end
            
            % transfer to make S_tes more familiar to source CNN
            [train_x, mu, sigma] = zscore(source);
            train_y = source_L;
                    
            % nomalization    
            S_vec_cal_dammy = (S_vec_cal - mu) ./ sigma;
            
            S_map_cal_dammy = (S_map_cal - local_z_mu(sub_ind_seq(nb_ind),:)') ./ local_z_sigma(sub_ind_seq(nb_ind),:)';
            S_map_tes_dammy = (S_map_tes - local_z_mu(sub_ind_seq(nb_ind),:)') ./ local_z_sigma(sub_ind_seq(nb_ind),:)';

            % style transfer mapping
            [~,A0,b0] = supervised_STM(train_x, train_y, S_vec_cal_dammy, L_vec_cal, [], nb_init, mov_num, beta, gamma);
               
            % transform S_map by A0 and b0
            S_map_cal_trans = zeros(feat_dim, map_len, nb_cal);
            S_map_tes_trans = zeros(feat_dim, map_len, nb_tes); 

            for i = 1:nb_cal
                for j = 1:map_len
                    S_map_cal_trans(:,j,i) = A0*S_map_cal_dammy(:,j,i) + b0;
                end
            end
            
            for i = 1:nb_tes
                for j = 1:map_len
                    S_map_tes_trans(:,j,i) = A0*S_map_tes_dammy(:,j,i) + b0;
                end
            end
            
            % reshape to 4D matrix (height, width, channel, data size)
            XCal_transfer = zeros(feat_dim, map_len, 1, size(S_map_cal_trans, 3));
            XTes_transfer = zeros(feat_dim, map_len, 1, size(S_map_tes_trans, 3));
                    
            XCal_transfer(:,:,1,:) = S_map_cal_trans;
            XTes_transfer(:,:,1,:) = S_map_tes_trans;  
                 
            % prepare pre-trained CNNs
            layersTransfer = CNNs{sub_ind_seq(nb_ind)}.Layers(1:end-4);
            
            % freeze weights of pre-trained layers
            for ii = 1:size(layersTransfer,1)
                props = properties(layersTransfer(ii));
                for p = 1:numel(props)
                    propName = props{p};
                    if ~isempty(regexp(propName, 'WeightLearnRateFactor$', 'once'))
                        layersTransfer(ii).(propName) = 0;
                    end
                    
                    if ~isempty(regexp(propName, 'BiasLearnRateFactor$', 'once'))
                        layersTransfer(ii).(propName) = 0;
                    end
                end
            end
            
            % set new layers for fine-tuning
            ft_layers = [
                layersTransfer
                fullyConnectedLayer(mov_num,'Name','fc')
                dropoutLayer(0.5, 'Name','drop')
                softmaxLayer('Name','softmax')
                classificationLayer('Name','output')
            ];

            options = trainingOptions('sgdm', ...
                'MiniBatchSize',64,...
                'MaxEpoch',10,...
                'InitialLearnRate',0.0001,...
                'Shuffle','every-epoch',...
                'Verbose',false,...
                'ExecutionEnvironment','auto');
            
            % fine-tuning
            netTransfer = trainNetwork(XCal_transfer, L_map_cal, ft_layers, options);     
            temp_prob = activations(netTransfer, XTes_transfer, 'softmax','OutputAs','rows');
            
            % weighted ensemble strategy
            transfered_prob = transfered_prob + temp_prob*weights(nb_ind);
        end
        
        % recognition
        [~, pred_transfered_temp] = max(transfered_prob');
        pred_ft_cnn{sub_ind} = pred_transfered_temp;
        acc_stm_ft_cnn(sub_ind) = sum(categorical(pred_transfered_temp)'==L_map_tes)/length(L_map_tes); 
        
        disp(['dataset ', num2str(dataset_ind), ', sub ', num2str(sub_ind)])
        disp(['STM-FT-CNN acc = ', num2str(acc_stm_ft_cnn(sub_ind))]);
    end

    %%%%%%%%%%%%%%%%
    % save results %
    %%%%%%%%%%%%%%%%
    cd(save_dir);
    filename = ['results_stm_ft_cnn_ds', num2str(dataset_ind)];
    save(filename,'acc_stm_ft_cnn','pred_stm_ft_cnn');
    cd(code_dir);
end
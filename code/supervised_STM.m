function [S_transfered, A0, b0] = supervised_STM(train_x, train_y, S_cal, L_cal, S_val, nb_init, mov_num, beta_coeff, gamma_coeff)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Style transfer mapping in supervised way

% train_x, train_y       : used to find prototypes and train subject-free classifiers
% beta_coeff, gamma_coeff: hyper-parameters to control the tradeoff between
%                          nontransfer and overtransfer (0:0.2:3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set local config
[nb_cal, m] = size(S_cal);
[nb_val, ~] = size(S_val);
T_cal = zeros(nb_cal, m);        % target of labeled data
f_cal = ones(nb_cal,1);          % confidence of labeled data (in supervised STM, all values are 1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find prototype (cluster center) of each class %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m_lib = cell(1, mov_num);
stream = RandStream('mlfg6331_64');  % Random number stream
opts = statset('UseParallel', 1, 'Streams',stream);

parfor i = 1:mov_num
    [id, ~] = find(train_y(:)==i);
    temp_train_x = train_x(id,:);
    [~, m_lib{i}] = kmeans(temp_train_x, nb_init, 'MaxIter', 1000, 'Replicates', 10, 'Option', opts);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% learn a supervised STM {A0, b0} with labeled data %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parfor i = 1:nb_cal
    temp_S_cal = S_cal(i,:);
    T_cal(i,:) = find_target(temp_S_cal, squeeze(m_lib{L_cal(i)}));
end

[A0, b0] = calculate_A_b(S_cal, T_cal, f_cal, beta_coeff, gamma_coeff);

% transform all S data by A0 and b0
S_cal_trans = zeros(nb_cal, m);    % labeled data transformed by A0, b0

for i = 1:nb_cal
    S_cal_trans(i,:) = (A0*S_cal(i,:)' + b0)';
end

if nb_val == 0
    S_transfered = S_cal_trans;
else
    S_val_trans = zeros(nb_val, m);    % unlabeled data transformed by A0, b0

    for i = 1:nb_val
        S_val_trans(i,:) = (A0*S_val(i,:)' + b0)';
    end
    
    S_transfered = [S_cal_trans; S_val_trans];
end
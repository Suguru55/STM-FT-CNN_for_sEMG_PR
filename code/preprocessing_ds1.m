function preprocessing_ds1(config)

% set local config
dataset_ind = 1;

data_dir = config.data_dir;
code_dir = config.code_dir;
sub_num = config.sub_num(dataset_ind);
trial_num = config.trial_num(dataset_ind);
mov_num = config.mov_num(dataset_ind);
fil_order = config.fil_order;
fs = config.fs(dataset_ind);
fs_highpass = config.fs_highpass(dataset_ind);
dataset_names = config.dataset_names{dataset_ind};
analysis_win = config.analysis_len*fs;
shift_win = config.shift_len*fs;
dim = 2;                   % dimension for modified sample entropy
samp_win = 8;              % window size for modified sample entropy (40 ms)
samp_th = 0.4;             % threshold for normalized samp_en

% buffer
F = cell(sub_num, trial_num, mov_num);
F_map = cell(sub_num, trial_num, mov_num);
c = cell(sub_num, trial_num, mov_num);
c_map = cell(sub_num, trial_num, mov_num);

% design fifth-order Buttorworth bandpass filter (cutoff freq.: [15, 700] Hz) for EMG
n = fil_order;
Wn_high = fs_highpass/(fs/2);
[b_high, a_high] = butter(n, Wn_high, 'high');

% get folder information
cd([data_dir, '/', dataset_names]);
folders = dir;
folders = folders(~ismember({folders.name}, {'.', '..'}));
isdir = cell2mat({folders.isdir});
folders(isdir==0) = [];

for sub_ind = 1:length(folders)
    for mov_ind = 1:mov_num
        for trial_ind = 1:trial_num
            % load data
            cd([data_dir, '/', dataset_names, '/', folders(sub_ind).name]);
            eval(sprintf('filename = [''M%dT%d.csv'']', mov_ind, trial_ind));
            data = csvread(filename);
            cd(code_dir);
            
            % apply highpass filter
            for i = 1:size(data,2)
                data(:, i) = filtfilt(b_high, a_high, data(:, i));
            end

            % extract epochs by modified multiscale sample entropy
            if mov_ind == 1
                % if motion label is 'rest', onset detection can be avoided 
                epoch = data(1+fs:fs+fs*1.5, :);
            else
                % detect onset point from 1 to 3 s
                dammy_data = data(1+fs*1:fs+fs*2, :);
                samp_en = zeros(size(data,2), length(dammy_data)-samp_win);
            
                for ch_ind = 1:size(data,2)
                    for n = 1:length(dammy_data)-samp_win
                        input = dammy_data(1+(n-1):samp_win+(n-1), ch_ind);  % 5ms shifting
                        r = 0.25 * std(input);                               % tolerance
                    
                        correl = zeros(1, 2);
                        inputMat = zeros(dim+1, samp_win-dim);
                    
                        for i = 1:dim+1
                            inputMat(i, :) = input(i:samp_win-dim+i-1);
                        end
                    
                        for m = dim:dim+1
                            count = zeros(1, samp_win-dim);
                            tempMat = inputMat(1:m, :);
                       
                            for i = 1:samp_win-m
                                % calculate Chebyshev distance extcuding
                                % self-matching case
                                dist = max(abs(tempMat(:, i+1:samp_win-dim)-repmat(tempMat(:,i), 1, samp_win-dim-i)));
                           
                                % calculate Heaviside function of the distance
                                D = 1./(1 + exp((dist-0.5)/r)); % Sigmoid function
                                count(i) = sum(D)/(samp_win-dim);
                            end
                            correl(m-dim+1) = sum(count)/(samp_win-dim);
                        end
                    
                        samp_en(ch_ind, n) = log(correl(1)/correl(2));
                    end
                end
            
                [maxvec, ~] = max(samp_en);
                [maxval, column] = max(maxvec);
                temp_samp = samp_en(samp_en(:, column)==maxval,:);
                    
                [A, ~] = max(temp_samp, [], 2);
                temp_samp = temp_samp./repmat(A, 1, length(temp_samp));
                    
                [~, col] = find(temp_samp > samp_th);
                if(size(col, 1)==0)
                    col(1, 1) = 1; 
                end
                epoch = data(col(1, 1)+fs:fs+(fs*1.5-1)+col(1, 1), :); 
            end

            [feat, feat_map] = extract_feature(epoch, analysis_win, shift_win);
            class = ones(size(feat, 1), 1).*mov_ind;
            class_map = categorical(ones(size(feat_map, 3), 1).*mov_ind);
            
            F{sub_ind, trial_ind, mov_ind} = feat;
            F_map{sub_ind, trial_ind, mov_ind} = feat_map;
            c{sub_ind, trial_ind, mov_ind} = class;
            c_map{sub_ind, trial_ind, mov_ind} = class_map;
        end
    end
    disp(['preprocess dataset 1 sub ', num2str(sub_ind), ' done']);
end

cd([data_dir, '/', dataset_names]);
filename = 'F_c';
save(filename, 'F', 'F_map', 'c', 'c_map');
cd(code_dir);
function preprocessing_ds2(config)

% set local config
dataset_ind = 2;

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

% buffer
F = cell(sub_num, trial_num, mov_num);
F_map = cell(sub_num, trial_num, mov_num);
c = cell(sub_num, trial_num, mov_num);
c_map = cell(sub_num, trial_num, mov_num);

% design fifth-order Buttorworth bandpass filter (cutoff freq.: [15, 450] Hz) for EMG
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
    cd([data_dir, '/', dataset_names, '/', folders(sub_ind).name]);
    
    % load data
    files = dir;
    files = files(~ismember({files.name}, {'.', '..'}));

    for file_ind = 1:length(files)
        filename = [data_dir, '/', dataset_names, '/', folders(sub_ind).name, '/', files(file_ind).name];
        load(filename);
        sig = double(emg);
        trig = restimulus;
        cd(code_dir);
        
        % filtering
        for ch_ind = 1:size(sig,2)
            sig(:,ch_ind) = filtfilt(b_high, a_high, sig(:,ch_ind)); % highpass
        end
        
        % segmentation and feature extraction
        change_point = [];
        for j = 1:length(trig)-1
            if (trig(j) - trig(j+1))~=0
                change_point = [change_point; j+1];
            end
        end
        
        trial_counter = 1;
        mov_counter = 1;
       
        for k = 1:2:length(change_point)
            seg = sig(change_point(k):change_point(k+1),:);
            [feat, feat_map] = extract_feature(seg, analysis_win, shift_win);
            class = ones(size(feat, 1), 1).*mov_counter;
            class_map = categorical(ones(size(feat_map, 3), 1).*mov_counter);
            
            F{sub_ind, trial_counter, mov_counter} = feat;
            F_map{sub_ind, trial_counter, mov_counter} = feat_map;
            c{sub_ind, trial_counter, mov_counter} = class;
            c_map{sub_ind, trial_counter, mov_counter} = class_map;
            
            trial_counter = trial_counter + 1;
            
            if trial_counter == trial_num + 1
                trial_counter = 1;
                mov_counter = mov_counter + 1;
            end
        end
    end
    
    disp(['preprocess dataset 2 sub ', num2str(sub_ind), ' done']);
end

cd([data_dir, '/', dataset_names]);
filename = 'F_c';
save(filename, 'F', 'F_map', 'c', 'c_map');
cd(code_dir);
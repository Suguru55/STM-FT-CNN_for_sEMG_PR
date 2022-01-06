function struct = set_config(dir)

% path
struct.data_dir = [dir, '\data'];
struct.code_dir = [dir, '\code'];
struct.save_dir = [dir, '\results'];

% general
struct.dataset_names = {'private', 'NinaPro DB5 exerciseA', 'NinaPro DB5 exerciseB', 'NinaPro DB5 exerciseC'};
struct.fs = [200, 200, 200, 200];

struct.fs_highpass = [15, 15, 15, 15];
struct.fs_lowpass = [nan, nan, nan, nan];
struct.fil_order = 5;

struct.analysis_len = 0.25; % 0.25 s
struct.shift_len = 0.05;    % 0.05 s

struct.mov_num = [8, 12, 17, 23];
struct.sub_num = [25, 10, 10, 10];
struct.trial_num = [5, 6, 6, 6];

% transfer learning
struct.cv_num = 5;                % parameter tuning
struct.STM_iter_num = 5;
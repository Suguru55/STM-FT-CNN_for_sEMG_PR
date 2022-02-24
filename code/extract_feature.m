function [feature, feature_map] = extract_feature(data,win_size,win_inc)

if nargin < 3
    if nargin < 2
        win_size = 256;
    end
    win_inc = 32;
end

deadzone = 0.01;
feature1 = getrmsfeat(data,win_size,win_inc); % RMS
feature2 = getmavfeat(data,win_size,win_inc); % MAV
feature3 = getzcfeat(data,deadzone,win_size,win_inc); % ZC
feature4 = getsscfeat(data,deadzone,win_size,win_inc); % SSC
feature5 = getwlfeat(data,win_size,win_inc); % WL

ar_order = 6;
feature6 = getarfeat(data,ar_order,win_size,win_inc); % AR

feature = [feature1 feature2 feature3 feature4 feature5 feature6]; % 11-dimansional vector

% for CNN
feature1_map_temp = getrmsfeat(data,win_size,1);
feature2_map_temp = getmavfeat(data,win_size,1);
feature3_map_temp = getzcfeat(data,deadzone,win_size,1);
feature4_map_temp = getsscfeat(data,deadzone,win_size,1);
feature5_map_temp = getwlfeat(data,win_size,1);
feature6_map_temp = getarfeat(data,ar_order,win_size,1);

feature_map_temp = [feature1_map_temp feature2_map_temp feature3_map_temp feature4_map_temp feature5_map_temp feature6_map_temp];
feature_map_temp = feature_map_temp';

num_win = floor((size(feature_map_temp,2) - win_size)/win_inc)+1;
% Note that feeding hand-crafted features to CNNs requires a double onset: 
% an onset for producing features and an onset until the features are accumulated as a map
% (i.e.,  the amount of data is smaller than that of hand-crafted features.).
feature_map = zeros(size(feature_map_temp,1), win_size, num_win);

st = 1;
en = win_size;

for i = 1:num_win
   feature_map(:, :, i) = feature_map_temp(:,st:en);
   
   st = st + win_inc;
   en = en + win_inc;
end


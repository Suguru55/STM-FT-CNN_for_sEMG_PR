%% Clear workspace
clear all
close all

%% Set config
main_dir = 'C:\Users\eyed164\Desktop\STM-FT-CNN_for_sEMG_PR'; % change to your directory
config = set_config(main_dir);
addpath('C:\Users\eyed164\Desktop\libsvm-master\matlab');

%% Preprocessing
preprocessing_ds1(config); % private
preprocessing_ds2(config); % NinaPro DB5 exercise A
preprocessing_ds3(config); % NinaPro DB5 exercise B
preprocessing_ds4(config); % NinaPro DB5 exercise C

%% Evaluation
evaluate_csa_lda(config);
evaluate_stm_svm(config);
evaluate_stm_cnn(config);
evaluate_ft_cnn(config);
evaluate_stm_ft_cnn(config);

%% Visualization
visualize_results(config);
function visualize_results(config)

for dataset_ind = 1:length(config.dataset_names)
    disp('------------------------------------------------')
    disp(['Data: ', config.dataset_names{dataset_ind}]);
    
    %%%%%%%%%%%%%%%%%%%%
    % set local config %
    %%%%%%%%%%%%%%%%%%%%
    save_dir = config.save_dir;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load and caliculate average results %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cd(save_dir);
    
    % lda and csa-lda acc
    load(['results_csa_lda_ds', num2str(dataset_ind)]); 
    disp(['LDA acc: ', num2str(mean(acc_lda)), ', CSA-LDA acc: ', num2str(mean(acc_csa_lda))]);
    
    % svm and stm-svm acc
    load(['results_stm_svm_ds', num2str(dataset_ind)]); 
    disp(['SVM acc: ', num2str(mean(acc_svm)), ', STM-SVM acc: ', num2str(mean(acc_stm_svm))]);
    
    % cnn and stm-cnn acc
    load(['results_stm_cnn_ds', num2str(dataset_ind)]); 
    disp(['CNN acc: ', num2str(mean(acc_cnn)), ', STM-CNN acc: ', num2str(mean(acc_stm_cnn))]);
    
    % ft-cnn acc
    load(['results_ft_cnn_ds', num2str(dataset_ind)]); 
    disp(['FT-CNN acc: ', num2str(mean(acc_ft_cnn))]);
    
    % stm-ft-CNN acc
    load(['results_stm_ft_cnn_ds', num2str(dataset_ind)]); 
    disp(['STM-FT-CNN acc: ', num2str(mean(acc_stm_ft_cnn))]);
    disp('------------------------------------------------')
end

cd(config.code_dir);
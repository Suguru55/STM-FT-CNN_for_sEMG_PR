# STM-FT-CNN_for_sEMG_PR

This sample scripts are described in the paper "A Style Transfer Mapping and Fine-tuning Subject Transfer Framework Using Convolutional Neural Networks for Surface Electromyogram Pattern Recognition" accepted by <i>ICASSP2022</i>.<br />

__\<Description\>__<br />
After changing information about your directories in main.m (lines 6 and 8), downloading getxxfeat.m, and installing LIBSVM package, you can use this codes.<br />
This project has three folders:<br />
1. data <br />
   you can get MyoDatasets from <a href="https://github.com/Suguru55/SS-STM_for_MyoDatasets" target="_blank">here</a> and NinaPro DB5 from <a href="https://zenodo.org/record/1000116" target="_blank">here</a>. After downloading them, sepalate NinaPro DB5 mat files into their respective folders and delete txt files with the same name and put subXX folders of SS-STM_for_MyoDatasets to private folder.<br />
   - private (MyoDatasets)<br />
   - NinaPro DB5 exerciseA<br />
   - NinaPro DB5 exerciseB<br />
   - NinaPro DB5 exerciseC<br />

2. code<br />
   this folder has one main m.file that uses:<br />
   - set_config<br />
   - preprocessing_ds1<br />
        - extract_features<br />
        you can get the following m.files from <a href="http://www.sce.carleton.ca/faculty/chan/index.php?page=matlab" target="_blank">here</a><br />
            - getrmsfeat<br />
            - getmavfeat<br />
            - getzcfeat<br />
            - getsscfeat<br />
            - getwlfeat<br />
            - getarfeat<br />
    - preprocessing_ds2<br />
    - preprocessing_ds3<br />
    - preprocessing_ds4<br />
    - evaluate_csa_lda<br />
    - evaluate_stm_svm<br />
        you can make the m scripts, svmtrain.m and svmpredict.m from <a href="https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download" target="_blank">here</a><br />
        - svmtrain (LIBSVM)<br />
        - svmpredict (LIBSVM)<br />
        - supervised_STM<br />
            - find_target<br />
            - calculate_A_b<br />
    - evaluate_stm_cnn<br />
    - evaluate_ft_cnn<br />
    - evaluate_stm_ft_cnn<br />
    - visualize_results<br />
        
3. resuts<br />
   this folder will store results_xxx_acc_dsx.mat.<br />
   because the random seed is not fixed, the results of the methods combining STM and FT (i.e., STM-SVM, STM-CNN, FT-CNN, STM-FT-CNN) will be slightly different to the paper.<br />

__\<Environments\>__<br />
Windows 10<br />
MATLAB R2020a<br />
 1. Signal Processing Toolbox<br />
 2. Statics and Machine Learning Toolbox<br />
 3. Parallel Comupting Toolbox<br />
 4. Deep Learning Toolbox<br />

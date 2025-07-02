# Enhancing Vehicular Channel Estimation with LNNs

This project gives support to reproduce the results presented in the paper "Enhancing Vehicular Channel Estimation with Liquid Neural Networks". The main folder includes the Matlab codes for the vehicular communication scenario considered and the Python_Codes folder presents the training and testing phases of the proposed and benchmark estimators. 

The following instructions will guide the execution:
1) Main: Present the main simulation file. The user needs to define the simulation parameters (Speed, channel model, modulation order, [...]). Note that each execution is used to generate the data in a specific scenario. 
2) Datasets_Generation: This file generates the dataset used for training and testing the estimators compared (As default, 80% of data is for training and 20% of data is for testing). 
3) Python_Codes/[DNN_Training, LSTM_Training, CfC_Training, CfC_Training_WithoutRestriction]: Training phase for each of the methods compared, best model is saved for the proposed DPA-CfC and the benchmarks presented in [1, 2].
4) Python_Codes/[DNN_Testing, LSTM_Testing, CfC_Testing, CfC_Testing_WithoutRestriction]: The models are tested the results are saved in .mat files.
5) Results_Processing: Process the testing datasets and calculates the BER and NMSE results.
	 
Additional files:
- Channel_functions: Includes the vehicular channel models based on [3].
- DPA_Estimation: Presents the implementation of the data-pilot aided (DPA) procedure.
- Python_codes/tf_cfc: Presents the implementation of the liquid neural network based on the CfC layer as presented in [4].
- Python_codes/count_flops: Presents the analysis on the complexity of the estimators as summarized in Table II. 
- Python_codes/CfC_Optimization: Presents the optimization process discussed in section III.C.
- Plotting_Results: Plot the results (the results presented in the paper are available in this repo).

As summarised in the manuscript, the analysis below compares the computational complexity and per-inference efficiency of the evaluated models (measurements taken over 10^5 events on an Intel Core i7-8565U CPU):

----------DPA-DNN----------
Parameters: 10 124 | FLOPs: 20 044
Average energy per inference: 0.0543 J
Average power per inference: 62.9495 W
Average inference time: 0.0008 s
----------DPA-LSTM-NN----------
Parameters: 35 115 | FLOPs: 161 500
Average energy per inference: 0.1078 J
Average power per inference: 90.9389 W
Average inference time: 0.0012 s
----------DPA-CfC No restriction----------
Parameters: 77 186 | FLOPs: 370 600
Average energy per inference: 0.1067 J
Average power per inference: 73.7802 W
Average inference time: 0.0014 s
----------DPA-CfC----------
Parameters: 22 764 | FLOPs: 126 880
Average energy per inference: 0.0738 J
Average power per inference: 58.8712 W
Average inference time: 0.0012 s


[1] A. F. Dos Reis, Y. Medjahdi, B. S. Chang, J. Sublime, G. Brante, and C. F. Bader, “Low complexity LSTM-NN-based receiver for vehicular communications in the presence of high-power amplifier distortions,” IEEE Access, vol. 10, pp. 121 985–122 000, 2022.

[2] S. Han, Y. Oh, and C. Song, “A deep learning based channel estimation scheme for IEEE 802.11p systems,” in ICC 2019 - 2019 IEEE International Conference on Communications (ICC), 2019, pp. 1–6

[3] G. Acosta-Marum and M. A. Ingram, ‘‘Six time- and frequency-selective empirical channel models for vehicular wireless LANs,’’ IEEE Veh. Technol. Mag., vol. 2, no. 4, pp. 4–11, Dec. 2007.

[4] R. Hasani, M. Lechner, A. Amini, L. Liebenwein, A. Ray, M. Tschaikowski, G. Teschl, and D. Rus, “Closed-form continuous-time neural networks,” Nature Machine Intelligence, vol. 4, no. 11, pp. 992–1003, 2022

If you use any of these codes for research that results in publications, please cite our reference:
[...]

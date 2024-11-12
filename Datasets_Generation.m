clc
clearvars
close all
warning('off','all')

path = pwd;

mod = '16QAM';
ChType = 'VTV_UC';
v = 100;                    % Moving speed of user in km/h

nSym                    = 20;        % Number of symbols within one frame
SNR                     = 0:5:30;    % bit to noise ratio
N_CH                    = [20000;20000;20000;20000;20000;20000;100000]; % Number of channel realizations

pathdata = [num2str(nSym) 'Sym_' mod '_' ChType '_' num2str(v) 'kmh'];

% idx = randperm(max(N_CH));
% training_indices          = idx(1:floor(0.8*max(N_CH)));               % number of channel realizations for training
% testing_indices           = idx(length(training_indices)+1:end);  % number of channel realizations for testing
% save('indices','testing_indices','training_indices')
load('indices.mat')
Testing_Data_set_size     = size(testing_indices,2);
Training_Data_set_size    = size(training_indices,2);

% Define Simulation parameters
nUSC                      = 52;
algo                      = 'DPA';

Train_X                   = zeros(nUSC*2, Training_Data_set_size * nSym);
Train_Y                   = zeros(nUSC*2, Training_Data_set_size * nSym);
Test_X                    = zeros(nUSC*2, Testing_Data_set_size * nSym);
Test_Y                    = zeros(nUSC*2, Testing_Data_set_size * nSym);

for n_snr = 1:size(SNR,2)
    if N_CH(n_snr) == 2000 % Dataset for test
        % Load simulation data according to the defined configurations (Ch, mod, algorithm) 
        load(['data_' pathdata '\Simulation_' num2str(n_snr),'.mat'], 'True_Channels_Structure', [algo '_Structure']);
        Algo_Channels_Structure = eval([algo '_Structure']);
        
        Testing_DatasetX  =  Algo_Channels_Structure;
        Testing_DatasetY  =  True_Channels_Structure;
       
        % Expend Testing and Training Datasets
        Testing_DatasetX_expended  = reshape(Testing_DatasetX, nUSC, nSym * Testing_Data_set_size);
        Testing_DatasetY_expended  = reshape(Testing_DatasetY, nUSC, nSym * Testing_Data_set_size);
    
        Test_X(1:nUSC,:)                       = real(Testing_DatasetX_expended);
        Test_X(nUSC+1:2*nUSC,:)                = imag(Testing_DatasetX_expended);
        Test_Y(1:nUSC,:)                       = real(Testing_DatasetY_expended);
        Test_Y(nUSC+1:2*nUSC,:)                = imag(Testing_DatasetY_expended);
        
        % Save training and testing datasets to the DNN_Datasets structure
        Datasets.('Test_X')  =  Test_X;
        Datasets.('Test_Y')  =  Test_Y;
    
        % Save the DNN_Datasets structure to the specified folder in order to be used later in the Python code 
        save([path '\Python_Codes\data\Dataset_' num2str(n_snr)],  'Datasets');
        disp(['Data generated for ' algo ', SNR = ', num2str(SNR(n_snr))]);
    elseif  N_CH(n_snr) == 10000 % Dataset for training
        % Load simulation data according to the defined configurations (Ch, mod, algorithm) 
        load(['data_' pathdata '\Simulation_'  num2str(n_snr),'.mat'], 'True_Channels_Structure', [algo '_Structure']);
        Algo_Channels_Structure = eval([algo '_Structure']);
        
        Training_DatasetX =  Algo_Channels_Structure(:,:,training_indices);
        Training_DatasetY =  True_Channels_Structure(:,:,training_indices);
        Testing_DatasetX  =  Algo_Channels_Structure(:,:,testing_indices);
        Testing_DatasetY  =  True_Channels_Structure(:,:,testing_indices);
           
        % Expend Testing and Training Datasets
        Training_DatasetX_expended = reshape(Training_DatasetX, nUSC, nSym * Training_Data_set_size);
        Training_DatasetY_expended = reshape(Training_DatasetY, nUSC, nSym * Training_Data_set_size);
        Testing_DatasetX_expended  = reshape(Testing_DatasetX, nUSC, nSym * Testing_Data_set_size);
        Testing_DatasetY_expended  = reshape(Testing_DatasetY, nUSC, nSym * Testing_Data_set_size);
        
        % Complex to Real domain conversion
        Train_X(1:nUSC,:)                    = real(Training_DatasetX_expended);
        Train_X(nUSC+1:2*nUSC,:)             = imag(Training_DatasetX_expended);
        Train_Y(1:nUSC,:)                    = real(Training_DatasetY_expended);
        Train_Y(nUSC+1:2*nUSC,:)             = imag(Training_DatasetY_expended);
        
        Test_X(1:nUSC,:)                       = real(Testing_DatasetX_expended);
        Test_X(nUSC+1:2*nUSC,:)                = imag(Testing_DatasetX_expended);
        Test_Y(1:nUSC,:)                       = real(Testing_DatasetY_expended);
        Test_Y(nUSC+1:2*nUSC,:)                = imag(Testing_DatasetY_expended);
        
        % Save training and testing datasets to the DNN_Datasets structure
        Datasets.('Train_X') =  Train_X;
        Datasets.('Train_Y') =  Train_Y;
        Datasets.('Test_X')  =  Test_X;
        Datasets.('Test_Y')  =  Test_Y;
        
        % Save the DNN_Datasets structure to the specified folder in order to be used later in the Python code 
        save([path '\Python_Codes\data\Dataset_' num2str(n_snr)],  'Datasets');
        disp(['Data generated for ' algo ', SNR = ', num2str(n_snr)]);
    end
end
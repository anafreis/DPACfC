clc
clearvars
% close all
warning('off','all')

path = pwd;

mod = '64QAM';
ChType = 'VTV_UC';
v                       = 100;                    % Moving speed of user in km/h
N_CH                    = [20000;20000;20000;20000;20000;20000;100000]; % Number of channel realizations
nSym                    = 20;        % Number of symbols within one frame

pathdata = [num2str(nSym) 'Sym_' mod '_' ChType '_' num2str(v) 'kmh'];

% Loading Simulation Data
load(['data_' pathdata '\Simulation_variables.mat']);
%% ------ Bits Modulation Technique------------------------------------------
if(strcmp(mod,'QPSK') == 1)
     nBitPerSym            = 2; 
elseif (strcmp(mod,'16QAM') == 1)
     nBitPerSym            = 4; 
elseif (strcmp(mod,'64QAM') == 1)
     nBitPerSym            = 6; 
end
M                     = 2 ^ nBitPerSym; % QAM Modulation Order   
load('indices.mat');
N_Test_Frames = length(testing_indices);
EbN0dB                    = (0:5:30)';
Pow                       = mean(abs(qammod(0:(M-1),M)).^2); 
constlen                  = 7;
trellis                   = poly2trellis(constlen,[171 133]);
tbl                       = 34;
nDSC                      = 48;
nUSC                      = 52;
ppositions                = [7,21,32,46].';                           % Pilots positions in Kset

N_SNR                      = size(EbN0dB,1);
Phf                        = zeros(N_SNR,1);

Err_DPA_DNN           = zeros(N_SNR,1);
Err_DPA_LSTM          = zeros(N_SNR,1);
Err_DPA_LNN           = zeros(N_SNR,1);
Err_DPA_LNN_WR        = zeros(N_SNR,1);

Ber_DPA_DNN           = zeros(N_SNR,1);
Ber_DPA_LSTM          = zeros(N_SNR,1);
Ber_DPA_LNN           = zeros(N_SNR,1);
Ber_DPA_LNN_WR        = zeros(N_SNR,1);

dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].';    % Data positions in the set of allocated subcarriers Kset 

for ii = 1:N_SNR 
    i = ii;

    % Loading Simulation Parameters Results
    load(['data_' pathdata '\Simulation_' num2str(i) '.mat']);

    % Loading DPA-DNN Results
    load([path '\Python_Codes\data\DPA_DNN_402040_Results_' num2str(i),'.mat']);
    DPA_DNN = eval(['DPA_DNN_402040_corrected_y_',num2str(i)]);
    DPA_DNN = reshape(DPA_DNN(1:52,:) + 1i*DPA_DNN(53:104,:), nUSC, nSym, N_Test_Frames);  

    % Loading DPA-LSTM Results
    load([path '\Python_Codes\data\DPA_LSTM_' num2str(nUSC) '15_Results_' num2str(i),'.mat']);
    DPA_LSTM = eval(['DPA_LSTM_' num2str(nUSC) '15_corrected_y_',num2str(i)]);
    DPA_LSTM = reshape(DPA_LSTM(1:52,:) + 1i*DPA_LSTM(53:104,:), nUSC, nSym, N_Test_Frames);  
   
    % Loading DPA-LNN Results
    load([path '\Python_Codes\data\DPA_LNN_Results_' num2str(i),'_Opt.mat']);
    DPA_LNN = eval(['DPA_LNN_corrected_y_',num2str(i)]);
    DPA_LNN = reshape(DPA_LNN(1:52,:) + 1i*DPA_LNN(53:104,:), nUSC, nSym, N_Test_Frames);  
    
    % Loading DPA-LNN Results
    load([path '\Python_Codes\data\DPA_LNN_Results_' num2str(i),'_Opt_WithoutRestriction.mat']);
    DPA_LNN_WR = eval(['DPA_LNN_corrected_y_',num2str(i)]);
    DPA_LNN_WR = reshape(DPA_LNN_WR(1:52,:) + 1i*DPA_LNN_WR(53:104,:), nUSC, nSym, N_Test_Frames);  
    
    tic;
    for u = 1:N_Test_Frames        
        % testing dataset (2000)
        if N_CH(ii) == 2000
            c = u;
        % training dataset (10000)
        elseif N_CH(ii) == 10000 
            c = testing_indices(1,u);
        end 
        Phf(ii)  = Phf(ii)  + norm(True_Channels_Structure(:,:,c))^ 2;
                
        % DPA-DNN
        H_DPA_DNN = DPA_DNN(:,:,u);
        Err_DPA_DNN (ii) =  Err_DPA_DNN (ii) +  norm(H_DPA_DNN - True_Channels_Structure(:,:,c)).^2;
        Equalized_OFDM_Symbols_DPA_DNN = Received_Symbols_FFT_Structure(dpositions,:,c) ./ H_DPA_DNN(dpositions,:);
        
        % DPA-LSTM
        H_DPA_LSTM = DPA_LSTM(:,:,u);
        Err_DPA_LSTM (ii) =  Err_DPA_LSTM (ii) +  norm(H_DPA_LSTM - True_Channels_Structure(:,:,c)).^2;
        Equalized_OFDM_Symbols_DPA_LSTM = Received_Symbols_FFT_Structure(dpositions,:,c) ./ H_DPA_LSTM(dpositions,:);
        
        % % DPA-LNN
        H_DPA_LNN = DPA_LNN(:,:,u);
        Err_DPA_LNN (ii) =  Err_DPA_LNN (ii) +  norm(H_DPA_LNN - True_Channels_Structure(:,:,c)).^2;
        Equalized_OFDM_Symbols_DPA_LNN = Received_Symbols_FFT_Structure(dpositions,:,c) ./ H_DPA_LNN(dpositions,:);
        
        % DPA-LNN - Without Restriction
        H_DPA_LNN_WR = DPA_LNN_WR(:,:,u);
        Err_DPA_LNN_WR (ii) =  Err_DPA_LNN_WR (ii) +  norm(H_DPA_LNN_WR - True_Channels_Structure(:,:,c)).^2;
        Equalized_OFDM_Symbols_DPA_LNN_WR = Received_Symbols_FFT_Structure(dpositions,:,c) ./ H_DPA_LNN_WR(dpositions,:);
        
        % QAM - DeMapping
        De_Mapped_DPA_DNN      = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_DPA_DNN,M);
        De_Mapped_DPA_LSTM     = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_DPA_LSTM,M);
        De_Mapped_DPA_LNN      = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_DPA_LNN,M);
        De_Mapped_DPA_LNN_WR   = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_DPA_LNN_WR,M);

        % Bits Extraction
        Bits_DPA_DNN        = zeros(nDSC,nSym,log2(M));
        Bits_DPA_LSTM       = zeros(nDSC,nSym,log2(M));
        Bits_DPA_LNN        = zeros(nDSC,nSym,log2(M));
        Bits_DPA_LNN_WR     = zeros(nDSC,nSym,log2(M));

        for b = 1 : nSym
           Bits_DPA_DNN(:,b,:)       = de2bi(De_Mapped_DPA_DNN(:,b),nBitPerSym);
           Bits_DPA_LSTM(:,b,:)      = de2bi(De_Mapped_DPA_LSTM(:,b),nBitPerSym);
           Bits_DPA_LNN(:,b,:)       = de2bi(De_Mapped_DPA_LNN(:,b),nBitPerSym);
           Bits_DPA_LNN_WR(:,b,:)    = de2bi(De_Mapped_DPA_LNN_WR(:,b),nBitPerSym);
        end
       
        % BER Calculation
        ber_DPA_DNN    = biterr(Bits_DPA_DNN(:),TX_Bits_Stream_Structure(:,c));
        ber_DPA_LSTM   = biterr(Bits_DPA_LSTM(:),TX_Bits_Stream_Structure(:,c));
        ber_DPA_LNN    = biterr(Bits_DPA_LNN(:),TX_Bits_Stream_Structure(:,c));
        ber_DPA_LNN_WR = biterr(Bits_DPA_LNN_WR(:),TX_Bits_Stream_Structure(:,c));

        Ber_DPA_DNN(ii)        = Ber_DPA_DNN(ii) + ber_DPA_DNN;  
        Ber_DPA_LSTM(ii)       = Ber_DPA_LSTM(ii) + ber_DPA_LSTM;  
        Ber_DPA_LNN(ii)        = Ber_DPA_LNN(ii) + ber_DPA_LNN;  
        Ber_DPA_LNN_WR(ii)     = Ber_DPA_LNN_WR(ii) + ber_DPA_LNN_WR;  
    end
    toc;
end

%% Bit Error Rate (BER)
BER_DPA_DNN                         = Ber_DPA_DNN / (N_Test_Frames * nSym * nDSC * nBitPerSym);
BER_DPA_LSTM                        = Ber_DPA_LSTM / (N_Test_Frames * nSym * nDSC * nBitPerSym);
BER_DPA_LNN_Optimized               = Ber_DPA_LNN / (N_Test_Frames * nSym * nDSC * nBitPerSym);
BER_DPA_LNN_WithoutRestriction      = Ber_DPA_LNN_WR / (N_Test_Frames * nSym * nDSC * nBitPerSym);

%% Normalized Mean Square Error
Phf = Phf ./ N_Test_Frames;
ERR_DPA_DNN                         = Err_DPA_DNN / (N_Test_Frames * Phf);
ERR_DPA_LSTM                        = Err_DPA_LSTM / (N_Test_Frames * Phf);
ERR_DPA_LNN_Optimized               = Err_DPA_LNN / (N_Test_Frames * Phf);
ERR_DPA_LNN_WithoutRestriction      = Err_DPA_LNN_WR / (N_Test_Frames * Phf);

save(['data_' pathdata '\DPA_DNN_Results'],'BER_DPA_DNN','ERR_DPA_DNN');
save(['data_' pathdata '\DPA_LSTM_Results'],'BER_DPA_LSTM','ERR_DPA_LSTM');
save(['data_' pathdata '\DPA_LNN_Results_Optimized'],'BER_DPA_LNN_Optimized','ERR_DPA_LNN_Optimized');
save(['data_' pathdata '\DPA_LNN_Results_WithoutRestriction'],'BER_DPA_LNN_WithoutRestriction','ERR_DPA_LNN_WithoutRestriction');

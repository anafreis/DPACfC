clc
clearvars
close all
warning('off','all')
ch_func = Channel_functions();

mod = '16QAM';
ChType = 'VTV_UC';
v = 100;                 % Moving speed of user in km/h

nSym                    = 20;     % Number of symbols within one frame
EbN0dB                  = 0:5:30; % bit to noise ratio
N_CH                    = [20000;20000;20000;20000;20000;20000;100000]; % Number of channel realizations

pathdata = [num2str(nSym) 'Sym_' mod '_' ChType '_' num2str(v) 'kmh'];
%% Physical Layer Specifications for IEEE 802.11p / OFDM
ofdmBW                 = 10 * 10^6 ;     % OFDM bandwidth (Hz)
nFFT                   = 64;             % FFT size 
nDSC                   = 48;             % Number of data subcarriers
nPSC                   = 4;              % Number of pilot subcarriers
nZSC                   = 12;             % Number of zeros subcarriers
nUSC                   = nDSC + nPSC;    % Number of total used subcarriers
K                      = nUSC + nZSC;    % Number of total subcarriers
deltaF                 = ofdmBW/nFFT;    % Bandwidth for each subcarrier - include all used and unused subcarriers 
Tfft                   = 1/deltaF;       % IFFT or FFT period = 6.4us
Tgi                    = Tfft/4;         % Guard interval duration - duration of cyclic prefix - 1/4th portion of OFDM symbols = 1.6us
Tsignal                = Tgi+Tfft;       % Total duration of BPSK-OFDM symbol = Guard time + FFT period = 8us
K_cp                   = nFFT*Tgi/Tfft;  % Number of symbols allocated to cyclic prefix 
pilots_locations       = [8,22,44,58].'; % Pilot subcarriers positions
pilots                 = [1 1 1 -1].';
data_locations         = [2:7, 9:21, 23:27, 39:43, 45:57, 59:64].'; % Data subcarriers positions
ppositions             = [7,21, 32,46].';                           % Pilots positions in Kset
dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].';        % Data positions in Kset
% Pre-defined preamble in frequency domain
dp = [ 0 0 0 0 0 0 +1 +1 -1 -1 +1  +1 -1  +1 -1 +1 +1 +1 +1 +1 +1 -1 -1 +1 +1 -1 +1 -1 +1 +1 +1 +1 0 +1 -1 -1 +1 +1 -1 +1 -1 +1 -1 -1 -1 -1 -1 +1 +1 -1 -1 +1 -1 +1 -1 +1 +1 +1 +1 0 0 0 0 0];
Ep                     = 1;              % preamble power per sample
dp                     = fftshift(dp);   % Shift zero-frequency component to center of spectrum    
predefined_preamble    = dp;
Kset                   = find(dp~=0);    % set of allocated subcarriers                  
Kon                    = length(Kset);   % Number of active subcarriers
dp                     = sqrt(Ep)*dp.';
%%%%%%%%%
xp                     = sqrt(K)*ifft(dp);
%%%%%%%%%
xp_cp                  = [xp(end-K_cp+1:end); xp];  % Adding CP to the time domain preamble
preamble_80211p        = repmat(xp_cp,1,2);         % IEEE 802.11p preamble symbols (tow symbols)
%% ------ Bits Modulation Technique------------------------------------------
Mod_Type                  = 1;              % 0 for BPSK and 1 for QAM 
if(Mod_Type == 0)
    nBitPerSym            = 1;
    Pow                   = 1;
    %BPSK Modulation Objects
    bpskModulator         = comm.BPSKModulator;
    bpskDemodulator       = comm.BPSKDemodulator;
    M                     = 1;
elseif(Mod_Type == 1)
    if(strcmp(mod,'QPSK') == 1)
         nBitPerSym            = 2; 
    elseif (strcmp(mod,'16QAM') == 1)
         nBitPerSym            = 4; 
    elseif (strcmp(mod,'64QAM') == 1)
         nBitPerSym            = 6; 
    end
    M                     = 2 ^ nBitPerSym; % QAM Modulation Order   
    Pow                   = mean(abs(qammod(0:(M-1),M)).^2); % Normalization factor for QAM        
end
%% -----------------Vehicular Channel Model Parameters--------------------------
fs                        = K*deltaF;               % Sampling frequency in Hz, here case of 802.11p with 64 subcarriers and 156250 Hz subcarrier spacing
fc                        = 5.9e9;                  % Carrier Frequecy in Hz.
c                         = 3e8;                    % Speed of Light in m/s
fD                        = (v/3.6)/c*fc;           % Doppler freq in Hz
plotFlag                  = 0;                      % 1 to display the channel frequency response
[rchan,~,avgPathGains]    = ch_func.GenFadingChannel(ChType, fD, fs);
init_seed = 22;
%% ---------Bit to Noise Ratio------------------%
SNR_p                     = EbN0dB;
SNR_p                     = SNR_p.';
EbN0Lin                   = 10.^(SNR_p/10);
N0 = Ep*10.^(-SNR_p/10);
%% Simulation Parameters 
N_SNR                   = length(SNR_p); % SNR length

% Normalized mean square error (NMSE) vectors
Err_DPA             = zeros(N_SNR,1);

% Bit error rate (BER) vectors
Ber_Ideal               = zeros(N_SNR,1);
Ber_LS                  = zeros(N_SNR,1);
Ber_DPA                 = zeros(N_SNR,1);

% average channel power E(|hf|^2)
Phf_H_Total             = zeros(N_SNR,1);
%% Simulation Loop
for n_snr = 1:N_SNR
    disp(['Running Simulation, SNR = ', num2str(EbN0dB(n_snr))]);
    tic;    
    TX_Bits_Stream_Structure                = zeros(nDSC * nSym  * nBitPerSym, N_CH(n_snr));
    Received_Symbols_FFT_Structure          = zeros(Kon,nSym, N_CH(n_snr));
    True_Channels_Structure                 = zeros(Kon, nSym, N_CH(n_snr));
    LS_Structure                            = zeros(Kon, nSym, N_CH(n_snr));
    DPA_Structure                           = zeros(Kon, nSym, N_CH(n_snr));

    for n_ch = 1:N_CH(n_snr) % loop over channel realizations
        % Bits Stream Generation 
        Bits_Stream = randi(2, nDSC * nSym  * nBitPerSym,1)-1;
        % Bits Mapping: M-QAM Modulation
        TxBits = reshape(Bits_Stream,nDSC , nSym  , nBitPerSym);
        % Gray coding goes here
        TxData = zeros(nDSC ,nSym);
        for m = 1 : nBitPerSym
           TxData = TxData + TxBits(:,:,m)*2^(m-1);
        end
        % M-QAM Modulation
        Modulated_Bits  =1/sqrt(Pow) * qammod(TxData,M);
        % OFDM Frame Generation
        OFDM_Frame = zeros(K,nSym);
        OFDM_Frame(data_locations,:) = Modulated_Bits;
        OFDM_Frame(pilots_locations,:) = repmat(pilots,1,nSym);
        % Taking FFT and normalizing (power of transmit symbol needs to be 1)
        IFFT_Data = ifft(OFDM_Frame);
        norm_factor = sqrt(sum(abs(IFFT_Data(:).^2))./length(IFFT_Data(:)));
        IFFT_Data = IFFT_Data/norm_factor;
        power_frame = sqrt(sum(abs(IFFT_Data(:).^2))./length(IFFT_Data(:)));
        % Appending cylic prefix
        CP = IFFT_Data((K - K_cp +1):K,:);
        IFFT_Data_CP = [CP; IFFT_Data];
        % Appending preamble symbol 
        IFFT_Data_CP_Preamble = [ preamble_80211p IFFT_Data_CP];
       
        % ideal estimation
        release(rchan);
        rchan.Seed = rchan.Seed+1;
        [ h, y ] = ch_func.ApplyChannel(rchan, IFFT_Data_CP_Preamble, K_cp);

        yp = y((K_cp+1):end,1:2);
        y  = y((K_cp+1):end,3:end);
        
        %%%%%
        yFD = sqrt(1/K)*fft(y);	
        yfp = sqrt(1/K)*fft(yp); % FD preamble
        %%%%%
        
        h = h((K_cp+1):end,:);
        hf = fft(h); % Fd channel
        hf  = hf(:,3:end);    
        
        Phf_H_Total(n_snr) = Phf_H_Total(n_snr) + norm(hf(Kset))^2;
        % add noise
	    noise_preamble = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,2], 1);	
        yfp_r = yfp +  noise_preamble;	
        noise_OFDM_Symbols = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,size(yFD,2)], 1);	
        y_r   = yFD + noise_OFDM_Symbols;   
        
        %% Channel Estimation
        % IEEE 802.11p LS Estimate at Preambles
        he_LS_Preamble = ((yfp_r(Kset,1) + yfp_r(Kset,2))./(2.*predefined_preamble(Kset).'));
        H_LS = repmat(he_LS_Preamble,1,nSym);
        err_LS_Preamble = norm(H_LS - hf(Kset,:))^2;
           
        % DPA Channel Estimation
        [H_DPA, Equalized_OFDM_Symbols_DPA] = DPA_Estimation(he_LS_Preamble ,y_r, Kset, ppositions, mod, nUSC, nSym);
        err_H_DPA = norm(H_DPA - hf(Kset,:))^2;
        Err_DPA(n_snr) = Err_DPA(n_snr) + err_H_DPA;
                
        % Equalization
        y_Ideal = y_r(data_locations ,:) ./ hf(data_locations,:); %Ideal
        y_LS = y_r(data_locations ,:)./ H_LS(dpositions,:); % LS
    
        % QAM - DeMapping
        De_Mapped_Ideal     = qamdemod(sqrt(Pow) * y_Ideal,M);
        De_Mapped_LS        = qamdemod(sqrt(Pow) * y_LS,M);
        De_Mapped_DPA       = qamdemod(sqrt(Pow) * Equalized_OFDM_Symbols_DPA(dpositions,:),M); 
    
        % Bits Extraction
        Bits_Ideal       = zeros(nDSC,nSym,log2(M));
        Bits_LS          = zeros(nDSC,nSym,log2(M));
        Bits_DPA         = zeros(nDSC,nSym,log2(M));
    
        for b = 1:nSym
            Bits_Ideal(:,b,:)     = de2bi(De_Mapped_Ideal(:,b),nBitPerSym);
            Bits_LS(:,b,:)        = de2bi(De_Mapped_LS(:,b),nBitPerSym); 
            Bits_DPA(:,b,:)       = de2bi(De_Mapped_DPA(:,b),nBitPerSym);  
        end
           
        % BER Calculation
        ber_Ideal   = biterr(Bits_Ideal(:),Bits_Stream);
        ber_LS      = biterr(Bits_LS(:),Bits_Stream);
        ber_DPA     = biterr(Bits_DPA(:),Bits_Stream);        
    
        Ber_Ideal (n_snr)    = Ber_Ideal (n_snr) + ber_Ideal;
        Ber_LS (n_snr)       = Ber_LS (n_snr) + ber_LS;
        Ber_DPA(n_snr)       = Ber_DPA(n_snr) + ber_DPA;
    
        TX_Bits_Stream_Structure(:, n_ch) = Bits_Stream;
        Received_Symbols_FFT_Structure(:,:,n_ch) = y_r(Kset,:);
        True_Channels_Structure(:,:,n_ch) = hf(Kset,:);
        LS_Structure(:,n_ch)     = he_LS_Preamble;
        DPA_Structure(:,:,n_ch)  = H_DPA;          
    end   
        save(['data_' pathdata '\Simulation_' num2str(n_snr)],...
               'TX_Bits_Stream_Structure',...
               'Received_Symbols_FFT_Structure',...
               'True_Channels_Structure',...
               'LS_Structure','DPA_Structure','-v7.3');
        toc;
end
%% Bit Error Rate (BER)
BER_Ideal             = Ber_Ideal ./(N_CH .* nSym * nDSC * nBitPerSym);
BER_LS                = Ber_LS ./ (N_CH .* nSym * nDSC * nBitPerSym);
BER_DPA               = Ber_DPA ./ (N_CH .* nSym * nDSC * nBitPerSym);

%% Normalized Mean Square Error
Phf_H       = Phf_H_Total./(N_CH);
ERR_DPA     = Err_DPA ./ (Phf_H .* N_CH * nSym);

save(['data_' pathdata '\Simulation_variables'],'mod','Kset','fD','ChType','avgPathGains');
save(['data_' pathdata '\Classical_Results'],'ERR_DPA','BER_Ideal','BER_LS','BER_DPA');
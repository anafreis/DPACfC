function [H_DPA, Equalized_OFDM_Symbols] = DPA_Estimation(he_LS_Preamble ,y_r, Kset, ppositions, mod, nUSC, nSym)
H_DPA = zeros(nUSC, nSym);
Equalized_OFDM_Symbols = zeros(nUSC, nSym);
for i = 1:nSym
    if(i == 1)
        % Step 1: Equalization.
        Equalized_OFDM_Symbol = y_r(Kset,i)./ he_LS_Preamble;
        Equalized_OFDM_Symbols(:,i) = Equalized_OFDM_Symbol;
       
        % Step 2: Constructing Data Pilot.
        De_Equalized_OFDM_Symbol = wlanClosestReferenceSymbol(Equalized_OFDM_Symbol,mod);
        De_Equalized_OFDM_Symbol(ppositions,:) = [1;1;1;-1]; 
        % Step 3: LS Estimation.
        H_DPA(:,i) = y_r(Kset,i)./ De_Equalized_OFDM_Symbol;
       
    elseif (i > 1)
        % Step 1: Equalization 
        Equalized_OFDM_Symbol = y_r(Kset,i)./ H_DPA(:,i-1);
        Equalized_OFDM_Symbols(:,i) = Equalized_OFDM_Symbol;
        % Step 2: Constructing Data Pilot.
        De_Equalized_OFDM_Symbol = wlanClosestReferenceSymbol(Equalized_OFDM_Symbol,mod);
        De_Equalized_OFDM_Symbol(ppositions,:) = [1;1;1;-1]; 
        % Step 3: LS Estimation.
        H_DPA(:,i)  = y_r(Kset,i)./ De_Equalized_OFDM_Symbol;       
    end
end
end


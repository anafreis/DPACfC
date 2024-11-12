clc;
clear all;

% Define Eb/N0 range
EbN0dB = 0:5:30;
% Define the parameters for the different scenarios
scenarios = {
    struct('mod', 'QPSK', 'ChType', 'VTV_UC', 'v', 100, 'linestyle', '--');
    struct('mod', '16QAM', 'ChType', 'VTV_UC', 'v', 100, 'linestyle', ':');
    struct('mod', '64QAM', 'ChType', 'VTV_UC', 'v', 100, 'linestyle', '-.'),
};
%% BER 
figure
hold on
for i = 1:length(scenarios)
    scenario = scenarios{i};
    nSym = 20;  % Number of symbols within one frame

    pathdata = [num2str(nSym) 'Sym_' scenario.mod '_' scenario.ChType '_' num2str(scenario.v) 'kmh'];

    load(['data_' pathdata '\Classical_Results']);
    load(['data_' pathdata '\DPA_DNN_Results'])
    load(['data_' pathdata '\DPA_LSTM_Results']);
    load(['data_' pathdata '\DPA_LNN_Results_Optimized']);
    load(['data_' pathdata '\DPA_LNN_Results_WithoutRestriction']);

    semilogy(EbN0dB, BER_Ideal, 'k*', 'LineWidth', 2, 'LineStyle', scenario.linestyle);
    colorOrder = get(gca, 'ColorOrder');
    semilogy(EbN0dB, BER_DPA_DNN, 'o', 'MarkerFaceColor', colorOrder(2,:), 'Color', colorOrder(2,:), 'MarkerSize', 8, 'LineWidth', 2, 'LineStyle', scenario.linestyle);
    semilogy(EbN0dB, BER_DPA_LSTM, '^', 'MarkerFaceColor', colorOrder(3,:), 'Color', colorOrder(3,:), 'MarkerSize', 8, 'LineWidth', 2, 'LineStyle', scenario.linestyle);
    semilogy(EbN0dB, BER_DPA_LNN_WithoutRestriction, '>', 'MarkerFaceColor', colorOrder(5,:), 'Color', colorOrder(5,:), 'MarkerSize', 8, 'LineWidth', 2, 'LineStyle', scenario.linestyle);
    semilogy(EbN0dB, BER_DPA_LNN_Optimized, 'h', 'MarkerFaceColor', colorOrder(4,:), 'Color', colorOrder(4,:), 'MarkerSize', 8, 'LineWidth', 2, 'LineStyle', scenario.linestyle);

end

% Aux structures for the legends 
estimator = [
    plot(nan, nan, 'k-*', 'LineWidth', 2), ...
    plot(nan, nan, '-o', 'MarkerFaceColor', colorOrder(2,:), 'Color', colorOrder(2,:), 'MarkerSize', 6, 'LineWidth', 1.5), ...
    plot(nan, nan, '-^', 'MarkerFaceColor', colorOrder(3,:), 'Color', colorOrder(3,:), 'MarkerSize', 6, 'LineWidth', 1.5), ...
    plot(nan, nan, '->', 'MarkerFaceColor', colorOrder(5,:), 'Color', colorOrder(5,:), 'MarkerSize', 6, 'LineWidth', 1.5),...
    plot(nan, nan, '-h', 'MarkerFaceColor', colorOrder(4,:), 'Color', colorOrder(4,:), 'MarkerSize', 6, 'LineWidth', 1.5)
    ];
modulation = [
    plot(nan, nan, 'k--', 'LineWidth', 1.5, 'DisplayName', 'QPSK'), ...
    plot(nan, nan, 'k:', 'LineWidth', 1.5, 'DisplayName', '16QAM'), ...
    plot(nan, nan, 'k-.', 'LineWidth', 1.5, 'DisplayName', '64QAM')

];

legend([estimator, modulation], ...
       {'Perfect Channel', 'DPA-DNN', 'DPA-LSTM-NN','DPA-CfC (No restriction)','DPA-CfC', 'QPSK', '16-QAM','64-QAM'},'Interpreter', 'latex', 'Location', 'southwest');

xlabel('SNR [dB]','Interpreter','latex');
ylabel('BER','Interpreter','latex');
set(gca, 'YScale', 'log');
grid on;
axis([min(EbN0dB) max(EbN0dB) 0.3*10^-3 10^0]);
xticks(0:5:30);
yticks([10^-3 10^-2 10^-1 10^0]);
set(gca,'FontSize',14,'TickLabelInterpreter','latex')

%% MSE 
for i = 1:length(scenarios)
    % subplot(1,length(scenarios),i)
    figure
    % figure
    scenario = scenarios{i};
   
    % Linear
    pathdata = [num2str(nSym) 'Sym_' scenario.mod '_' scenario.ChType '_' num2str(scenario.v) 'kmh'];    

    load(['data_' pathdata '\DPA_DNN_results'])
    load(['data_' pathdata '\DPA_LSTM_Results']);
    load(['data_' pathdata '\DPA_LNN_Results_Optimized']);
    load(['data_' pathdata '\DPA_LNN_Results_WithoutRestriction']);

    % Remove zero from data
    ERR_DPA_DNN(ERR_DPA_DNN == 0) = [];
    ERR_DPA_LSTM(ERR_DPA_LSTM == 0) = [];
    ERR_DPA_LNN_Optimized(ERR_DPA_LNN_Optimized == 0) = [];
    ERR_DPA_LNN_WithoutRestriction(ERR_DPA_LNN_WithoutRestriction == 0) = [];

    semilogy(EbN0dB, ERR_DPA_DNN, '-o', 'MarkerFaceColor', colorOrder(2,:), 'Color', colorOrder(2,:), 'MarkerSize', 6, 'LineWidth', 1.5);
    hold on
    semilogy(EbN0dB, ERR_DPA_LSTM, '-^', 'MarkerFaceColor', colorOrder(3,:), 'Color', colorOrder(3,:), 'MarkerSize', 6, 'LineWidth', 1.5);
    semilogy(EbN0dB, ERR_DPA_LNN_WithoutRestriction, '->', 'MarkerFaceColor', colorOrder(5,:), 'Color', colorOrder(5,:), 'MarkerSize', 6, 'LineWidth', 1.5);
    semilogy(EbN0dB, ERR_DPA_LNN_Optimized, '-h', 'MarkerFaceColor', colorOrder(4,:), 'Color', colorOrder(4,:), 'MarkerSize', 6, 'LineWidth', 1.5);
    grid on
    hold off

    if i == ceil((length(scenarios))/2)
        legend({'DPA-DNN','DPA-LSTM-NN', 'DPA-CfC (No restriction)', 'DPA-CfC'}, 'Interpreter', 'latex', 'Location', 'southwest');
    end
    scenario_titles = {'QPSK', '16-QAM', '64-QAM'};
    title(scenario_titles{i},'Interpreter','latex')
    axis([min(EbN0dB) max(EbN0dB) 10^-5 10^0]);
    xticks(0:5:30);
    yticks([10^-5 10^-4 10^-3 10^-2 10^-1 10^0]);
    xlabel('SNR [dB]','Interpreter','latex');
    ylabel('NMSE','Interpreter','latex');
    set(gcf,'position',[0 0 300 400])
    set(gca,'FontSize',16)
    set(gca,'FontSize',14,'TickLabelInterpreter','latex')

end

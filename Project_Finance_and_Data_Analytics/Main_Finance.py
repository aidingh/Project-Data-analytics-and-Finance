from pandas_datareader import data as wb
from Finance_Analytics_Tools import Finance_Analytics_Tools
from Finance_Investment_Risks import Finance_Investment_Risks
from Finance_Neural_Nets import Finance_Neural_Nets
from Finance_Trading_Strategy_And_Plot import Finance_Trading_Strategy_And_Plot
from Finance_Machine_Learning_Models import Finance_Machine_Learning_Models
import numpy as np

def main():


    D1 = Finance_Investment_Risks('yahoo', '2010-1-1', '2017-03-24')
    assets = ['PG', 'BEI.DE']
    w_list = [0.5, 0.5]
    D1.asset_risks(assets)
    a,b = D1.calculate_covariance_and_correlation_between_two_assets(assets, 1)
    c,d = D1.calculating_risk_of_portfolio(assets, w_list)
    D1.Markowitz_portfolio_optimization_efficient_frontier(assets)


    D1 = Finance_Analytics_Tools('yahoo', '2012-1-1', '2013-1-1')
    data_frame = D1.get_asset_data('AAPL')
    log_rets = D1.log_rets(data_frame)
    my_tuple = (0.00, 0.01)
    prob_0 = D1.normal_dist_model(log_rets, 0.005,'return between in interval',1, my_tuple)
    prob_1 = D1.normal_dist_model(log_rets, 0.005,'return greater or equal',0)


    D1 = Finance_Analytics_Tools('yahoo', '1995-1-1', '2020-1-25')
    data_frame = D1.get_asset_data('PG')

    average_daily_returns = D1.average_daily_returns(data_frame, 1)
    average_annual_returns = D1.average_annual_returns(data_frame, 1)
    log_daily_returns = D1.log_daily_returns(data_frame, 0)

    portfolio = ['PG', 'MSFT', 'F', 'GE']
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    D2 = Finance_Analytics_Tools('yahoo', '1995-1-1', '2020-1-25')
    portfolio_return_historic = D2.historic_return_of_investment_portfolio(portfolio, weights, 1)
    print(portfolio_return_historic)


    indices = ['^GSPC', '^IXIC', '^GDAXI']
    stocks = ['PG']
    D1 = Finance_Analytics_Tools('yahoo', '1995-1-1', '2020-1-25')
    ind_rets = D1.historic_return_of_indices(indices, 2, stocks)
    print(ind_rets)


    FNN = Finance_Neural_Nets('AAPL', 'yahoo', '2012-01-01', '2020-2-3')
    FNN.window = 60
    FNN.batch_size = 64
    FNN.epoch = 20
    FNN.data_column = 'Adj Close'
    x_train, y_train, x_train.shape, training_data_len, scaler, scaled_data, data, dataset = FNN.prepare_data()
    #model = FNN.neural_net_RNN_model(x_train)
    model = FNN.neural_net_LSTM_model(x_train)
    FNN.train_neural_net(model, x_train, y_train, training_data_len, scaler, scaled_data, data, dataset)


    FMLM = Finance_Machine_Learning_Models('TSLA', 'yahoo', '2000-1-1', '2020-2-4')
    FMLM.forecast_out = 1
    FMLM.data_column = 'Adj Close'
    x_train, x_test, y_train, y_test, df = FMLM.prepare_data_set()
    svm_var, svm_conf = FMLM.forecast_with_SVM(x_train, x_test, y_train, y_test, df)
    linreg_var, linreg_conf = FMLM.forecast_with_linear_reg(x_train, x_test, y_train, y_test, df)
    y_hat = FMLM.linear_reg_for_market(1200)


    #FTSAP = Finance_Trading_Strategy_And_Plot('TSLA', 'yahoo', '2000-1-1', '2020-1-1','Adj Close', 20, 50)
    FTSAP = Finance_Trading_Strategy_And_Plot('TSLA', 'yahoo', '2000-1-1', '2020-1-1', 'Close', 30, 90)
    FTSAP.moving_average_trading_strategy()
    FTSAP.resample = '10D'
    FTSAP.moving_average_trading_strategy_candlestick_plot()
    FTSAP.moving_average_trading_strategy_with_buy_and_sell_signals()


    FMLM = Finance_Machine_Learning_Models()
    econ_data = FMLM.prepare_data_set_multiple_reg()
    regression_model, X_test, y_test, X, Y = FMLM.multiple_regression_model(econ_data)
    y_hat, predicted_vals = FMLM.predict_with_multiple_linear_model(regression_model, X_test, 5)
    FMLM.evaluate_multiple_regression_model(X, Y, y_test, y_hat)


def debugg_function():
    print("Hello Debug")

if __name__ == '__main__':
        main()
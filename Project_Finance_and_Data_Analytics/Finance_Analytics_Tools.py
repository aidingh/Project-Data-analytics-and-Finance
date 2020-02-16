from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm

class Finance_Analytics_Tools:

    class_var = 0
    def __init__(self,  source, start_date, end_date):
        #self.asset_name = asset_name
        self.source = source
        self.start_date = start_date
        self.end_date = end_date

    def get_asset_data(self, asset_name):
        asset_data = wb.DataReader(asset_name, data_source=self.source, start=self.start_date, end=self.end_date)
        return asset_data

    def get_multiple_asset_data(self, asset_list):
        asset_data = pd.DataFrame()

        for data in asset_list:
            asset_data[data] = wb.DataReader(data, data_source=self.source, start=self.start_date, end=self.end_date)['Adj Close']

        return asset_data

    def log_rets(self, data_frame):
        data_frame['log_return'] = np.log(data_frame['Adj Close'] / data_frame['Adj Close'].shift(1))
        log_ret = data_frame['log_return']
        return log_ret

    def simple_rets(self, data_frame):
        data_frame['simple_return'] = (data_frame['Adj Close'] / data_frame['Adj Close'].shift(1)) - 1
        simple_ret = data_frame['simple_return']
        return simple_ret

    def historic_return_of_investment_portfolio(self, port_list, weights, plot_var):
        mydata = pd.DataFrame()
        for p in port_list:
            mydata[p] = wb.DataReader(p, data_source=self.source, start=self.start_date)['Adj Close']

        plot_data = mydata
        returns = (mydata / mydata.shift(1)) - 1
        annual_returns = returns.mean() * 250
        rate_of_return_of_port = np.dot(annual_returns, weights)

        if plot_var == 1:
            (plot_data / plot_data.iloc[0] * 100).plot(figsize=(15, 6))
            plt.show()
            return rate_of_return_of_port
        else:
            return rate_of_return_of_port

    def historic_return_of_indices(self, indices_list, plot_var, stock_list=None):
        # ---GSPC = S&P500, IXIC = NASDAQ, GDAXI = German DAX---
        ind_data = pd.DataFrame()
        data_2 = pd.DataFrame()
        for data in indices_list:
            ind_data[data] = wb.DataReader(data, data_source=self.source, start=self.start_date, end=self.end_date)['Adj Close']

        ind_returns = (ind_data / ind_data.shift(1)) -1
        annual_ind_returns = ind_returns.mean() * 250

        if plot_var == 1:
            (ind_data / ind_data.iloc[0] * 100).plot(figsize=(15, 6))
            plt.show()
            return annual_ind_returns

        if stock_list and plot_var == 2:
            stock_and_indices = indices_list + stock_list

            for d in stock_and_indices:
                data_2[d] = wb.DataReader(d, data_source=self.source, start=self.start_date, end=self.end_date)['Adj Close']

            (data_2 / data_2.iloc[0] * 100).plot(figsize=(15, 6))
            plt.show()
            return annual_ind_returns

        else:
            return annual_ind_returns

    def average_annual_returns(self, data_frame, plot_var):
        simple_ret = self.simple_rets(data_frame)
        average_annual_returns = simple_ret.mean() * 250

        if plot_var == 1:
            self.return_plots(simple_ret)
            return average_annual_returns
        else:
            return average_annual_returns

    def plot_logic(self, plot_logic, simple_ret, average_annual_returns):
        pass

    def average_daily_returns(self, data_frame, plot_var):
        simple_ret = self.simple_rets(data_frame)
        average_daily_returns = simple_ret.mean()

        if plot_var == 1:
            self.return_plots(simple_ret)
            return average_daily_returns
        else:
            return average_daily_returns

    def log_daily_returns(self, data_frame, plot_var):
        log_ret = self.log_rets(data_frame)
        log_ret_daily_returns = log_ret.mean()

        if plot_var == 1:
            self.return_plots(log_ret)
            return log_ret_daily_returns
        else:
            return log_ret_daily_returns

    def log_annual_returns(self, data_frame, plot_var):

        log_ret = self.log_rets(data_frame)
        log_ret_annual_ret = log_ret.mean() * 250

        if plot_var is 1:
            self.return_plots(log_ret)
            return log_ret_annual_ret
        else:
            return log_ret_annual_ret

    def return_plots(self, plot_data):
        plot_data.plot(figsize=(16, 8))
        plt.show()

    def dist_plots(self, returns):
        returns.plot.kde()
        plt.show()

    def norm_dist_get_prob(self, returns, prob):
        u = returns.mean()
        stdev = returns.std()
        value = scipy.stats.norm.ppf(prob, u, stdev)
        return value

    def normal_dist_model(self, returns, x, type_string, plot_data, interval_tuple=None):
        u = returns.mean()
        stdev = returns.std()

        if plot_data is 1:
            self.dist_plots(returns)
        else:
            if type_string == 'return greater or equal':
                prob_greater_eq = scipy.stats.norm.sf(x, u, stdev)
                return prob_greater_eq

            elif type_string == 'return loss greater than':
                x = (-1)*(x)
                prob_greater_eq_loss = scipy.stats.norm.sf(x, u, stdev)
                prob_greater_eq_loss = 1 - prob_greater_eq_loss
                return prob_greater_eq_loss

            elif type_string == 'return less or equal':
                prob_less_eq = scipy.stats.norm.cdf(x, u, stdev)
                return prob_less_eq

            elif type_string == 'return between in interval' and interval_tuple is not None:
                x1 = interval_tuple[0]
                x2 = interval_tuple[1]

                for_x1 = scipy.stats.norm.sf(x1, u, stdev)
                for_x2 = scipy.stats.norm.sf(x2, u, stdev)
                prob_in_interval = for_x1 - for_x2
                return prob_in_interval

            elif type_string == 'return or loss greater than' and interval_tuple is not None:
                x1 = interval_tuple[0]
                x2 = interval_tuple[1]

                x2 = (-1) * (x2)
                for_x1 = scipy.stats.norm.sf(x1, u, stdev)
                for_x2 = scipy.stats.norm.sf(x2, u, stdev)
                for_x2 = 1 - for_x2
                prob = for_x2 + for_x1
                return prob

    def Black_Scholes_model(self, data_frame, r, K, T):
        S = data_frame.iloc[-1]

        log_returns = np.log(1 + data_frame.pct_change())
        stdev = log_returns.std() * 250 ** 0.5

        #r = 0.025
        #K = 110.0
        #T = 1

        d1 = (np.log(S / K) + (r + stdev ** 2 / 2) * T) / (stdev * np.sqrt(T))
        d2 = (np.log(S / K) + (r - stdev ** 2 / 2) * T) / (stdev * np.sqrt(T))
        BSM = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))

        return BSM












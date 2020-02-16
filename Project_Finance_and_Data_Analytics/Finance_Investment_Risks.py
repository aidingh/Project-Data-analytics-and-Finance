from Finance_Analytics_Tools import Finance_Analytics_Tools
import numpy as np
from pandas_datareader import data as wb
import pandas as pd
import matplotlib.pyplot as plt

class Finance_Investment_Risks(Finance_Analytics_Tools):

    class_var = 0
    def __init__(self, source, start_date, end_date):
        #self.asset_name = asset_name
        super().__init__(source, start_date, start_date)
        self.source = source
        self.start_date = start_date
        self.end_date = end_date

    def asset_risks(self, asset_list):
        list_of_asset_annual_stds = []
        list_of_asset_daily_stds = []
        assets = Finance_Analytics_Tools.get_multiple_asset_data(self, asset_list)
        asset_log_returns = np.log(assets / assets.shift(1))

        for i, j in zip(asset_log_returns, asset_log_returns):
            print(i+': ' + 'Annual Standard deviation '+ str(asset_log_returns[i].std()*250**0.5), 'Control list vals')
            print(j + ': ' + 'Daily Standard deviation ' + str(asset_log_returns[j].std()), 'Control list vals')
            list_of_asset_annual_stds.append(i+': ' + 'Annual Standard deviation '+ str(asset_log_returns[i].std()*250**0.5))
            list_of_asset_daily_stds.append(j+': ' + 'Daily Standard deviation '+ str(asset_log_returns[j].std()))

        return list_of_asset_annual_stds, list_of_asset_daily_stds

    def calculate_covariance_and_correlation_between_two_assets(self, asset_list, plot_var):

        assets_data = Finance_Analytics_Tools.get_multiple_asset_data(self, asset_list)
        asset_log_returns = np.log(assets_data / assets_data.shift(1))
        asset_1 = asset_list[0]
        asset_2 = asset_list[1]

        cov_val = asset_log_returns[asset_1].cov(asset_log_returns[asset_2])
        corr_val = asset_log_returns[asset_1].corr(asset_log_returns[asset_2])

        if plot_var == 1:
            asset_log_returns.plot.scatter(x=asset_1, y=asset_2, c='DarkBlue')
            plt.show()
            return corr_val, cov_val
        else:
            return corr_val, cov_val

    def calculating_risk_of_portfolio(self, asset_list, weights_list):
        assets = Finance_Analytics_Tools.get_multiple_asset_data(self, asset_list)
        asset_log_returns = np.log(assets / assets.shift(1))
        weights = np.array(weights_list)

        pfolio_var = np.dot(weights.T, np.dot(asset_log_returns.cov() * 250, weights))
        pfolio_vol = (np.dot(weights.T, np.dot(asset_log_returns.cov() * 250, weights))) ** 0.5

        # ---plot the stocks a a probability density function ---
        asset_log_returns.plot.kde()
        plt.show()
        return pfolio_vol, pfolio_var

    def calculate_systematic_and_non_systematic_risk_of_portfolio(self, asset_list, weights_list):
        assets = Finance_Analytics_Tools.get_multiple_asset_data(self, asset_list)
        asset_log_returns = np.log(assets / assets.shift(1))
        weights = np.array(weights_list)

        asset_1 = asset_list[0]
        asset_2 = asset_list[1]

        # ---Annual Diversifiable Risk---
        # Systematic risk
        # Diversifiable risk = portfolio varaince - weighted annaual variances

        pfolio_var = np.dot(weights.T, np.dot(asset_log_returns.cov() * 250, weights))
        asset_1_annual_variance = asset_log_returns[asset_1].var() * 250
        asset_2_annual_variance = asset_log_returns[asset_2].var() * 250

        systematic_port_risk = pfolio_var - (weights[0] ** 2 * asset_1_annual_variance) - (weights[1] ** 2 * asset_2_annual_variance)

        # ---Annual Un-Diversifiable Risk---
        # Un-Systematic risk
        # Un-Diversifiable risk = portfolio varaince - diversifiable risk
        un_systematic_port_risk = pfolio_var - systematic_port_risk

        return systematic_port_risk, un_systematic_port_risk

    def Markowitz_portfolio_optimization_efficient_frontier(self, asset_list, iterations):
        pfolio_returns = []
        pfolio_volatilities = []
        pf_data = pd.DataFrame()

        for data in asset_list:
            pf_data[data] = wb.DataReader(data, data_source=self.source, start=self.start_date, end=self.end_date)['Adj Close']

        (pf_data / pf_data.iloc[0] * 100).plot(figsize=(15, 5))
        plt.show()

        log_returns = np.log(pf_data / pf_data.shift(1))
        log_returns_mean = log_returns.mean() * 250
        log_returns_cov = log_returns.cov() * 250
        log_returns_matrix = log_returns.corr()

        num_assets = len(asset_list)

        # ---------Simulation of 1000 observations of mean and varience---------
        # In this section we are considering 1000 different combinations of the same asset. We simulate the 1000 different combinations of their weight values
        for x in range(iterations):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            pfolio_returns.append(np.sum(weights * log_returns_mean) * 250)
            pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns_cov, weights))))

        pfolio_returns = np.array(pfolio_returns)
        pfolio_volatilities = np.array(pfolio_volatilities)

        portfolios = pd.DataFrame({'Return': pfolio_returns, 'Volatility': pfolio_volatilities})

        portfolios.plot(x='Volatility', y='Return', kind='scatter', figsize=(15, 6))
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.show()

    def capital_asset_pricing_model(self, asset_list):
        assets = Finance_Analytics_Tools.get_multiple_asset_data(self, asset_list)
        sec_returns = np.log(assets / assets.shift(1))

        asset_string = asset_list[0]
        market_string = asset_list[1]

        cov = sec_returns.cov() * 250
        cov_with_market = cov.iloc[0, 1]
        market_var = sec_returns[market_string].var() * 250

        asset_beta = cov_with_market / market_var

        asset_expected_return = 0.025 + asset_beta * 0.05
        Sharpe = (asset_expected_return - 0.025) / (sec_returns[asset_string] * 250 ** 0.5)

        return asset_beta, asset_expected_return, Sharpe

    def annual_portfolio_info(self, asset_list, weights_list):
        assets = Finance_Analytics_Tools.get_multiple_asset_data(self, asset_list)
        asset_log_returns = np.log(assets / assets.shift(1))
        weights = np.array(weights_list)

        #---------Expected portfolio return---------
        port_return = np.sum(weights * asset_log_returns.mean()) * 250
        print(port_return, 'Expected portfolio return')

        # ---------Expected portfolio varience---------
        port_var = np.dot(weights.T, np.dot(asset_log_returns.cov()*250, weights))
        print(port_var, 'Expected portfolio varience')

        #---------Expected portfolio volatility---------
        port_vola = np.sqrt(port_var)
        print(port_vola, 'Expected portfolio volatility')

        return port_return, port_var, port_vola
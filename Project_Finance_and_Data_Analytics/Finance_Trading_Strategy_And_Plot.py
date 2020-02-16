from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

class Finance_Trading_Strategy_And_Plot:

    resample = ''


    def __init__(self, asset_name, source, start_date, end_date, data_column, short_window, long_window):
        self.asset_name = asset_name
        self.data_column = data_column
        self.source = source
        self.start_date = start_date
        self.end_date = end_date
        self.short_window = short_window
        self.long_window = long_window


    def moving_average_trading_strategy(self):
        # Conclution: When MA is above the above price we have a bearish market and a downtrend
        # When MA is below price we have a bullish market and a uptrend
        # When MA goes right through price represent a directionless sideways market
        asset_data = wb.DataReader(self.asset_name, data_source=self.source, start=self.start_date, end=self.end_date)

        asset_data[str(self.long_window)] = asset_data[self.data_column].rolling(window=self.long_window, min_periods=0).mean()
        asset_data[str(self.short_window)] = asset_data[self.data_column].rolling(window=self.short_window, min_periods=0).mean()

        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
        ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=5, colspan=1, sharex=ax1)

        ax1.plot(asset_data.index, asset_data[self.data_column])
        ax1.plot(asset_data.index, asset_data[str(self.long_window)])
        ax1.plot(asset_data.index, asset_data[str(self.short_window)])
        ax2.plot(asset_data.index, asset_data['Volume'])

        plt.show()

    def moving_average_trading_strategy_candlestick_plot(self):
        asset_data = wb.DataReader(self.asset_name, data_source=self.source, start=self.start_date, end=self.end_date)

        asset_data_ohlc = asset_data[self.data_column].resample(self.resample).ohlc()
        asset_data_volume = asset_data['Volume'].resample(self.resample).sum()

        asset_data[str(self.long_window)] = asset_data[self.data_column].rolling(window=self.long_window, min_periods=0).mean()
        asset_data[str(self.short_window)] = asset_data[self.data_column].rolling(window=self.short_window, min_periods=0).mean()

        asset_data_ohlc.reset_index(inplace=True)
        asset_data_ohlc['Date'] = asset_data_ohlc['Date'].map(mdates.date2num)

        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
        ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=5, colspan=1, sharex=ax1)
        ax1.plot(asset_data.index, asset_data[str(self.long_window)])
        ax1.plot(asset_data.index, asset_data[str(self.short_window)])
        ax1.xaxis_date()

        candlestick_ohlc(ax1, asset_data_ohlc.values, width=2, colorup='g')
        ax2.fill_between(asset_data_volume.index.map(mdates.date2num), asset_data_volume.values, 0)
        plt.show()

    def moving_average_trading_strategy_with_buy_and_sell_signals(self):

        asset_data = wb.DataReader(self.asset_name, data_source=self.source, start=self.start_date, end=self.end_date)

        signals = pd.DataFrame(index=asset_data.index)
        signals['signal'] = 0.0
        signals['short_mavg'] = asset_data[self.data_column].rolling(window=self.short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = asset_data[self.data_column].rolling(window=self.long_window, min_periods=1, center=False).mean()

        # Generate a signal when the short MA crosses the long MA
        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()
        print(signals)

        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Price')
        asset_data['Close'].plot(ax=ax1, color='r', lw=2.)

        signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
        # plotting buy signal
        ax1.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0], '^', markersize=10, color='m')
        # plotting sell signal
        ax1.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0], 'v', markersize=10, color='k')

        plt.show()
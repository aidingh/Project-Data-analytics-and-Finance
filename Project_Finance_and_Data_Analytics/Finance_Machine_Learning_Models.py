from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from scipy.stats import linregress
import math
from scipy.stats import norm
import seaborn as sn
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
import pickle
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class Finance_Machine_Learning_Models:
    forecast_out = 0
    data_column = ''

    def __init__(self, asset_name=None, source=None, start_date=None, end_date=None):
        self.asset_name = asset_name
        self.source = source
        self.start_date = start_date
        self.end_date = end_date


    def prepare_data_set(self):
        asset = wb.DataReader(self.asset_name, data_source=self.source, start=self.start_date, end=self.end_date)
        df = asset[[self.data_column]]

        df['Prediction'] = df[['Adj Close']].shift(-self.forecast_out)

        X = np.array(df.drop(['Prediction'], 1))

        X = X[:-self.forecast_out]
        y = np.array(df['Prediction'])

        # Get all of the y values except the last 'n' rows
        y = y[:-self.forecast_out]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return x_train, x_test, y_train, y_test, df

    def forecast_with_SVM(self, x_train, x_test, y_train, y_test, df):
        # Create and train the Support Vector Machine (Regressor)
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        svr_rbf.fit(x_train, y_train)

        svm_confidence_r2 = svr_rbf.score(x_test, y_test)
        print("svm confidence: ", svm_confidence_r2)

        x_forecast = np.array(df.drop(['Prediction'], 1))[-self.forecast_out:]
        svm_prediction = svr_rbf.predict(x_forecast)
        print(svm_prediction, 'svm predict')

        return svm_prediction, svm_confidence_r2

    def forecast_with_linear_reg(self, x_train, x_test, y_train, y_test, df):
        # Create and train the Linear Regression  Model
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        lr_confidence_r2 = lr.score(x_test, y_test)
        print("lr confidence: ", lr_confidence_r2)

        x_forecast = np.array(df.drop(['Prediction'], 1))[-self.forecast_out:]

        lr_prediction = lr.predict(x_forecast)
        print(lr_prediction, 'lr predict')

        return lr_prediction, lr_confidence_r2

    def linear_reg_for_market(self, independet_var_to_predict):
        housing_data = pd.read_excel('Housing.xlsx')
        print(housing_data)

        X = housing_data['House Size (sq.ft.)']
        X_max = housing_data['House Size (sq.ft.)'].max()
        print(X_max)

        Y = housing_data['House Price']
        Y_max = housing_data['House Price'].max()
        print(Y_max)

        corr_val = housing_data['House Size (sq.ft.)'].corr(housing_data['House Price'])
        print(corr_val)

        X1 = sm.add_constant(X)
        reg = sm.OLS(Y, X1).fit()
        print(reg.summary())

        housing_data.plot.scatter(x='House Size (sq.ft.)', y='House Price', c='DarkBlue')
        plt.scatter(X, Y)
        plt.axis([0, X_max + 300, 0, 1500000])

        slope, intercept, r_value, p_value, stderr = linregress(X, Y)
        # ---y_hat = b_0 + b_1*X_i---
        # b_0 = intercept, b_0 = Slope , X_i = new value of size to predict price
        print(intercept, ' intercept')
        print(r_value ** 2, ' r_value')
        b_0 = intercept
        b_1 = slope

        X_i = independet_var_to_predict
        y_hat = b_0 + b_1 * X_i
        print(y_hat, ' y_hat')

        plt.plot(X, reg.fittedvalues)
        plt.show()

        return y_hat

    def Monte_carlo_forecast(self, t_intervals, iterations):
        data = pd.DataFrame()
        data[self.asset_name] = wb.DataReader(self.asset_name, data_source=self.source, start=self.start_date, end=self.end_date)[self.data_column]

        log_returns = np.log(1 + data.pct_change())

        data.plot(figsize=(10, 6))
        log_returns.plot(figsize=(10, 6))
        log_returns.plot.kde()

        u = log_returns.mean()
        var = log_returns.var()
        stdev = log_returns.std()

        drift = u - (0.5 * var)

        np.array(drift)
        np.array(stdev)

        x = np.random.rand(10, 2)
        Z = norm.ppf(x)

        daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(t_intervals, iterations)))

        S0 = data.iloc[-1]
        price_list = np.zeros_like(daily_returns)
        price_list[0] = S0

        for t in range(1, t_intervals):
            price_list[t] = price_list[t - 1] * daily_returns[t]

        print(price_list, 'Price list')

        plt.figure(figsize=(15, 6))
        plt.plot(price_list)
        plt.show()

        return price_list

    def multiple_regression_model(self, econ_data):

        econ_df_after = econ_data.drop(['birth_rate', 'final_consum_growth', 'gross_capital_formation', 'Year'], axis=1)

        X = econ_df_after.drop('gdp_growth', axis=1)
        Y = econ_df_after[['gdp_growth']]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

        regression_model = LinearRegression()
        regression_model.fit(X_train, y_train)

        intercept = regression_model.intercept_[0]
        coefficent = regression_model.coef_[0][0]

        print("The intercept for model is {:.4}".format(intercept))
        print('-' * 100)

        #loop b0, b1, bn: y_hat = x0*b0 + x1*b1 + xn*bn + error
        #Here i get by coeffs printed
        for coef in zip(X.columns, regression_model.coef_[0]):
            print("The Coefficient for {} is {:.2}".format(coef[0], coef[1]))

        return regression_model, X_test, y_test, X, Y

    def evaluate_multiple_regression_model(self, X, Y, y_test, y_hat):
        # Evaluate the model
        X2 = sm.add_constant(X)

        model = sm.OLS(Y, X2)
        est = model.fit()
        print(est.summary())

        #mean squared error
        model_mse = mean_squared_error(y_test, y_hat)
        #mean absolute error
        model_mae = mean_absolute_error(y_test, y_hat)
        #root mean squared error
        model_rmse = math.sqrt(model_mse)

        print("MSE {:.3}".format(model_mse))
        print("MAE {:.3}".format(model_mae))
        print("RMSE {:.3}".format(model_rmse))

        model_r2 = r2_score(y_test, y_hat)
        print("R2: {:.2}".format(model_r2))

        # check for the normality of the residuals
        sm.qqplot(est.resid, line='s')
        plt.show()


    def predict_with_multiple_linear_model(self, regression_model, X_test, predict_val):
        y_hat = regression_model.predict(X_test)
        predicted_vals = y_hat[:predict_val]
        return y_hat, predicted_vals

    def correlation_matrix_for_feature_exploration(self, data):
        data_corr = data.corr()
        sn.heatmap(data_corr, annot=True)
        plt.show()

    def variance_inflation_factor(self, econ_data):
        # Variance_inflation_factor which is a measure of how much a particular variable is contributing to the standard error in the regression model.
        # When significant multicollinearity exists, the variance inflation factor will be huge for the variables in the calculation.
        # General rule is to drop the features above the value of 5
        econ_data_before = econ_data
        X1 = sm.tools.add_constant(econ_data_before)
        series_before = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index=X1.columns)

        print('DATA BEFORE')
        print('-' * 100)
        print(series_before)

        econ_data_after = econ_data.drop(
            ['gdp_growth', 'birth_rate', 'final_consum_growth', 'gross_capital_formation', 'Year'], axis=1)
        X2 = sm.tools.add_constant(econ_data_after)
        series_after = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index=X2.columns)

        print('DATA After')
        print('-' * 100)
        print(series_after)

        # define the plot
        pd.plotting.scatter_matrix(econ_data_after, alpha=1, figsize=(30, 20))
        plt.show()

        desc_df = econ_data.describe()

        desc_df.loc['+3_std'] = desc_df.loc['mean'] + (desc_df.loc['std'] * 3)
        desc_df.loc['-3_std'] = desc_df.loc['mean'] - (desc_df.loc['std'] * 3)

        print(desc_df)

    def prepare_data_set_multiple_reg(self):
        # Many to one relationship
        # Step 1: Find correlation relationships
        # Step 2: Scatter plot the data
        # Step 3: Simple regression on the varibles one by one
        # Ideal is for all independent variables to be correlated with the dependent variable but not with each other

        econ_data = pd.read_excel('korea_data.xlsx')
        econ_data = pd.DataFrame(econ_data)
        econ_data = econ_data.replace('..', np.nan)
        econ_data = econ_data.astype(float)

        # Check for null values
        isNull = econ_data.isnull().any()
        print(isNull)

        # change column names, they are to long and cause trouble while plotting the data, hussle
        column_names = {'Unemployment, total (% of total labor force) (national estimate)': 'unemployment',
                        'GDP growth (annual %)': 'gdp_growth',
                        'Gross capital formation (% of GDP)': 'gross_capital_formation',
                        'Population growth (annual %)': 'pop_growth',
                        'Birth rate, crude (per 1,000 people)': 'birth_rate',
                        'Broad money growth (annual %)': 'broad_money_growth',
                        'Final consumption expenditure (annual % growth)': 'final_consum_growth',
                        'General government final consumption expenditure (annual % growth)': 'gov_final_consum_growth',
                        'Gross capital formation (annual % growth)': 'gross_cap_form_growth',
                        'Households and NPISHs Final consumption expenditure (annual % growth)': 'hh_consum_growth'}

        econ_data = econ_data.rename(columns=column_names, errors="raise")

        self.correlation_matrix_for_feature_exploration(econ_data)
        self.variance_inflation_factor(econ_data)

        return econ_data

    def save_model(self, model_name, regression_model):
        with open(model_name +'.sav', 'wb') as f:
            pickle.dump(regression_model, f)

    def load_model(self, model_name):
        with open(model_name +'.sav', 'rb') as pickle_file:
            regression_model = pickle.load(pickle_file)
            return regression_model

    def make_predict(self, regression_model, X_test, value):
        predicted_val = regression_model.predict([X_test.loc[value]])
        return predicted_val
import pandas as pd
import numpy as np
from datetime import datetime
import arch


FILE_PATH = './spx_option_data/'


def load_market_data(years, exclude_itm = False):
    option_data_ori = None
    for yr in years:
        if option_data_ori is None:
            option_data_ori = pd.read_csv(FILE_PATH+'spx'+yr+'.csv')
        else:
            option_data_ori = pd.concat([option_data_ori, pd.read_csv(FILE_PATH +'spx'+yr+'.csv')])
    option_data_ori.columns = [x[1:] if ' ' == x[0] else x for x in option_data_ori.columns]

    # convert to dates
    option_data_ori['expiration'] = option_data_ori['expiration'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))

    #get quartly expired
    option_data_ori = option_data_ori[option_data_ori['expiration'].dt.month.isin([3, 6, 9 ,12])]

    # option_data_ori = option_data_ori[(option_data_ori['expiration'].dt.month == 3) | (option_data_ori['expiration'].dt.month == 6) | (option_data_ori['expiration'].dt.month == 9)| (option_data_ori['expiration'].dt.month == 12)]
    option_data_ori['quotedate'] = option_data_ori['quotedate'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))

    # exclude options with low trading volume
    option_data_ori = option_data_ori[option_data_ori['optionvolume'] >= 5]

    # exclude options trading at minimum price tick
    option_data_ori = option_data_ori[option_data_ori['last'] > 0.1]
    option_data_ori = option_data_ori[~option_data_ori['optionroot'].str.contains("LS")]

    # tenor & moneyness
    option_data_ori['tenor'] = option_data_ori[['expiration', 'quotedate']].apply(lambda x: (x[0] - x[1]).days / 365, axis=1)
    # expire in more than two weeks and recent two quarter
    option_data_ori = option_data_ori[(option_data_ori['tenor'] > (14 / 365)) & (option_data_ori['tenor'] < 0.5)]
    option_data_ori['moneyness'] = option_data_ori[['strike', 'underlying_last', 'tenor']].apply(lambda x: np.log(x[0] / x[1]) / np.sqrt(x[2]), axis=1)

    # exclude ITM calls and ITM puts
    if exclude_itm:
        otm_calls = option_data_ori[(option_data_ori['strike']>option_data_ori['underlying_last'])&(option_data_ori['optiontype']=='call')]
        otm_puts = option_data_ori[(option_data_ori['strike']<option_data_ori['underlying_last'])&(option_data_ori['optiontype']=='put')]
        option_data_ori = pd.concat([otm_calls, otm_puts])
        print('otm call',otm_calls['delta'].mean())
        print('otmput',otm_puts['delta'].mean())

    return option_data_ori


def load_risk_free_rates():
    risk_free_rates = pd.read_csv('./risk_free_rates_data/FRB_H15.csv')
    risk_free_rates.columns = [x[1:] if ' ' in x else x for x in risk_free_rates.columns]
    risk_free_rates['quotedate'] = risk_free_rates['quotedate'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    return risk_free_rates


def index_forecast_egarch(option_insample_data, option_outsample_data):
    option_data = pd.concat([option_insample_data, option_outsample_data], ignore_index=True)

    # sort by quotedate
    quotedate_sorted = list(set(option_data['quotedate']))
    quotedate_sorted.sort()

    # sort by in-sample quotedate
    quotedate_insample_sorted = list(set(option_insample_data['quotedate']))
    quotedate_insample_sorted.sort()

    # sort by in-sample quotedate
    quotedate_outsample_sorted = list(set(option_outsample_data['quotedate']))
    quotedate_outsample_sorted.sort()

    # log returns
    log_returns = []
    for (today, tmr) in zip(quotedate_sorted[:-1], quotedate_sorted[1:]):
        option_data_today = option_data[option_data['quotedate'] == today].iloc[0]
        option_data_tmr = option_data[option_data['quotedate'] == tmr].iloc[0]
        log_returns.append(np.log(option_data_tmr['underlying_last'] / option_data_today['underlying_last']))
    log_returns = pd.DataFrame(log_returns, index=quotedate_sorted[1:])

    am = arch.arch_model(log_returns, mean='AR', lags=1, vol='EGARCH', p=1)

    res = am.fit(last_obs=quotedate_insample_sorted[-1])
    f = res.forecast()
    print('underlying forecast: [mean]:', f.mean)
    print('underlying forecast: [variance]', f.variance)

    return f


def get_params(option_insample_data):
    # sort by quotedate
    quotedate_sorted = list(set(option_insample_data['quotedate']))
    quotedate_sorted.sort()

    # cubic fit the first quote date
    option_data_first_date = option_insample_data[option_insample_data['quotedate'] == quotedate_sorted[0]]
    while len(option_data_first_date) ==0:
        quotedate_sorted = quotedate_sorted[1:]
        option_data_first_date = option_insample_data[option_insample_data['quotedate'] == quotedate_sorted[0]]
    init_skew_coeffs = np.polyfit(option_data_first_date['moneyness'], option_data_first_date['meanIV'], 3)[::-1]
    matrice_g_timeseries_list = []
    impliedvol_obs_timeseries_list = []

    for quotedate in quotedate_sorted:
        option_data_today = option_insample_data[option_insample_data['quotedate'] == quotedate]
        option_data_today = option_data_today[option_data_today['meanIV'] > 0.01]
        g = option_data_today['moneyness'].apply(lambda x: [x ** 0, x ** 1, x ** 2, x ** 3])
        matrice_g = np.array(g.tolist())
        impliedvol_obs = option_data_today['meanIV'].to_numpy()
        matrice_g_timeseries_list.append(matrice_g)
        impliedvol_obs_timeseries_list.append(impliedvol_obs)

    return init_skew_coeffs,  impliedvol_obs_timeseries_list, matrice_g_timeseries_list


if __name__ == '__main__':
    option_insample_data = load_market_data(['2003', '2004', '2005'], exclude_itm=True)
    option_outsample_data = load_market_data(['2006'], exclude_itm=False)     # include all quotes in backtesting

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from data_processing import load_risk_free_rates
from option_pricer import *


def backtest_portfolios(option_outsample_data, x_series, s_series, index_forecast):
    # sort by quotedate
    quotedate_sorted = list(set(option_outsample_data['quotedate']))
    quotedate_sorted.sort()

    risk_free_rates = load_risk_free_rates()
    result = dict(
        short_straddle_log_return_actuals =[],
        short_straddle_log_return_predicts = [],
        long_rr_log_return_actuals = [],
        long_rr_log_return_predicts = [],
        long_bf_log_return_actuals = [],
        long_bf_log_return_predicts = []
    )
    for (today, tmr, x, s) in zip(quotedate_sorted[:-1], quotedate_sorted[1:], x_series, s_series):

        option_data_today = option_outsample_data[option_outsample_data['quotedate'] == today]
        call_option_data_today = option_data_today[option_data_today['optiontype'] == 'call']
        put_option_data_today = option_data_today[option_data_today['optiontype'] == 'put']

        option_data_tmr = option_outsample_data[option_outsample_data['quotedate'] == tmr]
        call_option_data_tmr = option_data_tmr[option_data_tmr['optiontype'] == 'call']
        put_option_data_tmr = option_data_tmr[option_data_tmr['optiontype'] == 'put']

        # risk free rate tomorrow - assume ~= today's rate
        risk_free_rate_tmr = risk_free_rates[risk_free_rates['quotedate'] == today].squeeze()['rate'] / 100

        # underlying price tomorrow (predict & actual)
        log_return_forecast = index_forecast.mean.loc[tmr].squeeze()
        underlying_tmr_predict = option_data_today.iloc[0]['underlying_last'] * np.exp(log_return_forecast)

        underlying_tmr_actual = option_data_tmr.iloc[0]['underlying_last']
        underlying_today_actual = option_data_today.iloc[0]['underlying_last']

        # [1] short straddle
        def short_straddle_backtest():
            atm_call_today = call_option_data_today.loc[(call_option_data_today['underlying_last'] - call_option_data_today['strike']).abs().idxmin(),]
            atm_put_today = put_option_data_today.loc[(put_option_data_today['underlying_last'] - put_option_data_today['strike']).abs().idxmin(),]
            atm_call_tmr = call_option_data_tmr[
                (call_option_data_tmr['strike'] == atm_call_today['strike']) &
                (call_option_data_tmr['expiration'] == atm_call_today['expiration'])
            ]
            atm_put_tmr = put_option_data_tmr[
                (put_option_data_tmr['strike'] == atm_put_today['strike']) &
                (put_option_data_tmr['expiration'] == atm_put_today['expiration'])
            ]

            short_straddle_log_return_actual = log_return(
                ['short', 'short'],
                [atm_call_today['last'], atm_put_today['last']],
                [atm_call_tmr['last'].squeeze(), atm_put_tmr['last'].squeeze()]
            )
            # print('short_straddle_log_return_actual:', tmr, short_straddle_log_return_actual)

            # prediction
            tenor_tmr = (atm_call_today['tenor'] * 365 - 1) / 365
            moneyness_tmr = np.log(atm_call_today['strike'] / underlying_tmr_predict) / np.sqrt(tenor_tmr)
            implied_vol_tmr = x[0] + x[1] * moneyness_tmr + x[2] * (moneyness_tmr**2) + x[3] * (moneyness_tmr**3)
            atm_call_price_tmr_predict = european_option_bs_formula(
                S=underlying_tmr_predict,
                K=atm_call_today['strike'],
                T=tenor_tmr,
                sigma=implied_vol_tmr,
                r=risk_free_rate_tmr,
                type='C'
            )
            tenor_tmr = (atm_put_today['tenor'] * 365 - 1) / 365
            moneyness_tmr = np.log(atm_put_today['strike'] / underlying_tmr_predict) / np.sqrt(tenor_tmr)
            implied_vol_tmr = x[0] + x[1] * moneyness_tmr + x[2] * (moneyness_tmr**2) + x[3] * (moneyness_tmr**3)
            atm_put_price_tmr_predict = european_option_bs_formula(
                S=underlying_tmr_predict,
                K=atm_call_today['strike'],
                T=tenor_tmr,
                sigma=implied_vol_tmr,
                r=risk_free_rate_tmr,
                type='P'
            )
            short_straddle_log_return_predict = log_return(
                ['short', 'short'],
                [atm_call_today['last'], atm_put_today['last']],
                [atm_call_price_tmr_predict, atm_put_price_tmr_predict]
            )
            return short_straddle_log_return_actual, short_straddle_log_return_predict


        # [2] long risk-reversal
        def long_risk_reversal():
            otm_call_today = call_option_data_today[
                call_option_data_today['underlying_last'] < call_option_data_today['strike']
                ].loc[(call_option_data_today['delta'] - 0.25).abs().idxmin(),]
            otm_put_today = put_option_data_today[
                put_option_data_today['underlying_last'] > put_option_data_today['strike']
                ].loc[(put_option_data_today['delta'] - -0.25).abs().idxmin(),]
            otm_call_tmr = call_option_data_tmr[
                (call_option_data_tmr['strike'] == otm_call_today['strike']) &
                (call_option_data_tmr['expiration'] == otm_call_today['expiration'])
            ]
            otm_put_tmr = put_option_data_tmr[
                (put_option_data_tmr['strike'] == otm_put_today['strike']) &
                (put_option_data_tmr['expiration'] == otm_put_today['expiration'])
            ]


            # if tmr data is missing
            otm_call_tmr = otm_call_today if len(otm_call_tmr) == 0 else otm_call_tmr
            otm_put_tmr = otm_put_today if len(otm_put_tmr) == 0 else otm_put_tmr

            long_rr_log_return_actual = log_return(
                ['long', 'short'],
                [otm_call_today['last'], otm_put_today['last']],
                [otm_call_tmr['last'].squeeze(), otm_put_tmr['last'].squeeze()]
            )
            # print('long_risk_reversal_log_return_actual:', tmr, long_rr_log_return_actual)

            # prediction
            tenor_tmr = (otm_call_today['tenor'] * 365 - 1) / 365
            moneyness_tmr = np.log(otm_call_today['strike'] / underlying_tmr_predict) / np.sqrt(tenor_tmr)
            implied_vol_tmr = x[0] + x[1] * moneyness_tmr + x[2] * (moneyness_tmr**2) + x[3] * (moneyness_tmr**3)
            otm_call_price_tmr_predict = european_option_bs_formula(
                S=underlying_tmr_predict,
                K=otm_call_today['strike'],
                T=tenor_tmr,
                sigma=implied_vol_tmr,
                r=risk_free_rate_tmr,
                type='C'
            )
            tenor_tmr = (otm_put_today['tenor'] * 365 - 1) / 365
            moneyness_tmr = np.log(otm_put_today['strike'] / underlying_tmr_predict) / np.sqrt(tenor_tmr)
            implied_vol_tmr = x[0] + x[1] * moneyness_tmr + x[2] * (moneyness_tmr**2) + x[3] * (moneyness_tmr**3)
            otm_put_price_tmr_predict = european_option_bs_formula(
                S=underlying_tmr_predict,
                K=otm_put_today['strike'],
                T=tenor_tmr,
                sigma=implied_vol_tmr,
                r=risk_free_rate_tmr,
                type='P'
            )

            long_rr_log_return_predict = log_return(
                ['long', 'short'],
                [otm_call_today['last'], otm_put_today['last']],
                [otm_call_price_tmr_predict, otm_put_price_tmr_predict]
            )
            # print('long_risk_reversal_log_return_predict:', tmr, long_rr_log_return_predict)
            return long_rr_log_return_actual, long_rr_log_return_predict

        # [3] long butterfly
        def long_butterfly():
            otm_call_today = call_option_data_today[
                call_option_data_today['underlying_last'] < call_option_data_today['strike']
                ].loc[(call_option_data_today['delta'] - 0.25).abs().idxmin(),]
            otm_put_today = put_option_data_today[
                put_option_data_today['underlying_last'] > put_option_data_today['strike']
                ].loc[(put_option_data_today['delta'] - -0.25).abs().idxmin(),]
            atm_call_today = call_option_data_today.loc[(call_option_data_today['underlying_last'] - call_option_data_today['strike']).abs().idxmin(),]
            atm_put_today = put_option_data_today.loc[(put_option_data_today['underlying_last'] - put_option_data_today['strike']).abs().idxmin(),]
            otm_call_tmr = call_option_data_tmr[
                (call_option_data_tmr['strike'] == otm_call_today['strike']) &
                (call_option_data_tmr['expiration'] == otm_call_today['expiration'])
            ]
            otm_put_tmr = put_option_data_tmr[
                (put_option_data_tmr['strike'] == otm_put_today['strike']) &
                (put_option_data_tmr['expiration'] == otm_put_today['expiration'])
            ]
            atm_call_tmr = call_option_data_tmr[
                (call_option_data_tmr['strike'] == atm_call_today['strike']) &
                (call_option_data_tmr['expiration'] == atm_call_today['expiration'])
            ]
            atm_put_tmr = put_option_data_tmr[
                (put_option_data_tmr['strike'] == atm_put_today['strike']) &
                (put_option_data_tmr['expiration'] == atm_put_today['expiration'])
            ]

            # if tmr data is missing
            otm_call_tmr = otm_call_today if len(otm_call_tmr) == 0 else otm_call_tmr
            otm_put_tmr = otm_put_today if len(otm_put_tmr) == 0 else otm_put_tmr
            atm_call_tmr = atm_call_today if len(atm_call_tmr) == 0 else atm_call_tmr
            atm_put_tmr = atm_put_today if len(atm_put_tmr) == 0 else atm_put_tmr

            long_bf_log_return_actual = log_return(
                ['long', 'long', 'short', 'short'],
                [otm_call_today['last'], otm_put_today['last'], atm_call_today['last'], atm_put_today['last']],
                [otm_call_tmr['last'].squeeze(), otm_put_tmr['last'].squeeze(), atm_call_tmr['last'].squeeze(), atm_put_tmr['last'].squeeze()]
            )
            # prediction
            tenor_tmr = (otm_call_today['tenor'] * 365 - 1) / 365
            moneyness_tmr = np.log(otm_call_today['strike'] / underlying_tmr_predict) / np.sqrt(tenor_tmr)
            implied_vol_tmr = x[0] + x[1] * moneyness_tmr + x[2] * (moneyness_tmr**2) + x[3] * (moneyness_tmr**3)
            otm_call_price_tmr_predict = european_option_bs_formula(
                S=underlying_tmr_predict,
                K=otm_call_today['strike'],
                T=tenor_tmr,
                sigma=implied_vol_tmr,
                r=risk_free_rate_tmr,
                type='C'
            )
            tenor_tmr = (otm_put_today['tenor'] * 365 - 1) / 365
            moneyness_tmr = np.log(otm_put_today['strike'] / underlying_tmr_predict) / np.sqrt(tenor_tmr)
            implied_vol_tmr = x[0] + x[1] * moneyness_tmr + x[2] * (moneyness_tmr**2) + x[3] * (moneyness_tmr**3)
            otm_put_price_tmr_predict = european_option_bs_formula(
                S=underlying_tmr_predict,
                K=otm_put_today['strike'],
                T=tenor_tmr,
                sigma=implied_vol_tmr,
                r=risk_free_rate_tmr,
                type='P'
            )
            tenor_tmr = (atm_call_today['tenor'] * 365 - 1) / 365
            moneyness_tmr = np.log(atm_call_today['strike'] / underlying_tmr_predict) / np.sqrt(tenor_tmr)
            implied_vol_tmr = x[0] + x[1] * moneyness_tmr + x[2] * (moneyness_tmr**2) + x[3] * (moneyness_tmr**3)
            atm_call_price_tmr_predict = european_option_bs_formula(
                S=underlying_tmr_predict,
                K=atm_call_today['strike'],
                T=tenor_tmr,
                sigma=implied_vol_tmr,
                r=risk_free_rate_tmr,
                type='C'
            )
            tenor_tmr = (atm_put_today['tenor'] * 365 - 1) / 365
            moneyness_tmr = np.log(atm_put_today['strike'] / underlying_tmr_predict) / np.sqrt(tenor_tmr)
            implied_vol_tmr = x[0] + x[1] * moneyness_tmr + x[2] * (moneyness_tmr**2) + x[3] * (moneyness_tmr**3)
            atm_put_price_tmr_predict = european_option_bs_formula(
                S=underlying_tmr_predict,
                K=atm_call_today['strike'],
                T=tenor_tmr,
                sigma=implied_vol_tmr,
                r=risk_free_rate_tmr,
                type='P'
            )
            long_bf_log_return_predict = log_return(
                ['long', 'long', 'short', 'short'],
                [otm_call_today['last'], otm_put_today['last'], atm_call_today['last'], atm_put_today['last']],
                [otm_call_price_tmr_predict, otm_put_price_tmr_predict, atm_call_price_tmr_predict, atm_put_price_tmr_predict]
            )
            return long_bf_log_return_actual, long_bf_log_return_predict

        short_straddle_log_return_actual, short_straddle_log_return_predict = short_straddle_backtest()
        long_rr_log_return_actual, long_rr_log_return_predict = long_risk_reversal()
        long_bf_log_return_actual, long_bf_log_return_predict = long_butterfly()

        result['short_straddle_log_return_actuals'].append(short_straddle_log_return_actual)
        result['short_straddle_log_return_predicts'].append(short_straddle_log_return_predict)

        result['long_rr_log_return_actuals'].append(long_rr_log_return_actual)
        result['long_rr_log_return_predicts'].append(long_rr_log_return_predict)

        result['long_bf_log_return_actuals'].append(long_bf_log_return_actual)
        result['long_bf_log_return_predicts'].append(long_rr_log_return_actual)

    return pd.DataFrame(result, index=quotedate_sorted[:-1])

def log_return(asset_positions, t0_assets, t1_assets):  # non-negative asset prices
    t0_market_value = 0
    for t0_asset in t0_assets:
        t0_market_value += t0_asset
    portfolio_log_return = 0
    for t0_asset, t1_asset, asset_position in zip(t0_assets, t1_assets, asset_positions):
        portfolio_log_return += t0_asset / t0_market_value * np.log(t1_asset / t0_asset) * (1 if asset_position == 'long' else -1)
    return portfolio_log_return

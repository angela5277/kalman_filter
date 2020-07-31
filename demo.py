from kalman_filter_im import KalmanFilterVolSkew
from data_processing import load_market_data, get_params, index_forecast_egarch
import numpy as np
import pickle
from backtest_portfolios import backtest_portfolios
from sklearn.metrics import mean_squared_error, r2_score

def kalman_filter_insample(para_func):
    data = load_market_data(['2003','2003', '2004', '2005'], exclude_itm=True)
    init_skew_coeffs, impliedvol_obs_timeseries, matrice_g_timeseries = get_params(data)
    kf_model = KalmanFilterVolSkew(initial_state_mean=init_skew_coeffs)
    estimated_paras = para_func(init_skew_coeffs, impliedvol_obs_timeseries, matrice_g_timeseries)
    kf_model.transition_matrix, kf_model.transition_offset, kf_model.C, kf_model.observation_covariance = kf_model.unpack_params(
        estimated_paras)
    kf_model.transition_covariance = np.dot(kf_model.C, kf_model.C.T)
    return kf_model


def get_para_estimates(init_skew_coeffs,impliedvol_obs_timeseries, matrice_g_timeseries):
    mod = KalmanFilterVolSkew(initial_state_mean=init_skew_coeffs)
    model_result = mod.fit(observations=impliedvol_obs_timeseries,
                              observation_matrices=matrice_g_timeseries)
    return model_result.x

def get_estimated_paras_from_paper(*args):
    U = [0.9268, 0.9671, 0.8612, 0.7852]
    mu = [0.2165, 0.1945, 0.1152, 0.1530]
    cov = [1 * 0.0113 * 0.0113, -0.2459 * 0.0113 * 0.0091, -0.3032 * 0.0113 * 0.0286, -0.1794 * 0.0113 * 0.0410,
           1 * 0.0091 * 0.0091, 0.1845 * 0.0091 * 0.0286, -0.1489 * 0.0091 * 0.0410,
           1 * 0.0286 * 0.0286, 0.6871 * 0.0286 * 0.0410,
           1 * 0.0410 * 0.0410]
    transition_covariance = np.eye(4)
    transition_covariance[np.triu_indices(4)] = cov
    transition_covariance[np.tril_indices(4, -1)] = transition_covariance.T[np.tril_indices(4, -1)]
    C = np.linalg.cholesky(transition_covariance)
    observatipn_covariance = np.square(0.00175)
    return np.concatenate([U, mu, C[np.tril_indices(4)], [observatipn_covariance]])


def plot_filter_result(filtered_state_means, filtered_state_covariances):
    import matplotlib.pyplot as plt
    ftitle = ['Filtered path ATM volatility', 'Filtered path Slope coefficient', 'Filtered path Curvature coefficient',
              'Filtered path Skewness coefficient']
    fig, ax = plt.subplots(nrows=2, ncols=2)
    for i in range(filtered_state_means.shape[1]):
        if i == 3:
            test = True
        ax_t = ax[i // 2, (i) % 2]
        l1, = ax_t.plot(filtered_state_means[:, i], color='blue')
        l2, = ax_t.plot(filtered_state_means[1:, i] + np.sqrt(filtered_state_covariances[1:, i, i]) * 2, color='r')
        ax_t.plot(filtered_state_means[:, i] - np.sqrt(filtered_state_covariances[1:, i, i]) * 2, color='r')
        ax_t.title.set_text(ftitle[i])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__' :
    #esimtate using mle
    # filtered_state_means, filtered_state_covariances = kalman_filter_insample(get_para_estimates)
    #training
    kf_model = kalman_filter_insample(get_estimated_paras_from_paper)

    # market data loading\
    # (load from dump)
    f = open('./pickled_data/observations_full.pkl', 'rb')
    option_all_data, option_insample_data, option_outsample_data = pickle.load(f)

    # (load from dump)
    f = open('./pickled_data/data_full.pkl', 'rb')
    init_skew_coeffs, impliedvol_obs_timeseries, matrice_g_timeseries, moneyness_list = pickle.load(f)

    # predict insample and outsample
    filtered_state_means, filtered_state_covariances = kf_model.filter(impliedvol_obs_timeseries, matrice_g_timeseries)

    plot_filter_result(filtered_state_means,filtered_state_covariances)

    # cut-off date for backtesting
    cutoff_date = option_outsample_data.iloc[0]['quotedate']
    date_series = sorted(set(option_all_data['quotedate']))
    cutoff_idx = date_series.index(cutoff_date)

    # underlying index's forecast
    index_forecast = index_forecast_egarch(option_insample_data, option_outsample_data)

    # backtest, have bug in finding tmr option data
    backtest_result = backtest_portfolios(option_outsample_data, filtered_state_means[cutoff_idx:], filtered_state_covariances[cutoff_idx:], index_forecast)
    for x in ['short_straddle', 'long_rr', 'long_bf']:
        print(x)
        print('Mean squared error: %.4f'
              % mean_squared_error(backtest_result[x+'_log_return_actuals'], backtest_result[x+'_log_return_predicts']))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.4f'
              % r2_score(backtest_result[x+'_log_return_actuals'], backtest_result[x+'_log_return_predicts']))

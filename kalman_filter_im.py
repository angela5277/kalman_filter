import numpy as np
from scipy.stats import norm
from scipy import optimize
from utils import log_multivariate_normal_density
from data_processing import load_market_data, get_params
import logging

class KalmanFilterVolSkew:
    def __init__(self,  initial_state_mean):

        self.initial_state_mean = initial_state_mean
        self.n_dim_state = len(initial_state_mean)
        self.transition_matrix = 0.9 * np.eye(self.n_dim_state)
        self.transition_covariance = 0.9 * np.eye(self.n_dim_state)
        self.C =  np.linalg.cholesky(self.transition_covariance)
        self.transition_offset = np.zeros(self.n_dim_state)
        self.observation_covariance = 0.5
        self.initial_state_covariance = np.eye(self.n_dim_state)

    def start_params(self):
        return self.pack_params(self.transition_matrix, self.transition_offset, self.C, self.observation_covariance)

    def pack_params(self,transition_matrix, transition_offset, C, observation_covariance):
        return np.concatenate([np.diag(transition_matrix),transition_offset, C[np.tril_indices(self.n_dim_state)], [observation_covariance]])

    def unpack_params(self, params):
        idx = 0
        transition_matrix = np.diag(params[idx:self.n_dim_state])
        idx = idx + self.n_dim_state
        transition_offset =  np.array(params[idx:idx+self.n_dim_state])
        idx = idx + self.n_dim_state
        C = np.eye(self.n_dim_state)
        n_elements = sum(range(self.n_dim_state+1))
        C[np.tril_indices(self.n_dim_state)] = params[idx:idx+n_elements]
        idx = idx + n_elements
        observation_covariance = params[idx]
        return transition_matrix, transition_offset, C, observation_covariance

    def filter(self,observations, observation_matrices):
        n_timesteps = len(observations)
        n_dim_state = len(self.initial_state_mean)
        filtered_state_means = np.zeros((n_timesteps, n_dim_state))
        filtered_state_covariances = np.zeros(
            (n_timesteps, n_dim_state, n_dim_state)
        )
        for t in range(n_timesteps):
            if t == 0:
                filtered_state_means[t] = self.initial_state_mean
                filtered_state_covariances[t] = np.dot(self.C,np.dot(np.linalg.pinv(
                                                        np.eye(self.n_dim_state) - self.transition_matrix \
                                                        + (np.eye(self.n_dim_state) - self.transition_matrix).T), self.C.T))

            else:
                (filtered_state_means[t], filtered_state_covariances[t]) = self.predict_update(self.transition_matrix,
                                    self.transition_covariance,
                                    self.transition_offset,
                                    filtered_state_means[t-1],
                                    filtered_state_covariances[t-1],
                                    observation_matrices[t-1],
                                    self.observation_covariance,observations[t-1])

        return  (filtered_state_means, filtered_state_covariances)

    def predict_update(self,transition_matrix, transition_covariance,
                    transition_offset, current_state_mean,
                    current_state_covariance,observation_matrix, observation_covariance, observation):

        T = np.dot(observation_matrix,
                   np.dot(current_state_covariance, observation_matrix.T)) \
                        + observation_covariance * np.eye(observation_matrix.shape[0])

        K =  np.dot(transition_matrix,
                np.dot(current_state_covariance,
                   np.dot(observation_matrix.T,
                          np.linalg.pinv(T))))

        corrected_state_mean = np.dot(transition_matrix,
                                np.dot(np.eye(transition_matrix.shape[0]) \
                                 - np.dot(current_state_covariance,
                                    np.dot(observation_matrix.T,
                                        np.dot(np.linalg.pinv(T),observation_matrix))), current_state_mean)) \
                                         + np.dot(K, observation)\
                                            + np.dot(np.eye(transition_matrix.shape[0])
                                                     - transition_matrix, transition_offset.T)

        corrected_state_covariance  = np.dot(transition_matrix,
                                    np.dot(current_state_covariance - (
                                    np.dot(current_state_covariance,np.dot(observation_matrix.T,
                                    np.dot(np.linalg.pinv(T),
                                    np.dot(observation_matrix,current_state_covariance))))),
                                    transition_matrix.T))  + transition_covariance
        return (corrected_state_mean,
                corrected_state_covariance)

    def loglikelihood(self, params):
        self.transition_matrix, self.transition_offset,self.C, self.observation_covariance = self.unpack_params(
            params)
        self.transition_covariance = np.dot(self.C, self.C.T)
        filtered_state_means, filtered_state_covariances = self.filter(self.observations, self.observation_matrices)
        # get likelihoods for each time step
        loglikelihoods = self._loglikelihoods(
          self.observation_matrices, self.observation_covariance,
          filtered_state_means, filtered_state_covariances, self.observations
        )
        return np.sum(loglikelihoods)

    def _loglikelihoods(self, observation_matrices,
                        observation_covariance, predicted_state_means,
                        predicted_state_covariances, observations):
        n_timesteps = len(observations)
        loglikelihoods = np.zeros(n_timesteps)
        for t in range(n_timesteps):
            observation = observations[t]
            observation_matrix = observation_matrices[t]
            predicted_state_mean =  predicted_state_means[t]
            predicted_state_covariance = predicted_state_covariances[t]
            predicted_observation_mean = (
                    np.dot(observation_matrix,
                           predicted_state_mean)
            )
            predicted_observation_covariance = (
                    np.dot(observation_matrix,
                           np.dot(predicted_state_covariance,
                                  observation_matrix.T))
                    + observation_covariance  * np.eye(observation_matrix.shape[0])
            )
            loglikelihoods[t] = log_multivariate_normal_density(
                observation[np.newaxis, :],
                predicted_observation_mean[np.newaxis, :],
                predicted_observation_covariance[np.newaxis, :, :]
            )
        return loglikelihoods

    def fit(self, observations, observation_matrices):
        self.observations = observations
        self.observation_matrices = observation_matrices
        def f(params):
            logging.info('loglikely hood', - self.loglikelihood(params)/ len(self.observations))
            return - self.loglikelihood(params)/ len(self.observations)
        start_params = self.start_params()
        bnds = []
        #set bound for transition matrix
        bnds.extend([(0,0.99999) for i in range(0,4)])
        #set bound for offset
        bnds.extend([(-0.5,0.5) for i in range(0,4)])
        #set bound for covariance
        bnds.extend([(1e-7,0.5) for i in range(0,len(start_params)-8)])

        res = optimize.minimize(f, start_params,method='SLSQP', bounds=bnds, options={'maxiter': 10,'ftol': 1e-6,'disp': True})
        return res


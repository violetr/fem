import numpy as np
import random
import pandas as pd

# MATH and STATS:
import math
from scipy.stats import multivariate_normal

from sklearn.cluster import KMeans

def compute_condition_prob_matrix(X, theta_old):
    # computes the conditional probability of the model
    #
    # INPUT:
    # X: (n, p) np.array 
    #     data set, one observation by row 
    # theta_old: pd series
    #     current parameters of model: alpha, mean, sigma, tau
    #
    # OUTPUT:
    # cond_prob_matrix: (n, K) np.array
    #     (cond_prob_matrix)_ik = P(Z_i=k|X_i=x_i)

    n, p = X.shape
    K = len(theta_old['alpha'])
    cond_prob_matrix = np.zeros((n,K))

    for i in range(n):
        
        for k in range(K):

            normal = multivariate_normal(mean = theta_old['mu'][k], 
                                             cov = theta_old['tau'][i][k] * theta_old['Sigma'][k])
            density = normal.pdf(X[i,:])

            cond_prob_matrix[i, k] = theta_old['alpha'][k] * density
        
        if np.sum(cond_prob_matrix[i,:]) == 0: # if point is likely outlier
            cond_prob_matrix[i, :] = theta_old['alpha'] # "prior"
            
        cond_prob_matrix[i, :] /= (np.sum(cond_prob_matrix[i,:]))

    return cond_prob_matrix

def FREM(K, X, max_iter = 200, rand_initialization = False, version = 1, max_iter_fp = 20):
    # Computes the parameters of the model
    # theta = (alpha, mu, sigma, tau) in a EM like fashion
    #
    # INPUT:
    # K: integer>0
    #    number of distributions (clusters)
    # X: (n, p) np.array
    #    data, one observation each row
    # max_iter: integer>0
    #    maximum number of iterations
    # rand_initialization: bool
    #    True if random initialization
    #    False if K-Means initialization
    # version: {1, 2, 3, 4}
    #    version of the algorithm
    #    1: with old, old, not square root
    #    2: with old, new, not square root
    #    3: with old, old, with square root
    #    4: with old, new, with square root
    # max_iter_fp: integer>0
    #    maximum number of fixed-point iterations
    #
    # OUTPUT:
    # cond_prob_matrix.shape = n x K
    # (cond_prob_matrix)_ik = P(Z_i=k|X_i=x_i)

    n, p = X.shape

    if rand_initialization:
        alpha = np.random.rand(3)
        alpha /= np.sum(alpha) 
        mu = (np.amax(X, axis=0)-np.amin(X, axis=0)) * np.random.random_sample((K, p))+ np.amin(X, axis=0)
        Sigma = np.zeros((K, p, p))
        tau = np.ones((n, K))
        for k in range(K):
            Sigma[k] = np.eye(p)

    else:
        kmeans = KMeans(n_clusters=K, max_iter=200).fit(X)
        alpha = np.zeros((K,))
        mu = np.zeros((K, p))
        Sigma = np.zeros((K, p, p))

        for k in range(K):
            nk = np.count_nonzero(kmeans.labels_ == k)
            alpha[k] = float(nk)/float(n)
            mu[k] = kmeans.cluster_centers_[k]
            X_k = X[kmeans.labels_ == k, :]
            Sigma[k] = np.eye(p) # cov result in nan sometimes
           
        tau = np.ones((n, K))

    theta_old = pd.Series([alpha, mu, Sigma, tau], index=['alpha', 'mu', 'Sigma', 'tau'])
    convergence = False

    ite = 0
    delta_history_mu = []
    delta_history_sigma = []
    while not(convergence) and  ite<max_iter:
        
        if ite % 10 == 0:
            print("ite: ", ite)
            
        # E-step:
        cond_prob = compute_condition_prob_matrix(X, theta_old)

        # M-step:
        alpha_new = np.zeros((K,))
        mu_new = np.zeros((K, p))
        Sigma_new = np.zeros((K, p, p))
        tau_new = np.ones((n, K))

        delta_sigma = np.ones((K, max_iter_fp))
        delta_mu = np.ones((K, max_iter_fp))

        for k in range(K):

            # UPDATE alpha:
            alpha_new[k] = np.mean(cond_prob[:, k])

            # Fixed-point equation for Sigma and mu:
            # UPDATE mu
            # UPDATE Sigma
            mu_fixed_point = theta_old['mu'][k].copy()
            Sigma_fixed_point = theta_old['Sigma'][k].copy()
            tau_ite = np.ones((n, ))
            tau_ite_sr = np.ones((n, ))
            convergence_fp = False
            ite_fp = 1
            while not(convergence_fp) and ite_fp<max_iter_fp:
                inv_Sigma_fixed_point = np.linalg.inv(Sigma_fixed_point)
                for i in range(n):
                    sq_maha = ((X[i,:]-mu_fixed_point).T @ inv_Sigma_fixed_point @ (X[i,:]-mu_fixed_point))
                    tau_ite[i] = sq_maha / p
                    tau_ite_sr[i] = (sq_maha**(0.5))/p
                    if tau_ite[i] <  10**(-8):
                        tau_ite[i] =  10**(-8)
                    if tau_ite[i] >  10**(8):
                        tau_ite[i] =  10**(8)
                    if tau_ite_sr[i] <  10**(-8):
                        tau_ite_sr[i] =  10**(-8)
                    if tau_ite_sr[i] >  10**(8):
                        tau_ite_sr[i] =  10**(8)

                if version == 1 or version ==2:
                    Ck = (cond_prob[:, k]/tau_ite)/np.sum(cond_prob[:,k]/tau_ite)
                else: # 3 or 4
                    Ck = (cond_prob[:, k]/tau_ite_sr)/np.sum(cond_prob[:,k]/tau_ite_sr)

                mu_fixed_point_new = np.sum(np.multiply(X, Ck[:, np.newaxis]), 0)
                Sigma_sum = np.zeros((p, p))
                for i in range(n):
                    if version == 2 or version == 4:
                        sq_maha = ((X[i,:]-mu_fixed_point_new).T @ inv_Sigma_fixed_point @ (X[i,:]-mu_fixed_point_new))
                        tau_ite[i] = sq_maha / p
                        tau_ite_sr[i] = (sq_maha**(0.5))/p
                        if tau_ite[i] <  10**(-8):
                            tau_ite[i] =  10**(-8)
                        if tau_ite[i] >  10**(8):
                            tau_ite[i] =  10**(8)
                        if tau_ite_sr[i] <  10**(-8):
                            tau_ite_sr[i] =  10**(-8)
                        if tau_ite_sr[i] >  10**(8):
                            tau_ite_sr[i] =  10**(8)
                            
                    if version==1:
                        Sigma_sum += (cond_prob[i, k] * (X[i, :]-mu_fixed_point)[:, np.newaxis]@(X[i, :]-mu_fixed_point)[np.newaxis,:])/tau_ite[i]
                    if version==2:
                        Sigma_sum += (cond_prob[i, k] * (X[i, :]-mu_fixed_point_new)[:, np.newaxis]@(X[i, :]-mu_fixed_point_new)[np.newaxis,:])/tau_ite[i]
                    if version==3:
                        Sigma_sum += (cond_prob[i, k] * (X[i, :]-mu_fixed_point)[:, np.newaxis]@(X[i, :]-mu_fixed_point)[np.newaxis,:])/tau_ite_sr[i]
                    if version==4: 
                        Sigma_sum += (cond_prob[i, k] * (X[i, :]-mu_fixed_point_new)[:, np.newaxis]@(X[i, :]-mu_fixed_point_new)[np.newaxis,:])/tau_ite_sr[i]
                        
                Sigma_sum /= (n * alpha_new[k])
                Sigma_sum *= p / np.trace(Sigma_sum)

                delta_mu[k, ite_fp] = np.inner(mu_fixed_point_new-mu_fixed_point, mu_fixed_point_new-mu_fixed_point)
                delta_sigma[k, ite_fp] = (np.linalg.norm(Sigma_sum-Sigma_fixed_point, ord='fro'))
                
                convergence_fp = True
                convergence_fp = convergence_fp and (math.sqrt(np.inner(mu_fixed_point_new - mu_fixed_point_new, mu_fixed_point_new - mu_fixed_point_new)/p) < 10**(-5))
                convergence_fp = convergence_fp and (np.linalg.norm(Sigma_sum-Sigma_fixed_point, ord='fro')/p) < 10**(-5)

                mu_fixed_point = mu_fixed_point_new.copy()
                Sigma_fixed_point = Sigma_sum.copy() 
                
                ite_fp += 1

            mu_new[k] = mu_fixed_point
            Sigma_new[k] = Sigma_fixed_point 

            # UPDATE tau
            for i in range(n):
                tau_new[i][k] = ((X[i,:]-mu_new[k]).T @ np.linalg.inv(Sigma_new[k]) @ (X[i,:]-mu_new[k]))/p
                if tau_new[i][k] <  10**(-6):
                    tau_new[i][k] =  10**(-6)
                if tau_new[i][k] >  10**(6):
                    tau_new[i][k] =  10**(6)

        delta_history_mu.append(delta_mu)
        delta_history_sigma.append(delta_sigma)

        theta_new = pd.Series([alpha_new, mu_new, Sigma_new, tau_new], index=['alpha', 'mu', 'Sigma', 'tau'])

        # Check convergence:
        if ite > 5:
            convergence = True
            k = 0
            while convergence and k<K:
                convergence = convergence and math.sqrt(np.inner(theta_new['mu'][k]-theta_old['mu'][k], theta_new['mu'][k]-theta_old['mu'][k])/p) < 10**(-5)
                convergence = convergence and ((np.linalg.norm(theta_new['Sigma'][k]-theta_old['Sigma'][k], ord='fro')/(p)) < 10**(-5))
                convergence = convergence and (math.fabs(theta_new['alpha'][k]-theta_old['alpha'][k]) < 10**(-3))
                k += 1
        
        theta_old = theta_new.copy()

        ite += 1

    print('convergence: ', convergence)
    print('number of iterations: ', ite)

    return theta_new, cond_prob, delta_history_mu, delta_history_sigma


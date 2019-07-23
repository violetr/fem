import numpy as np
import random
import pandas as pd
import os

# MATH and STATS:
import math
from scipy.stats import multivariate_normal

from sklearn.cluster import KMeans

def compute_condition_prob_matrix(X, theta_old):
    # computes the conditional probability of the model
    #
    # INPUT:
    # X one observation by row (n x p)
    # theta_old actual parameters of model: alpha, mean, sigma, tau
    #
    # OUTPUT:
    # cond_prob_matrix.shape = n x K
    # (cond_prob_matrix)_ik = P(Z_i=k|X_i=x_i)

    n, p = X.shape
    K = len(theta_old['alpha'])
    cond_prob_matrix = np.zeros((n,K))

    for i in range(n):

        for k in range(K):

            if theta_old['tau'][i][k]<10**(-8):

                normal = multivariate_normal(mean = theta_old['mu'][k], cov = 10**(-8) * theta_old['Sigma'][k])
                density = normal.pdf(X[i,:])

            else:

                normal = multivariate_normal(mean = theta_old['mu'][k], cov = theta_old['tau'][i][k] * theta_old['Sigma'][k])
                density = normal.pdf(X[i,:])

            # to avoid nans:
            if density<10**(-150):
                density = 10**(-150)

            cond_prob_matrix[i, k] = theta_old['alpha'][k] * density

        cond_prob_matrix[i, :] /= (np.sum(cond_prob_matrix[i,:]))

    return cond_prob_matrix

def CG_EM(K, X, max_iter, rand_initialization, version, max_iter_fp):
    # Computes the parameters of the model
    # theta = (alpha, mu, sigma, tau)
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
    #
    # OUTPUT:
    # cond_prob_matrix.shape = n x K
    # (cond_prob_matrix)_ik = P(Z_i=k|X_i=x_i)

    n, p = X.shape

    if rand_initialization:
        alpha = np.random.rand(3)
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
    ite = 1
    delta_history_mu = []
    delta_history_sigma = []
    while not(convergence) and  ite<max_iter:

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
            for ite_fp in range(max_iter_fp):
                inv_Sigma_fixed_point = np.linalg.inv(Sigma_fixed_point)
                for i in range(n):
                    tau_ite[i] = ((X[i,:]-mu_fixed_point).T @ inv_Sigma_fixed_point @ (X[i,:]-mu_fixed_point))/p
                    if tau_ite[i] <  10**(-10):
                        tau_ite[i] =  10**(-10)
                    if tau_ite[i] >  10**(10):
                        tau_ite[i] =  10**(10)
                    tau_ite_sr[i] = (tau_ite[i])**(0.5)


                if version == 1 or version ==2:
                    Ck = (cond_prob[:, k]/tau_ite)/np.sum(cond_prob[:,k]/tau_ite)
                else: # 3 or 4
                    Ck = (cond_prob[:, k]/tau_ite_sr)/np.sum(cond_prob[:,k]/tau_ite_sr)

                mu_fixed_point_new = np.sum(np.multiply(X, Ck[:, np.newaxis]), 0)
                Sigma_sum = np.zeros((p, p))
                for i in range(n):
                    if version==1 or version==3:
                        Sigma_sum += (cond_prob[i, k] * (X[i, :]-mu_fixed_point)[:, np.newaxis]@(X[i, :]-mu_fixed_point)[np.newaxis,:])/tau_ite[i]
                    if version==2 or version==4: # 2 or 4
                        Sigma_sum += (cond_prob[i, k] * (X[i, :]-mu_fixed_point_new)[:, np.newaxis]@(X[i, :]-mu_fixed_point_new)[np.newaxis,:])/tau_ite[i]
                Sigma_sum /= (n * alpha_new[k])

                delta_mu[k, ite_fp] = np.inner(mu_fixed_point_new-mu_fixed_point, mu_fixed_point_new-mu_fixed_point)
                delta_sigma[k, ite_fp] = (np.linalg.norm(Sigma_sum-Sigma_fixed_point, ord='fro'))

                mu_fixed_point = mu_fixed_point_new.copy()
                Sigma_fixed_point = Sigma_sum.copy()

            mu_new[k] = mu_fixed_point
            Sigma_new[k] = Sigma_fixed_point * p / np.trace(Sigma_fixed_point)

            # UPDATE tau
            for i in range(n):
                tau_new[i][k] = ((X[i,:]-mu_new[k]).T @ np.linalg.inv(Sigma_new[k]) @ (X[i,:]-mu_new[k]))/p
                if tau_new[i][k] <  10**(-10):
                    tau_new[i][k] =  10**(-10)
                if tau_new[i][k] >  10**(10):
                    tau_new[i][k] =  10**(10)

        delta_history_mu.append(delta_mu)
        delta_history_sigma.append(delta_sigma)

        theta_new = pd.Series([alpha_new, mu_new, Sigma_new, tau_new], index=['alpha', 'mu', 'Sigma', 'tau'])


        # Check convergence:
        convergence = True
        k = 0
        while convergence and k<K:
            convergence = convergence and np.inner(theta_new['mu'][k]-theta_old['mu'][k], theta_new['mu'][k]-theta_old['mu'][k]) < 10**(-6)
            convergence = convergence and ((np.linalg.norm(theta_new['Sigma'][k]-theta_old['Sigma'][k], ord='fro')) < 10**(-3))
            convergence = convergence and (math.fabs(theta_new['alpha'][k]-theta_old['alpha'][k]) < 10**(-3))
            convergence = convergence and ((np.linalg.norm(theta_new['tau'][:, k]-theta_old['tau'][:, k])) < 10**(-3))
            k += 1

        theta_old = theta_new.copy()

        ite += 1

    print('convergence: ', convergence)
    print('number of iterations: ', ite)

    return theta_new, cond_prob, delta_history_mu, delta_history_sigma

import numpy as np
import random
import pandas as pd

# MATH and STATS:
import math
from scipy.stats import multivariate_normal

# for initialization of cluster's centers:
from sklearn.cluster import KMeans

class FEM():
    '''
    Implements the F-EM algorithm 
    
    Parameters
    ----------
    K : int
        The number of mixture components.
    max_iter: int
        maximum number of iterations of the algorithm.
    rand_initialization: bool
        True if random initialization
        False if K-Means initialization.
    version: {1, 2, 3, 4}
        version of the algorithm
        1: with old, old, not square root
        2: with old, new, not square root
        3: with old, old, with square root
        4: with old, new, with square root.
     max_iter_fp: integer>0
        maximum number of fixed-point iterations
    
    
    Attributes
    ----------
    alpha_ : array-like, shape (n,)
        The weights of each mixture components.
    mu_ : array-like, shape (n, p)
        The mean of each mixture component.
    Sigma_ : array-like, shape (p, p)
        The covariance of each mixture component.
    tau_ : array-like, shape (n, K)
        The collection of tau values.
    labels_ : array-like, shape (n,)
        The clustering labels of the data points.
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    n_iter_ : int
        Number of step used to reach the convergence.               
    '''
    
    def __init__(self, K, max_iter = 200, 
                 rand_initialization = False, 
                 version = 1, max_iter_fp = 20):
        self.K = K
        self.converged_ = False
        self.version = version
        self.rand_initialization = rand_initialization
        self.max_iter = max_iter
        self.max_iter_fp = max_iter_fp
        self.alpha_ = None
        self.mu_ = None
        self.Sigma_ = None
        self.tau_ = None
        self.n_iter_ = None
        self.labels_ = None
    
    def _initialize(self, X):
        
        n, p = X.shape

        if self.rand_initialization:
            self.alpha_ = np.random.rand(3)
            self.alpha_ /= np.sum(self.alpha_) 
            self.mu_ = (np.amax(X, axis=0)-np.amin(X, axis=0)) * np.random.random_sample((K, p))+ np.amin(X, axis=0)
            self.Sigma_ = np.zeros((self.K, p, p))
            self.tau_ = np.ones((n, self.K))
            for k in range(self.K):
                Sigma[k] = np.eye(p)

        else:
            kmeans = KMeans(n_clusters=self.K, max_iter=200).fit(X)
            self.alpha_ = np.zeros((self.K,))
            self.mu_ = np.zeros((self.K, p))
            self.Sigma_ = np.zeros((self.K, p, p))

            for k in range(self.K):
                nk = np.count_nonzero(kmeans.labels_ == k)
                self.alpha_[k] = float(nk)/float(n)
                self.mu_[k] = kmeans.cluster_centers_[k]
                X_k = X[kmeans.labels_ == k, :]
                self.Sigma_[k] = np.eye(p) # cov result in nan sometimes

            self.tau_ = np.ones((n, self.K))
    
    def _e_step(self, X):
        '''
        Computes the conditional probability of the model
        
        INPUT:
          X: (n, p) np.array 
             data set, one observation by row 
         theta_old: pd series
             current parameters of model: alpha, mean, sigma, tau
        
        OUTPUT:
        cond_prob_matrix: (n, K) np.array
             (cond_prob_matrix)_ik = P(Z_i=k|X_i=x_i)
        '''
        n, p = X.shape
        
        K = len(self.alpha_)
        
        cond_prob_matrix = np.zeros((n,K))

        for i in range(n):

            for k in range(K):

                normal = multivariate_normal(mean = self.mu_[k], 
                                             cov  = self.tau_[i][k] * self.Sigma_[k])
                density = normal.pdf(X[i,:])

                cond_prob_matrix[i, k] = self.alpha_[k] * density

            if np.sum(cond_prob_matrix[i,:]) == 0: # if point is likely outlier
                cond_prob_matrix[i, :] = self.alpha_ # "prior"

            cond_prob_matrix[i, :] /= (np.sum(cond_prob_matrix[i,:]))

        return cond_prob_matrix
    
    def _m_step(self, X, cond_prob):
        '''
        Computes the conditional probability of the model
        
        INPUT:
          X: (n, p) np.array 
             data set, one observation by row 
         theta_old: pd series
             current parameters of model: alpha, mean, sigma, tau
        
        OUTPUT:
        cond_prob_matrix: (n, K) np.array
             (cond_prob_matrix)_ik = P(Z_i=k|X_i=x_i)
        '''
        
        n, p = X.shape
        
        alpha_new = np.zeros((self.K,))
        mu_new = np.zeros((self.K, p))
        Sigma_new = np.zeros((self.K, p, p))
        tau_new = np.ones((n, self.K))

        for k in range(self.K):

            # UPDATE alpha:
            alpha_new[k] = np.mean(cond_prob[:, k])

            # Fixed-point equation for Sigma and mu:
            # UPDATE mu
            # UPDATE Sigma
            mu_fixed_point = self.mu_[k].copy()
            Sigma_fixed_point = self.Sigma_[k].copy()
            tau_ite = np.ones((n, ))
            tau_ite_sr = np.ones((n, ))
            convergence_fp = False
            ite_fp = 1
            while not(convergence_fp) and ite_fp<self.max_iter_fp:
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

                if self.version == 1 or self.version ==2:
                    Ck = (cond_prob[:, k]/tau_ite)/np.sum(cond_prob[:,k]/tau_ite)
                else: # 3 or 4
                    Ck = (cond_prob[:, k]/tau_ite_sr)/np.sum(cond_prob[:,k]/tau_ite_sr)

                mu_fixed_point_new = np.sum(np.multiply(X, Ck[:, np.newaxis]), 0)
                Sigma_sum = np.zeros((p, p))
                for i in range(n):

                    if self.version == 2 or self.version == 4: # if usig new estim, update denominator
                        
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

                    if self.version==1:
                        Sigma_sum += (cond_prob[i, k] * (X[i, :]-mu_fixed_point)[:, np.newaxis]@(X[i, :]-mu_fixed_point)[np.newaxis,:])/tau_ite[i]
                    if self.version==2:
                        Sigma_sum += (cond_prob[i, k] * (X[i, :]-mu_fixed_point_new)[:, np.newaxis]@(X[i, :]-mu_fixed_point_new)[np.newaxis,:])/tau_ite[i]
                    if self.version==3:
                        Sigma_sum += (cond_prob[i, k] * (X[i, :]-mu_fixed_point)[:, np.newaxis]@(X[i, :]-mu_fixed_point)[np.newaxis,:])/tau_ite_sr[i]
                    if self.version==4: 
                        Sigma_sum += (cond_prob[i, k] * (X[i, :]-mu_fixed_point_new)[:, np.newaxis]@(X[i, :]-mu_fixed_point_new)[np.newaxis,:])/tau_ite_sr[i]

                Sigma_sum /= (n * alpha_new[k])
                Sigma_sum *= p / np.trace(Sigma_sum)

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

        return alpha_new, mu_new, Sigma_new, tau_new
    
    def fit(self, X):
        
        n, p = X.shape
        
        self._initialize(X)
        
        # until here OK

        convergence = False

        ite = 0
        
        while not(convergence) and  ite<self.max_iter:

            # E-step:
            cond_prob = self._e_step(X)

            # M-step:
            alpha_new, mu_new, Sigma_new, tau_new = self._m_step(X, cond_prob)

            # Check convergence:
            if ite > 5:  # tol from fixed point should be bigger than general tolerance rate 
                convergence = True
                k = 0
                while convergence and k<self.K:
                    
                    convergence = convergence and math.sqrt(np.inner(mu_new[k]-self.mu_[k], mu_new[k]-self.mu_[k])/p) < 10**(-5)
                    convergence = convergence and ((np.linalg.norm(Sigma_new[k]-self.Sigma_[k], ord='fro')/(p)) < 10**(-5))
                    convergence = convergence and (math.fabs(alpha_new[k]-self.alpha_[k]) < 10**(-3))
                    
                    k += 1
                    
            self.alpha_ = np.copy(alpha_new)
            self.mu_ = np.copy(mu_new)
            self.Sigma_ = np.copy(Sigma_new)
            self.tau_ = np.copy(tau_new)

            ite += 1
        
        self.labels_ = np.array([i for i in np.argmax(cond_prob, axis=1)])
        self.n_iter_ = ite
        self.converged_ = convergence
        
        return(self)

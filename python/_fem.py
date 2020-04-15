import numpy as np
import random
import pandas as pd

# MATH and STATS:
import math
from scipy.stats import multivariate_normal
from scipy.stats import chi2
from scipy.stats._multivariate import _PSD  

# for initialization of cluster's centers:
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

class FEM():
    '''Implements the F-EM algorithm     
    
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
        The weight of each mixture components.
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
                 version = 1, max_iter_fp = 20, thres = None):
        self.K = K
        self.converged_ = False
        self.version = version
        self.rand_initialization = rand_initialization
        self.max_iter = max_iter
        self.max_iter_fp = max_iter_fp
        self.thres = thres
        self.alpha_ = None
        self.mu_ = None
        self.Sigma_ = None
        self.tau_ = None
        self.n_iter_ = None
        self.labels_ = None
    
    def _initialize(self, X):
        '''Initialize all the parameters of the model:
        theta = (alpha, mu, sigma, tau)
        Either randomly or with kmeans centers.
    
        Parameters
        ----------
        X: array-like, shape (n, p)
    
        '''
        
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
            
            one_point_clusters = False
            
            kmeans = KMeans(n_clusters = self.K, max_iter = 200).fit(X)
            
            for k in range(self.K):
                
                nk = np.count_nonzero(kmeans.labels_ == k)
                
                if nk <= 2 and n>10:
                    one_point_clusters = True
                    
            ite_filter = 0
            n_filter = n
        
            if one_point_clusters:
                
                tree = cKDTree(X)#tree of nearest neighbors
                KNN=4
                dd, index = tree.query(X, k=[KNN]) # query for all points in data the Kth NN, returns distances and indexes
                
                dd = np.reshape(dd, (n,))
                
                alpha_quantile = 0.95
            
                while one_point_clusters and alpha_quantile > 0.5:
                    
                    ite_filter += 1
                    
                    alpha_quantile -= (0.1) * (ite_filter - 1)
                    
                    one_point_clusters = False
                    
                    X_without_extremes = X[dd < np.quantile(dd, alpha_quantile) , :]
                    
                    n_filter = X_without_extremes.shape[0]

                    kmeans = KMeans(n_clusters=self.K, max_iter=200).fit(X_without_extremes)
                    
                    for k in range(self.K):
                
                        nk = np.count_nonzero(kmeans.labels_ == k)
                
                        if nk <= 2:
                        
                            one_point_clusters = True
            
            self.alpha_ = np.zeros((self.K,))
            self.mu_ = np.zeros((self.K, p))
            self.Sigma_ = np.zeros((self.K, p, p))            

            for k in range(self.K):
                nk = np.count_nonzero(kmeans.labels_ == k)
                self.alpha_[k] = float(nk)/float(n_filter)
                self.mu_[k] = kmeans.cluster_centers_[k]
                self.Sigma_[k] = np.eye(p) # cov result in nan sometimes

            self.tau_ = np.ones((n, self.K))                   

    
    def _e_step(self, X):
        ''' E-step of the algorithm
        Computes the conditional probability of the model
        
        Parameters
        ----------
        X: array-like, shape (n, p)
            data
    
        Returns
        ----------
        cond_prob_matrix: array-like, shape (n, K)
             (cond_prob_matrix)_ik = P(Z_i=k|X_i=x_i)
        '''
        n, p = X.shape
        
        K = len(self.alpha_)
        
        cond_prob_matrix = np.zeros((n,K))
    
        for k in range(K):
            
            psd = _PSD(self.Sigma_[k])
            prec_U, logdet = psd.U, psd.log_pdet
            diff = X - self.mu_[k]
            logdensity = -0.5 * (p * np.log(2 * np.pi) + p * np.log(self.tau_[:, k]) + logdet + p)            
            cond_prob_matrix[:, k] = np.exp(logdensity)  * self.alpha_[k]            
        
        sum_row = np.sum(cond_prob_matrix, axis = 1) 
        bool_sum_zero = (sum_row == 0)
        
        cond_prob_matrix[bool_sum_zero, :] = self.alpha_      
        cond_prob_matrix /= cond_prob_matrix.sum(axis=1)[:,np.newaxis]

        return cond_prob_matrix
    
    def _m_step(self, X, cond_prob):
        ''' M-step of the algorithm
        Updates all the parameters with the new conditional probabilities
        
        Parameters
        ----------
        X: array-like, shape (n, p)
            data 
        cond_prob_matrix: array-like, shape (n, K)
             (cond_prob_matrix)_ik = P(Z_i=k|X_i=x_i)
    
        Returns
        ----------
        alpha_new: array-like, shape (n,)
            The new weights of each mixture components.
        mu_new: array-like, shape (n, p)
            The new mean of each mixture component.
        Sigma_new: array-like, shape (p, p)
            The new covariance of each mixture component.
        tau_new: array-like, shape (n, K)
            The collection of tau values.
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
                diff = X - mu_fixed_point             
                sq_maha = (np.dot(diff, inv_Sigma_fixed_point) * diff).sum(1) # multiple quadratic form
                
                tau_ite = sq_maha / p 
                tau_ite_sr = (sq_maha**(0.5))/p
                tau_ite = np.where(tau_ite<10**(-8) , 10**(-8),
                                   np.where(tau_ite>10**(8), 10**(8), tau_ite))
                tau_ite_sr = np.where(tau_ite_sr<10**(-8) , 10**(-8),
                                      np.where(tau_ite_sr>10**(8), 10**(8), tau_ite_sr))

                if self.version == 1 or self.version ==2:
                    Ck = (cond_prob[:, k]/tau_ite)/np.sum(cond_prob[:,k]/tau_ite)
                else: # 3 or 4
                    Ck = (cond_prob[:, k]/tau_ite_sr)/np.sum(cond_prob[:,k]/tau_ite_sr)

                mu_fixed_point_new = np.sum(np.multiply(X, Ck[:, np.newaxis]), 0)
                
                if self.version == 2 or self.version == 4: # if usig new estim, update denominator
                        
                    diff = X - mu_fixed_point_new             
                    sq_maha = (np.dot(diff, inv_Sigma_fixed_point) * diff).sum(1) # multiple quadratic form
                    tau_ite = sq_maha / p 
                    tau_ite_sr = (sq_maha**(0.5))/p
                    tau_ite = np.where(tau_ite<10**(-8) , 10**(-8),
                                       np.where(tau_ite>10**(8), 10**(8), tau_ite))
                    tau_ite_sr = np.where(tau_ite_sr<10**(-8) , 10**(-8),
                                          np.where(tau_ite_sr>10**(8), 10**(8), tau_ite_sr))
                    
                if self.version==1:
                    
                    diff = X - mu_fixed_point
                    Sigma_fixed_point_new = np.dot(cond_prob[:, k]/tau_ite * diff.T, diff) / (n * alpha_new[k])
                    Sigma_fixed_point_new *= p / np.trace(Sigma_fixed_point_new)
                    
                if self.version==2:
                    
                    diff = X - mu_fixed_point_new
                    Sigma_fixed_point_new = np.dot(cond_prob[:, k]/tau_ite * diff.T, diff) / (n * alpha_new[k])
                    Sigma_fixed_point_new *= p / np.trace(Sigma_fixed_point_new)
                    
                if self.version==3:
                    
                    diff = X - mu_fixed_point
                    Sigma_fixed_point_new = np.dot(cond_prob[:, k]/tau_ite_sr * diff.T, diff) / (n * alpha_new[k])
                    Sigma_fixed_point_new *= p / np.trace(Sigma_fixed_point_new)

                if self.version==4: 
                    
                    diff = X - mu_fixed_point_new
                    Sigma_fixed_point_new = np.dot(cond_prob[:, k]/tau_ite_sr * diff.T, diff) / (n * alpha_new[k])
                    Sigma_fixed_point_new *= p / np.trace(Sigma_fixed_point_new)

                convergence_fp = True
                convergence_fp = convergence_fp and (math.sqrt(np.inner(mu_fixed_point - mu_fixed_point_new, mu_fixed_point - mu_fixed_point_new)/p) < 10**(-5))
                convergence_fp = convergence_fp and (np.linalg.norm(Sigma_fixed_point_new-Sigma_fixed_point, ord='fro')/p) < 10**(-5)

                mu_fixed_point = mu_fixed_point_new.copy()
                Sigma_fixed_point = Sigma_fixed_point_new.copy() 

                ite_fp += 1

            mu_new[k] = mu_fixed_point
            Sigma_new[k] = Sigma_fixed_point 

            # UPDATE tau
            diff = X - mu_new[k]
            tau_new[:, k] = (np.dot(diff, np.linalg.inv(Sigma_new[k])) * diff).sum(1) / p
            tau_new[:, k] = np.where(tau_new[:, k] < 10**(-12) , 10**(-12),
                                     np.where(tau_new[:, k] > 10**(12), 10**(12), tau_new[:, k]))

        return alpha_new, mu_new, Sigma_new, tau_new
    
    def fit(self, X):
        ''' Fit the data to the model running the F-EM algorithm
        
        Parameters
        ----------
        X: array-like, shape (n, p)
            data 
    
        Returns
        ----------
        self
        '''
        
        n, p = X.shape
        
        self._initialize(X)

        convergence = False

        ite = 0
        
        while not(convergence) and  ite < self.max_iter:

            # Compute conditional probabilities:
            cond_prob = self._e_step(X)

            # Update estimators:
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
        
        # Outlier rejection 
        
        outlierness = np.zeros((n, )).astype(bool)
        
        if self.thres is None :
            self.thres = 0.05         
        thres = chi2.ppf(1 - self.thres, p)
            
        for k in range(self.K):
            
            data_cluster = X[self.labels_ == k,:]
            diff_cluster = data_cluster - self.mu_[k]
            sig_cluster = np.mean(diff_cluster * diff_cluster) 
            maha_cluster = (np.dot(diff_cluster, np.linalg.inv(self.Sigma_[k])) * diff_cluster).sum(1) / sig_cluster
            outlierness[self.labels_ == k] = (maha_cluster >  thres)
            
        self.labels_[outlierness] = -1
        
        self.labels_ = self.labels_.astype(str)
        
        return(self)
    
    def predict(self, Xnew, thres = None):
        
        n, p = Xnew.shape
        
        cond_prob_matrix = np.zeros((n, self.K))
    
        for k in range(self.K):
            
            psd = _PSD(self.Sigma_[k])
            prec_U, logdet = psd.U, psd.log_pdet
            diff = Xnew - self.mu_[k]
            sig = np.mean(diff * diff) 
            maha = (np.dot(diff, np.linalg.inv(self.Sigma_[k])) * diff).sum(1) 
            logdensity = -0.5 * (logdet + maha)            
            cond_prob_matrix[:, k] = np.exp(logdensity)  * self.alpha_[k]            
        
        sum_row = np.sum(cond_prob_matrix, axis = 1) 
        bool_sum_zero = (sum_row == 0)
        
        cond_prob_matrix[bool_sum_zero, :] = self.alpha_      
        cond_prob_matrix /= cond_prob_matrix.sum(axis=1)[:,np.newaxis]
        
        new_labels = np.array([i for i in np.argmax(cond_prob_matrix, axis=1)])
        
        outlierness = np.zeros((n, )).astype(bool)
        
        if thres is None :
            thres = self.thres           
        thres = chi2.ppf(1 - thres, p)
            
        for k in range(self.K):
            
            data_cluster = Xnew[new_labels == k,:]
            diff_cluster = data_cluster - self.mu_[k]
            sig_cluster = np.mean(diff_cluster * diff_cluster) 
            maha_cluster = (np.dot(diff_cluster, np.linalg.inv(self.Sigma_[k])) * diff_cluster).sum(1) / sig_cluster
            outlierness[new_labels == k] = (maha_cluster >  thres)
            
        new_labels[outlierness] = -1
        
        new_labels = new_labels.astype(str)
        
        return(new_labels)

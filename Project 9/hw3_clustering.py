import numpy as np
import sys
from scipy.spatial import distance
from scipy.stats import multivariate_normal

X = np.genfromtxt(sys.argv[1], delimiter = ",")


def KMeans(data):
    
    n_rows = data.shape[0]
    K = 5
    initial_centroids = data[np.random.choice(n_rows, K, replace=False), :]
    for i in range(10):
        cluster_list = []
        for m in range(n_rows):
            cluster_no = np.argmin([distance.euclidean(data[m],initial_centroids[j]) for j in range(K)])
            cluster_list.append(cluster_no)
        
        for k in range(K):
            indices = [p for p, x in enumerate(cluster_list) if x == k]
            initial_centroids[k] = np.mean(data[indices],axis = 0)
           
        filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
        np.savetxt(filename, initial_centroids, delimiter=",")
        
KMeans(X)


def EMGMM(data):
    
    n_rows = data.shape[0]
    n_cols = data.shape[1]
    
    #number of clusters 
    K = 5
    
    #initializing the centroids/clusters
    mu = data[np.random.choice(n_rows, K, replace=False), :]
    
    #covariance for each cluster
    sigma = [np.eye(n_cols) for i in range(K)]
    
    #pi - apriori uniform distribution
    pi = np.ones(K)/K
    
    #intializing phi
    phi = np.zeros((n_rows,K))
    for itr in range(10):
        
        print("Iteration number "+str(itr+1))
        
        for i in range(n_rows):
            
            #This is the normalizing constant (denominator) used while calculating phi(k)
            norm_constant = np.sum([pi[k] * multivariate_normal.pdf(data[i],mean=mu[k],cov=sigma[k],allow_singular=True) for k in range(K)])
            
            if norm_constant == 0:
                phi[i] = pi/K
            else:
                phi[i] = [(pi[k] * multivariate_normal.pdf(data[i],mean=mu[k],cov=sigma[k],allow_singular=True))/norm_constant for k in range(K)]
            
        
        for k in range(K):
            
            nk = np.sum(phi[:,k])
            
            pi[k] = nk/n_rows
            
            if nk == 0:
                mu[k] = data[np.random.choice(n_rows, 1, replace=False), :]
                sigma[k] = np.eye(n_cols)
            else:
                mu[k] = np.sum(data*phi[:,k].reshape(n_rows,1),axis=0)/nk
                cov_sum = np.zeros((n_cols,n_cols))
                
                for i in range(n_rows):
                    centered_data = data[i] - mu[k] 
                    cov_sum += phi[i,k]*np.outer(centered_data,centered_data)
                
                sigma[k] = cov_sum/nk
            
    
        filename = "pi-" + str(itr+1) + ".csv" 
        np.savetxt(filename, pi, delimiter=",") 
        filename = "mu-" + str(itr+1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration
            
        for j in range(K): #k is the number of clusters 
            filename = "Sigma-" + str(j+1) + "-" + str(itr+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, sigma[j], delimiter=",")
        
        
